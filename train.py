import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import clip_vit_base_patch32_multimodal_adapter12x384
from video_dataset import VideoDataset
from video_dataset.random_erasing import RandomErasing
import logging
import os
from tqdm import tqdm
from utils import text_prompt, load_word_index_mapping, get_masked_sample, gen_label
import math

def setup_logging(save_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

class CLIPTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load MLM label mapping
        self.mlm_labels_dict = load_word_index_mapping(config['mlm_labels_path'])
        self.mlm_head_len = len(self.mlm_labels_dict)
        # Get text prompts and class mappings
        self.text_tokens, self.num_text_aug, self.text_dict, self.classes_dict = text_prompt(
            config['classes_path'], 
            config['mlm_labels_path']
        )
        self.text_tokens = self.text_tokens.to(self.device)
        
        # Create model
        self.model = clip_vit_base_patch32_multimodal_adapter12x384(
            num_classes=config['num_classes'],
            num_frames=config['num_frames'],
            mlm_head_len=self.mlm_head_len
        ).to(self.device).train()
        # only train adapter layer  
        for k,v in self.model.named_parameters():
            if 'adapter' in k:
                v.requires_grad = True
                v.data = v.data.float()
            # if '12' in k:
            #     print(k)
            else:
                v.requires_grad = False
                
        # print the trainable parameters
        n_trainable_params = 0
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                print('Trainable param: %s, %s, %s' % (n, p.size(), p.dtype))
                n_trainable_params += p.numel()
        print('Total trainable params:', n_trainable_params, '(%.2f M)' % (n_trainable_params / 1000000))

        # Setup optimizer with different learning rates for different parameter groups
        params_with_decay, params_without_decay = [], []
        text_params_with_decay, text_params_without_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if '.bias' in n:
                if 'textad_' in n:
                    text_params_without_decay.append(p)
                else:
                    params_without_decay.append(p)
            else:
                if 'textad_' in n:
                    text_params_with_decay.append(p)
                else:
                    params_with_decay.append(p)
        
        self.optimizer = optim.AdamW(
            [
                {'params': params_with_decay, 'lr': config['learning_rate'], 'weight_decay': config['weight_decay']},
                {'params': text_params_with_decay, 'lr': config['learning_rate'] / 2, 'weight_decay': config['weight_decay']},
                {'params': params_without_decay, 'lr': config['learning_rate'], 'weight_decay': 0.},
                {'params': text_params_without_decay, 'lr': config['learning_rate'] / 2, 'weight_decay': 0.}
            ]
        )
        
        # Setup learning rate scheduler with warmup
        def lr_func(step):
            epoch = step / len(train_loader)
            if epoch < config['warmup_epochs']:
                return epoch / config['warmup_epochs']
            else:
                return 0.5 + 0.5 * math.cos((epoch - config['warmup_epochs']) / (config['num_epochs'] - config['warmup_epochs']) * math.pi)
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_func)
        
        self.scaler = torch.cuda.amp.GradScaler()  # Initialize the GradScaler for AMP
        
        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        setup_logging(self.save_dir)

    def get_text_and_mlm_labels(self, label):
        # Get text description for the class
        text = self.classes_dict[label.item()]
        
        # Generate masked tokens and MLM labels
        text_tokens, mlm_labels = get_masked_sample(
            text, 
            self.mlm_labels_dict, 
            masked_rate=self.config['masked_rate']
        )
        
        return torch.tensor(text_tokens).to(self.device), torch.tensor(mlm_labels).to(self.device)

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Get text tokens and MLM labels for each video
            text_tokens_list = []
            mlm_labels_list = []
            for label in labels:
                text_tokens, mlm_labels = self.get_text_and_mlm_labels(label)
                text_tokens_list.append(text_tokens)
                mlm_labels_list.append(mlm_labels)
            
            text_tokens = torch.stack(text_tokens_list)
            mlm_labels = torch.stack(mlm_labels_list)
            
            self.optimizer.zero_grad()
            
            # Use AMP for the forward and backward pass
            with torch.cuda.amp.autocast():
                # Forward pass
                logits_per_image, logits_per_text, image_embedding, \
                text_mlm_inputs, image_mlm_inputs, logits = self.model(images, text_tokens)
                
                # Calculate individual losses
                loss_fc = F.cross_entropy(logits, labels)
                
                mlm_output = self.model.compute_mlm(text_mlm_inputs, mlm_labels, image_mlm_inputs)
                mlm_loss = mlm_output['mlm_loss']
                
                ground_truth = torch.tensor(gen_label(labels), dtype=images.dtype, device=self.device)
                loss_imgs = F.cross_entropy(logits_per_image, ground_truth)
                loss_texts = F.cross_entropy(logits_per_text, ground_truth)
                contrastive_loss = (loss_imgs + loss_texts) / 2
                
                # Normalize text features
                text_features, _ = self.model.encode_text(self.text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity
                similarity = (100.0 * image_embedding @ text_features.T).softmax(dim=-1)
                acc1 = (similarity.topk(1, dim=-1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
                acc5 = (similarity.topk(5, dim=-1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
                loss_similarity = F.cross_entropy(similarity, labels)
                
                # Total loss
                total_loss = (contrastive_loss + loss_similarity) / 2 + loss_fc + mlm_loss / 10
            
            # Backward pass with AMP
            # self.scaler.scale(total_loss).backward()
            total_loss.backward()
            # Step the optimizer and update the learning rate scheduler 
            self.optimizer.step()
            self.scheduler.step()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            
            total_loss += total_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'cont_loss': f'{contrastive_loss.item():.4f}',
                'mlm_loss': f'{mlm_loss.item():.4f}',
                'acc1': f'{acc1:.2f}%',
                'acc5': f'{acc5:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss, acc1, acc5

    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            # 'scaler_state_dict': self.scaler.state_dict(),
            'loss': loss,
        }
        path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        logging.info(f'Checkpoint saved: {path}')

    def load_checkpoint(self, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f'Checkpoint loaded: {checkpoint_path}, starting from epoch {start_epoch}')
            return start_epoch
        else:
            logging.error(f'No checkpoint found at {checkpoint_path}')
            return 0

    def train(self, train_loader, val_loader=None, resume_from_checkpoint=None):
        logging.info("Starting training...")
        
        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint(resume_from_checkpoint)
        
        for epoch in range(start_epoch, self.config['num_epochs']):
            # Train for one epoch
            avg_loss, acc1, acc5 = self.train_epoch(train_loader, epoch)
            # Log the results
            logging.info(f'Epoch {epoch}: Average Loss = {avg_loss:.4f} , acc1 = {acc1:.2f}% , acc5 = {acc5:.2f}%')
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_frequency'] == 0:
                self.save_checkpoint(epoch, avg_loss)
            
            # Validation if provided
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f'Validation Loss = {val_loss:.4f}')  # Debugging print statement
                logging.info(f'Validation Loss = {val_loss:.4f}')

    def validate(self, val_loader):
        self.model.eval()

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device).squeeze()
                labels = labels.to(self.device)
                
            # Get text tokens and MLM labels
            text_tokens_list = []
            mlm_labels_list = []
            for label in labels:
                text_tokens, mlm_labels = self.get_text_and_mlm_labels(label)
                text_tokens_list.append(text_tokens)
                mlm_labels_list.append(mlm_labels)
            
            text_tokens = torch.stack(text_tokens_list)
            mlm_labels = torch.stack(mlm_labels_list)
            
            with torch.cuda.amp.autocast():
            
                logits_per_image, logits_per_text, image_embedding, \
                text_mlm_inputs, image_mlm_inputs, logits = self.model(images, text_tokens)
        
                # Normalize text features
                text_features, _ = self.model.encode_text(self.text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity
                similarity = (100.0 * image_embedding @ text_features.T).softmax(dim=-1)
                acc1 = (similarity.topk(1, dim=-1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
                acc5 = (similarity.topk(5, dim=-1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
            
            total_acc1 += acc1
            total_acc5 += acc5
            total_samples = len(val_loader)
            avg_acc1 = total_acc1 / total_samples * 100
            avg_acc5 = total_acc5 / total_samples * 100
        
        logging.info(f'Validation Acc@1: {avg_acc1:.2f}%, Acc@5: {avg_acc5:.2f}%')
        return avg_acc1, avg_acc5   

if __name__ == "__main__":
    # Training configuration
    config = {
        'num_classes': 400,  # Number of Kinetics-400 classes
        'num_frames': 8,     # Number of frames per video
        # 'mlm_head_len': 30522,  # BERT vocabulary size
        'learning_rate': 1e-4,
        'min_lr': 1e-5,
        'weight_decay': 0.01,
        'num_epochs': 12,
        'sampling_rate': 16,
        'num_temporal_views': 3,
        'num_spatial_views': 4,
        'warmup_epochs': 2,  # Number of warmup epochs
        'mlm_loss_weight': 1.0,
        'max_grad_norm': 1.0,
        'save_frequency': 1,
        'scale_range': (1.0, 1.15),
        'save_dir': 'checkpoints',
        'batch_size': 32,
        'masked_rate': 0.15,  # MLM masking rate
        'mlm_labels_path': 'k400_mlm_lables.txt',
        'classes_path': 'kinetics_400_labels.csv',  # kinetics_400_labels
    }
    
    # Initialize dataset with augmentations
    random_erasing = RandomErasing(
        probability=0.25,
        mode='pixel',
        max_count=1,
        num_splits=0,
        device='cpu'
    )

    train_dataset = VideoDataset(
        list_path='k400_train.txt',  # Replace with your train list path
        data_root=r'Kinetics-400\videos_train',            # Replace with your data root path
        num_frames=config['num_frames'],
        sampling_rate=config['sampling_rate'],
        spatial_size=224,
        random_sample=True,
        random_erasing=random_erasing,
        resize_type='random_short_side_scale_jitter',
        # auto_augment='rand-m7-n4-mstd0.5-inc1',
        scale_range=config['scale_range']
    )

    val_dataset = VideoDataset(
        list_path='k400_val.txt',  # Using the provided validation list
        data_root=r'Kinetics-400\videos_val',            # Replace with your data root path
        num_frames=config['num_frames'],
        sampling_rate=config['sampling_rate'],
        spatial_size=224,
        random_sample=False,
        # num_temporal_views=config['num_temporal_views'],
        # num_spatial_views=config['num_spatial_views']
        )
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,                                                                  
        pin_memory=True
    )
    
    # Initialize trainer
    trainer = CLIPTrainer(config)
    
    # Start training, optionally resuming from a checkpoint
    trainer.train(train_loader, val_loader=val_loader, resume_from_checkpoint='checkpoints/checkpoint_epoch_4.pt')
