import torch
import torch.nn.functional as F
import logging
import os
from video_dataset import VideoDataset
from train import CLIPTrainer
from model import clip_vit_base_patch32_multimodal_adapter12x384
from utils import text_prompt, load_word_index_mapping, get_masked_sample
from tqdm import tqdm

def setup_logging(save_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_dir, 'eval.log')),
            logging.StreamHandler()
        ]
    )

class CLIPEvaluator:
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
        setup_logging(config['save_dir'])

        # Create model
        self.model = clip_vit_base_patch32_multimodal_adapter12x384(
            num_classes=config['num_classes'],
            num_frames=config['num_frames'],
            mlm_head_len=self.mlm_head_len
        ).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(config['save_dir'], 'checkpoint_epoch_11.pt'))['model_state_dict'])

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

    def evaluate(self, dataset):
        self.model.eval()
        total_acc1 = 0
        total_acc5 = 0

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        progress_bar = tqdm(data_loader, desc=f'----Evaluating----')
        for batch_idx, (images, labels) in enumerate(progress_bar):
            if len(images.shape) == 6:
                images = images.squeeze()
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
            
            # Update progress bar
            progress_bar.set_postfix({
                'acc1': f'{acc1:.2f}%',
                'acc5': f'{acc5:.2f}%'
            })

            total_acc1 += acc1
            total_acc5 += acc5
            avg_acc1 = total_acc1 / (batch_idx+1)
            avg_acc5 = total_acc5 / (batch_idx+1)
            logging.info(f'Realtime Acc@1: {avg_acc1:.2f}%, Acc@5: {avg_acc5:.2f}%')

        final_avg_acc1 = total_acc1 / (len(dataset)/self.config['batch_size'])
        final_avg_acc5 = total_acc5 / (len(dataset)/self.config['batch_size'])
        logging.info(f'Final Acc@1: {final_avg_acc1:.2f}%, Acc@5: {final_avg_acc5:.2f}%')
        return final_avg_acc1, final_avg_acc5

def main():
    # Load model and data
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
        'batch_size': 2,
        'masked_rate': 0,  # MLM masking rate
        'mlm_labels_path': 'k400_mlm_lables.txt',
        'classes_path': 'kinetics_400_labels.csv', # kinetics_400_labels
    }
    
    # Initialize evaluator
    evaluator = CLIPEvaluator(config)
    
    # Load validation dataset
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

    # Evaluate the model
    avg_acc1, avg_acc5 = evaluator.evaluate(val_dataset)

    # Report accuracy
    logging.info(f'Validation Acc@1: {avg_acc1:.2f}%, Acc@5: {avg_acc5:.2f}%')


if __name__ == "__main__":
    main()