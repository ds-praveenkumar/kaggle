from torch.utils.data import Dataset, DataLoader
import torch 
import random
import os
import numpy as np
import pandas as pd
from config import Config

print( 'torch version => ',torch.__version__)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device='cpu'

print('device selected =>', torch.cuda.get_device_name(0) )


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print( 'seed set to =>', seed)

class PatentTrainDataset(Dataset):
    def __init__( self, text_input, labels, tokenizer):
        self.text_input = text_input
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len( self.text_input)
        
    def __getitem__(self, idx):
        
        text_data = self.tokenizer.encode_plus(
            self.text_input[ idx ],
            add_special_tokens = True,
            pad_to_max_length = True,
            return_attention_mask = True,
            max_length = Config.max_len,
        )
        input_ids =text_data[ 'input_ids' ]
        masks = text_data['attention_mask']
        labels = self.labels[ idx ]
        return {
            'input_ids': torch.tensor( input_ids, dtype=torch.long),
            'labels': torch.tensor( labels, dtype=torch.long),
            'masks': torch.tensor( masks, dtype=torch.long )
        }



if __name__=='__main__':
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv( 'train.csv')
    sub_df = pd.read_csv( 'sample_submission.csv')
    print( 'data read completed ...!')
    seed_everything(42)
    score_map = dict(zip( range(5), ['0.00', '0.25', '0.50', '0.75', '1.00']))
    inverse_score_map = dict(zip( [0.00, 0.25, 0.50, 0.75, 1.00],range(5) ))
    train = pd.DataFrame()
    train['text_input'] = train_df['anchor']+ '[sep]' + train_df['target'] + '[sep]' + train_df['context']
    train['label'] = train_df['score'].map( inverse_score_map)


