import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer,\
         get_linear_schedule_with_warmup,TrainingArguments, Trainer
from transformers import BertModel
import torch.nn as nn
from torch.optim import Adam
from config import Config


class PatentModel(nn.Module):
    def __init__( self, dropout):
        super(PatentModel, self).__init__()
        self.bert = BertModel.from_pretrained( Config.model)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(1024,5)
        self.relu = nn.ReLU()
        
    def forward( self, input_id, mask):
        _, pooled_data = self.bert(  input_ids= input_id, attention_mask=mask,return_dict=False )
        dropuout_output = self.dropout( pooled_data )
        linear_output = self.linear( dropuout_output)
        final_layer = self.relu( linear_output )
        return final_layer