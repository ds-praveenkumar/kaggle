import torch
from data_loader import *
from transformers import AutoModel, AutoConfig, AutoTokenizer,\
         get_linear_schedule_with_warmup,TrainingArguments, Trainer
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained( Config.model)   

def train_bert( model, train, val, lr=1e-6, epochs=3):
    print( 'train started... ')
    train_patent_ds= PatentTrainDataset(
                        text_input= train.text_input,
                        labels = train.label,
                        tokenizer = tokenizer
    )

    val_patent_ds = PatentTrainDataset(
                        text_input= val.text_input,
                        labels = val.label,
                        tokenizer = tokenizer
    )
    train_dl = DataLoader( train_patent_ds, batch_size=Config.batch_size, shuffle=True)
    val_dl = DataLoader( val_patent_ds, batch_size=Config.batch_size, )
       
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam( model.parameters(), lr = lr)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        
    for epoch_num in range(epochs):
        print( 'Epochs =>', epoch_num) 
        total_acc_train = 0
        total_loss_train = 0
        for item in tqdm(train_dl):
            train_label = item['labels'].to(device)
            mask = item['masks'].to(device)
            input_id = item['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
                
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

        with torch.no_grad():

            for item in val_dl:

                val_label = item['labels'].to(device)
                mask = item['masks'].to(device)
                input_id = item['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                    
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        save_path =  'bert.pt'  #f'bert_{epoch_num}.pt'
        torch.save({
            'epoch': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
#             'loss': LOSS,
            }, save_path)    
        print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')
        print( 'model saved to =>', save_path)

    