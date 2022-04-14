from  data_loader import *
from train import *
from model import * 
from config import Config

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
    train_samples = int(train.shape[0] * Config.train_ration)
    train_data = train.iloc[:train_samples,:]
    val_data = train.iloc[train_samples:, :]
    val_data.reset_index(inplace=True)
    val_data.drop( ['index'], axis=1,inplace=True)
    print( 'train samples =>', len( train_data))
    print( 'validation samples =>', len(val_data))
    model = PatentModel(dropout=Config.dropout)
    train_bert(model, train_data, val_data, Config.lr, Config.epochs)