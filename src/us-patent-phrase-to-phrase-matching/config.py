from torch import dropout


class Config:
    # data
    train_csv = '../input/us-patent-phrase-to-phrase-matching/train.csv'
    test_cssv = '../input/us-patent-phrase-to-phrase-matching/test.csv'
    sub_csv = '../input/us-patent-phrase-to-phrase-matching/sample_submission.csv'
    
    # model
    model = 'anferico/bert-for-patents'
    
    max_len = 32
    num_epoch = 2
    batch_size = 64
    epochs = 7
    lr = 1e-6
    dropout= 0.5
    
    train = False
    train_ration = 0.9