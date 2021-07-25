import pandas as pd
from config import Config

class CommonlitreadabilityprizeDataset:
    """ Data set for commonlitreadabilityprize"""
    
    def __init__(self, path):
        self.data_dir = path
        self.df_dict = {}
        self.df_all = pd.DataFrame()

    def read_csv(self):
        """ Reades .csv files from folder """
        for file in self.data_dir.glob("*.csv"):
            df_name = str(file.name).split('.')[0]
            df_ = pd.read_csv(file)
            self.df_dict[df_name] = df_
        print(f"No. of files read: {len(self.df_dict)}")



if __name__ == "__main__":
    cl = CommonlitreadabilityprizeDataset( Config.data )
    cl.read_csv( )