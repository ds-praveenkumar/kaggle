import pandas as pd
import os
from pathlib import Path

def load_csv_data(data_path):
    """ Reads csv as daraframe from input folder
        Author: PROKAGGLER22 (https://www.kaggle.com/prokaggler22)
    ARGS:
        data_path(str): input data folder
    RETURN:
        data_dict(dictionary): dictonary containing keys as file name and values as data folder
    """
    data_dict = {}
    print( 'Data source folder:', data_path)
    for folder, files ,name in os.walk( data_path ):
        for file in name:
            try:
                if file.split('.')[-1] == 'csv':
                    csv_path = os.path.join( folder, file )
                    data_dict[file] = pd.read_csv( csv_path )
                    print( 'file path:', csv_path)
            except Execption as e:
                print( e )
    print( 'No. of csv files loaded:' ,len(data_dict))
    return data_dict

if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'novozymes-enzyme-stability-prediction')
    data_dict = load_csv_data(path)
    print(data_dict)