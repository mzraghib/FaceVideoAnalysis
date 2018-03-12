import pickle
import numpy as np
import pandas as pd


def CreateXY(label_path, paths_txt_file):    
    with open(label_path, 'rb',) as f:
        data = pickle.load(f,encoding='latin1')
       
    # binarize all values         
    for traits in data.keys():
        for key,traits_values in data[traits].items():
            data[traits][key] = (0, 1)[traits_values < 0.5] # 0 if number < 0.5 else 1


    # return table for 1 chosen trait {video name : binary score }
    Table = data['agreeableness']

    #load paths for all images into dataframe
    f = open(paths_txt_file, 'r')
    x = f.readlines()
    df = pd.DataFrame(np.array(x))

     

        
    return df, Table
    
        
    
