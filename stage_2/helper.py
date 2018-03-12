import numpy as np
import os
import pandas as pd

"""
feat_dir = feature directory
label_folder = location of .txt files
column_title = title of column in y labels
"""
def features(feat_dir, label_folder, column_title):
    
    labels = pd.read_csv(feat_dir) 
    
    # read all .txt files (video file features)
    all_files = os.listdir(label_folder)
    
    # 'sorted' doesn't work...not needed currently
    """ build X and y """
    X = np.zeros(shape=(len(all_files),26*4))   # considering 26 features
    

    y = np.zeros(len(all_files))
    y = pd.DataFrame(y) #convert to dataframe to accept both strings and floats   
    
    X_entry = np.zeros(26*4)     # considering 26 features
    
    print("processing .txt files . . .")
    i = 0
    for data_file in sorted(all_files):
        # X      
        Features = pd.read_csv(label_folder + data_file)
        
        #drop useless features
        Features = Features.drop(['frame', ' timestamp', ' success', ' confidence', \
                               ' p_scale',' p_rx',' p_ry',' p_rz',' p_tx',' p_ty'], axis=1)
        
        Features = Features.drop([' pose_Rx',' pose_Ry',' pose_Rz' ], axis=1)        
        
        for ii in range(0,46):
            st = ' AU'+ str(ii) + "_c"
            st2 = ' AU0'+ str(ii) + "_c"
            if st in Features:
                Features = Features.drop([st], axis=1)  
            if st2 in Features:
                Features = Features.drop([st2], axis=1)             
            
        for ii in range(0,34):
            st = str(ii)
            Features = Features.drop([' p_'+ st], axis=1)  

        for ii in range(0,68):
            st = str(ii)
            Features = Features.drop([' x_'+ st,' y_'+ st,' X_'+ st,' Y_'+ st, ' Z_' + st], axis=1)           
            
        # add row to X
        j = 0
        for column in Features:
            X_entry[j] =  Features[column][:].mean(axis=0)
            X_entry[j+1] = Features[column][:].max(axis=0)
            X_entry[j+2] = Features[column][:].min(axis=0)
            X_entry[j+3] = Features[column][:].std(axis=0)

            
            j += 4
            
        X[i] = X_entry
     
        
        # add row (element) to y    
        y_index = labels.index[labels['Video_File_Name'] == data_file[:-4]].tolist() #returns single value in a list
        y.iloc[i] = labels[column_title][y_index[0]]
        
        i += 1
    print("finished processing " + str(i) + " .txt files")        
    return X,y  
    

""" 
    - returns a shuffled list of indeces of a given input feature vector 'X'    
    - use before splitting feature vector into train and test data    
"""
def shuffle(X, y):
    from sklearn.utils import shuffle
    X, y, = shuffle(X, y, random_state=0)
    
    return X, y,
    
"""
    - generates K_0 . . . K_n lists of arrays of indeces
        of an input vectors X,y for train and test data
    - k = number of bins
"""
def TrainTestSplit(X,y,k):
    
    from sklearn.model_selection import KFold
    
    
    kFoldIndeces = {}
    kf = KFold(n_splits=k)
    kf.get_n_splits(X)
    KFold(n_splits=2, random_state=0, shuffle=False) #shuffle=True not working....
    i = 0
    for train_index, test_index in kf.split(X):
        kFoldIndeces[i] = [train_index , test_index]
        i += 1
    
    return kFoldIndeces
    
    
def accuracyScore(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)
    
    
"""
    -converts input list of strings/ ints to a binary list of ints
    -sets values == 'value' to 1 and all other values to 0
"""
def binarize(y, valueOfInterest):
    new_y = [0]*len(y)
    for i in range(0,len(y)):
        if (y[i] == valueOfInterest):
            new_y[i] = 1  
 
    return new_y
    
def plot(y_true, y_pred):
    import matplotlib.pyplot as plt
    
    ax = plt.subplot(1,1,1)
    b1, = ax.plot(y_true)
    b2, = ax.plot(y_pred)
    plt.legend([b1, b2],['y_true', 'y_pred'])
    plt.ylabel('Mastery')
    
    plt.savefig('Random_forest.png')
    
    plt.show()

    plt.close()    