import numpy as np
import os
import pandas as pd


def features(feat_dir, label_folder):
    
    labels = pd.read_csv(feat_dir)
    
    
    
    # read all .txt files (video file features)
    all_files = os.listdir(label_folder)
    
    # 'sorted' doesn't work...not needed currently
    """ build X and y """
    X = np.zeros(shape=(len(all_files),431)) # change 431 to only number of required columns
    y = np.zeros(len(all_files))
    X_entry = np.zeros(431)
    
    i = 0
    for data_file in sorted(all_files):
        
        print("processing    " + data_file)
        
        # X    
        #TODO: Get rid of unneccessary columns from X
    
        Features = pd.read_csv(label_folder + data_file)
        
        j = 0
        for column in Features:
    #        X_entry = np.append(X_entry, Features[column][:].mean(axis=0))
            X_entry[j] =  Features[column][:].mean(axis=0)
            j += 1
            
        X[i] = X_entry
     
        
        # y    
        #TODO: requried index returned in list - fix later    
        y_index = labels.index[labels['Video_File_Name'] == data_file[:-4]].tolist() 
        y[i] = float(labels['Mastery'][y_index[0]])
        
        i += 1
        
        
    return X,y

    
    
def RF_regressor(X,y, test_X, y_true):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import accuracy_score
    
    
    print('fds')
    regr = RandomForestRegressor(max_depth=2, random_state=0)    
    regr.fit(X, y)
    
#    print(regr.feature_importances_)  

#    pd.DataFrame(test_X)
    
    y_pred = np.zeros(len(y_true))
    for i in range(0,len(test_X[0])):
        print(regr.predict(test_X[i]))
        y_pred[i] = regr.predict(test_X[i])    
    
    return accuracy_score(y_true, y_pred)    
    
    
def main():
     #train
     X,y = features('eureka_train_labels.csv', 'eureka_train/')
     
     #test
     test_X, y_true = features('eureka_test_labels.csv', 'eureka_test/')
     
     score = RF_regressor(X,y,test_X,y_true)

#     score = RF_regressor(pd.DataFrame(X),pd.DataFrame(y), \
#                                               pd.DataFrame(test_X), pd.DataFrame(y_true))
     
     print(score)
    
       
if __name__ == '__main__':
    main()    

    
    
    
    
    
    
    
    
    
    
    



    




    