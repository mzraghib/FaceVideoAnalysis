import numpy as np
import os
import pandas as pd
import helper

    
#set variables manually
feat_dir  = 'all_labels.csv'
label_folder = 'train_files/'
column_title = 'emotionAfter'
valueOfInterest = 'Interest'
k = 5
datafiles = 110

   
def RF_Classifier(X,y, test_X, y_true):
    from sklearn.ensemble import RandomForestClassifier 
    
    regr = RandomForestClassifier(max_depth=2, random_state=0)    
    regr.fit(X, y)
    
   
    y_pred = np.zeros(len(y_true))
    
    y_pred = regr.predict(test_X)    
    
    return y_pred    

    
def train(X, y, k, kFoldIndeces):
    resultTable = {}
    
    X_train = []
    for trial in range(k):
        X_train = X[kFoldIndeces[trial][0]]
        X_test = X[kFoldIndeces[trial][1]]
        
        y_train = y[kFoldIndeces[trial][0]]
        y_test = y[kFoldIndeces[trial][1]]
        print(y_test)
    
        y_pred =  RF_Classifier(X_train,y_train, X_test, y_test)
        resultTable[trial] = [y_test, y_pred]
    
    return resultTable

def getMetrics(resultTable,k,datafiles):
    from sklearn.metrics import confusion_matrix    
    
    scores = []
    confusionMatrices = []
    for i in range(0,k):
        score = helper.accuracyScore(resultTable[i][0], resultTable[i][1])
        confusionMatrix = confusion_matrix(resultTable[i][0], resultTable[i][1])
        confusionMatrices.append(confusionMatrix)
        scores.append(score)
    
    meanScore = round(float(sum(scores)/len(scores)),3) * 100 # accuracy in percentage
    
#    norm_conf_matrix = np.around((sum(confusionMatrices)/datafiles),decimals =1)
    norm_conf_matrix = sum(confusionMatrices)/datafiles
        
    return meanScore, norm_conf_matrix
    
    
def main():
       
     #train
    X,y = helper.features(feat_dir, label_folder, column_title)
    y = y.values.flatten().tolist()
    X,y = helper.shuffle(X,y)
    
    new_y = [0]*len(y)

    new_y = helper.binarize(y,valueOfInterest)
    y = new_y
    
    kFoldIndeces = helper.TrainTestSplit(X,y,k)
     
    resultTable = train(X, np.array(y), k, kFoldIndeces)
    meanScore, norm_conf_matrix = getMetrics(resultTable,k,datafiles)
    
    print("accuracy = ", meanScore)

    return X, y, resultTable, meanScore, norm_conf_matrix
       
if __name__ == '__main__':
    X, y, resultTable,meanScore, norm_conf_matrix = main()    




    