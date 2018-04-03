import os
import json
import pickle
import numpy as np
import copy
from statistics import mean
"""
The input file requires all the video file names for the 3D resnet. All
the files listed in the input file are processed with a single call to 
extract featues
""" 
def populateInputFile(file_dir = '/scratch/mzraghib/stage4/chalearn'):   
    with open("input", "w") as a:
        for path, subdirs, files in os.walk(file_dir):
           for filename in files:
             a.write(str(filename) + os.linesep) 


"""
Read .pkl file of labels for Chalearn dataset and create 
label vector Y
"""
def create_label_vector_Chalearn(label_path = None):
    if(label_path == None):
        label_path = '/scratch/mzraghib/stage4/chalearn/labels/annotation_training.pkl'   

    
    with open(label_path, 'rb',) as f:
        data = pickle.load(f,encoding='latin1')
           
       
    # binarize all values         
    for traits in data.keys():
        for key,traits_values in data[traits].items():
            data[traits][key] = (0, 1)[traits_values < 0.5] # 0 if number < 0.5 else 1

    # return table for 1 chosen trait {video name : binary score }
    Table = data['extraversion'] # highest mean label of 0.566, next is agreeableness at 0.52, rest <0.5
        
    return Table




"""
open json file with 512 dim features for each video
"""
def openJson(fileName = 'output.json'):
    f = open(fileName,'r')
    features = json.load(f)
    f.close()

    return features


def reduced_train_set(Y, vid_path = None):
    if(vid_path == None):
        vid_path = '/scratch/mzraghib/stage4/chalearn/dataset'   
      
    shortList = []
    for path, subdirs, files in os.walk(vid_path):
       for filename in files:  
           shortList.append(str(filename))


    newY = copy.deepcopy(Y)    
    
    for k,v in Y.items():
        if(k in shortList):
            continue
        else:
            del newY[k]
        
    return newY


"""
Parse the list of features from the output.json file containing a 512 dim feature
array for each 16 frame segment of a video.

"""
def parse_feature_list(features):
    X = {}
    
    for i in range(len(features)):
        video_name = features[i]['video']
        clips_dict = features[i]['clips']        
       
        combined = clips_dict[0]['features'] # inital velue to stack addtional colums to
        for j in range(1,len(clips_dict)):
            segment_feat = clips_dict[j]['features'] # why 2048 dim?
            combined = np.column_stack((combined,segment_feat))            
        X.update({video_name:combined})
        
    # take mean of each row to reduce dimentionality to  single 2048 vector
    newX = copy.deepcopy(X)    

    for key in X:
        mean_val = []
        for row in range(len(X[key])):
            temp = mean(X[key][row])
            mean_val.append(temp)
            

        newX.update({key:mean_val})
    return newX
    

def printMetrics(y_class, y_pred_class):
    from sklearn import metrics
    
    # bottom right -> TP, upper left -> TN, upper right -> FP, bottom left --> FN
    print (metrics.confusion_matrix(y_class,y_pred_class))
    
    # FP + FN / TP + TN + FP + FN
    print (metrics.accuracy_score(y_class,y_pred_class))
    
    
    # sensitivity
    # TP / TP + FN 
    print (metrics.recall_score(y_class,y_pred_class))
    
    # precision
    # TP / TP + FP
    print (metrics.precision_score(y_class,y_pred_class))
    
    # precision recall score
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_class, y_pred_class)
    
    print('Average precision-recall score: {0:0.2f}'.format(
          average_precision))
          
    # precision recall curve
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    
    precision, recall, _ = precision_recall_curve(y_class, y_pred_class)
    
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    
    
    # classification report
    from sklearn.metrics import classification_report
    print (classification_report(y_class,y_pred_class))
    
    
    # f1, f2 scores
    from sklearn.metrics import fbeta_score
    print ("f1 score" , fbeta_score(y_class,y_pred_class, 1))
    print ("f2 score" , fbeta_score(y_class,y_pred_class, 2))
    
"""
create X and Y matrices for use by different ML algorithms
"""
def createXY():
    features = openJson()
    temp_X= parse_feature_list(features)
    
    temp_Y = reduced_train_set(create_label_vector_Chalearn('annotation_training.pkl') , os.getcwd()+'\\dataset')          

    X = []
    Y = []    
    for key in temp_Y: 
        X.append(temp_X[key])
        Y.append(temp_Y[key])

    
    return np.array(X),np.array(Y)



    