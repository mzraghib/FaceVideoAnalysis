import os
import json
import pickle
import numpy as np
from statistics import mean
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


"""
Read .pkl file of labels for Chalearn dataset and create 
label vector Y of format: 'video_name.mp4': 1
"""
def create_label_vector_Chalearn(label_path = 'annotation_validation.pkl'):
    
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
create X and Y matrices -> 2 json files for train data

"""


def createY_train(label_list):
   
    video_labels = create_label_vector_Chalearn('annotation_training.pkl')   
    
    Y = []    
    for key in label_list[0]: 
        Y.append(video_labels[str(key)+'.mp4'])

    
    return np.array(Y)

def createY_valid(label_list):
   
    video_labels = create_label_vector_Chalearn('annotation_validation.pkl')   
    
    Y = []    
    for key in label_list[0]: 
        Y.append(video_labels[str(key)+'.mp4'])

    
    return np.array(Y)
    
def createY_test(label_list):
   
    video_labels = create_label_vector_Chalearn('annotation_test.pkl')   
    
    Y = []    
    for key in label_list[0]: 
        Y.append(video_labels[str(key)+'.mp4'])

    
    return np.array(Y)
"""
Perform PCA for dimentionality reduction

Returns new X_train, X_test
"""
def peform_PCA(X_train, X_test, retained_var = 0.95):
    # Choose the minimum number of principal components such that 95% of the variance is retained (if retained_var = 0.95).

    # Make an instance of the Model
    pca = PCA(retained_var)

    # Fit PCA on the training set only
    pca.fit(X_train)

    # Apply the mapping (transform) to both the training set and the test set
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)


    print('number of components = ', pca.n_components_)

    plt.figure(1, figsize=(6, 5))
    plt.clf()
    plt.plot(pca.explained_variance_ratio_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_ratio_')
    plt.show()    

    return X_train_pca, X_test_pca