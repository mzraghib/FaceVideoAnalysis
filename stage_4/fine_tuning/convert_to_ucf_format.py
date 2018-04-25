"""
Arrange chalearn dataset and labels to UCF-101 format

"""
import shutil
import os
import pickle

curr_dir = os.getcwd()
source = curr_dir + '/train/'
pos_dest = curr_dir + '/train_pos'
neg_dest = curr_dir +'/train_neg'

txt_file = '/annotations/trainlist01.txt'

files = os.listdir(source)


"""
Read .pkl file of labels for Chalearn dataset and create 
label vector Y of format: 'video_name.mp4': 1
"""
def create_label_vector_Chalearn(label_path = 'annotation_training.pkl'):
    
    with open(label_path, 'rb',) as f:
        data = pickle.load(f,encoding='latin1')           
       
    # binarize all values         
    for traits in data.keys():
        for key,traits_values in data[traits].items():
            data[traits][key] = (0, 1)[traits_values < 0.5] # 0 if number < 0.5 else 1

    # return table for 1 chosen trait {video name : binary score }
    Table = data['extraversion'] # highest mean label of 0.566, next is agreeableness at 0.52, rest <0.5
        
    return Table

    
def separate_files_into_diff_directories(labels):
    for f in files:
        vid = source + f

        if (labels[f] == 1):
            shutil.move(vid, pos_dest)
        else:
            shutil.move(vid, neg_dest)
            
            
""" Run AFTER converting dataset to UCF-101 directory format"""
def create_label_txt_file(labels, txt_file):
    directories = ['/train_pos', '/train_neg']
    with open(txt_file, 'w') as txt:
        for d in directories:        
            for f in os.listdir(curr_dir + '/ChalearnFT' + d):
                entry = d[1:] + '/' + str(f) + ' ' + str(labels[f]+1) + '\n' # added 1 to labels[f] to keep positive
                txt.write(entry)
            
        
        
if __name__ == "__main__" :
    
    labels = create_label_vector_Chalearn('annotation_training.pkl')
    create_label_txt_file(labels, txt_file)