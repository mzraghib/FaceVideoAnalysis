from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from keras import optimizers
import vggface_helper
import scipy.misc
import numpy as np
import os
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model

#test

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

train_dir = '/scratch/ajjendj/BIMODALDEEPREGRESSION/INTERVIEW/Video/Train/'
test_dir = '/scratch/ajjendj/BIMODALDEEPREGRESSION/INTERVIEW/Video/Test/'
validation_dir = '/scratch/ajjendj/BIMODALDEEPREGRESSION/INTERVIEW/Video/Validation/'
labels_dir = '/scratch/ajjendj/BIMODALDEEPREGRESSION/INTERVIEW/Video/'
train_label_path = labels_dir + 'annotation_training.pkl'
valid_label_path = labels_dir + 'annotation_validation.pkl'
test_label_path = labels_dir + 'annotation_test.pkl'
train_txt_file = 'train_paths.txt'
valid_txt_file = 'validation_paths.txt'

numClasses = 1
hidden_dim = 512 #???
image_shape = (224,224)
batch_size  = 100
epochs = 1

train_samples = 0
validation_samples = 0

"""for debugging"""
def debugInfo(model):
    # Check the trainable status of the individual layers
    for layer in model.layers:
        print(layer, layer.trainable)

    # print summary / status
    model.summary()

    # Check the trainable status of the individual layers
    for layer in model.layers:
        print(layer, layer.trainable)

  

"""
    Create batches of training data --> only 100 frames per video considered
    :param batch_size: Batch Size
    :return: Batches of training data X,Y
   
"""    
def get_batches_fn_train(batch_size,label_path,paths_txt_file,image_shape = (224,224)):

    #load all image paths and dict of labels 
    image_paths, labels = vggface_helper.CreateXY(label_path,paths_txt_file)  
    
    for batch_i in range(0, len(image_paths), batch_size):
        X = []
        y = []
        for image_file in image_paths[0][batch_i:batch_i+batch_size]:
            ##load image and resize
            image = scipy.misc.imresize(scipy.misc.imread(image_file[:-1]), image_shape)
            X.append(image)
            
            #video name that is common for a batch of images
            vid_name = image_file[61:76] + '.mp4'      
            y.append(labels[vid_name]) #add the label for the image           

        yield np.array(X), np.array(y)


def get_batches_fn_valid(batch_size,label_path,paths_txt_file,image_shape = (224,224)):

    #load all image paths and dict of labels 
    image_paths, labels = vggface_helper.CreateXY(label_path,paths_txt_file)  
    
    for batch_i in range(0, len(image_paths), batch_size):
        X = []
        y = []
        for image_file in image_paths[0][batch_i:batch_i+batch_size]:
            ##load image and resize
            image = scipy.misc.imresize(scipy.misc.imread(image_file[:-1]), image_shape)
            X.append(image)
            
            #video name that is common for a batch of images
            vid_name = image_file[66:81] + '.mp4'      
            y.append(labels[vid_name]) #add the label for the image           

        yield np.array(X), np.array(y)
""" Create custom model """

#load pretrained model
#vggface = VGGFace(model='vgg16')

#create model
model = VGGFace(include_top=False, input_shape=(224, 224, 3))

# Freeze the layers except the last 4 layers
for layer in model.layers[:-4]:
    layer.trainable = False
print("1. model loaded")
#Adding custom Layers 
last_layer = model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(numClasses, activation='softmax', name='fc8')(x)

# creating the final model 
model_final = Model(model.input, out)
#model_f = Model(model.input, out)
#model_final = multi_gpu_model(model_f,gpus =1)
print("2. completed final custom  model")
# print some info about the model
#debugInfo(model_final)

""" Setup data generators """
train_generator = get_batches_fn_train(batch_size,train_label_path,train_txt_file,image_shape)
validation_generator = get_batches_fn_valid(batch_size,valid_label_path,valid_txt_file,image_shape)
print("3. finished setting up data generators")
""" Train the custom model """

# compile/configure the model 
##this includes: specifying loss fn, evaluation metrics,create model graph etc...
sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#model_final.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model_final.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])


# Train the model 
with tf.device('/gpu:1'):
	history = model_final.fit_generator(
		                train_generator,
		                samples_per_epoch = 2500000//batch_size,
		                epochs = epochs,
		                validation_data = validation_generator,
		                validation_steps = 800000//batch_size,
		                verbose=1)

# Save the model
model_final.save('vggface_custom_zuhayr.h5')
print("4. model saved")



# 10. Evaluate model on test data
#score = model_final.evaluate(X_test, Y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])





