# stage 3 - Classifying personality trait labels by transfer learning. A pretrained model of VGG-Face is fine-tuned using the ChaLearn dataset

##Note: 
- A single personality trait is chosen in vgg_face_helper.py. the default is set to 'aggreeableness'. 

- The .txt files containing all the paths to each frame can be found in the team google drive


## First execute :
python vgg_face.py

## Each epoch takes approximately 9 hours to complete, using a single K40 GPU. Therefore, train using one epoch. The program saved the model, which can be read and further trained till the required accuracy is achieved

## Test a model using the script:
python vgg_test.py path_to_model.md5


