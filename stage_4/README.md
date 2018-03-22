# Stage 4 
## Classifying ??  labels by transfer learning. A pretrained resnext model (on the Kinetics dataset) of a 3D CNN that is fine-tuned using the ?? dataset


### main_stage4.py:
 Extrtact model and pretrain

### stage4helper.py
Helper functions




## Note: 
- in progress


## Instructions

### 1
clone the repo from reference 1

Copy stage 4 code to the directory containing 'main.py'. 


### 2
clone the repo form reference 2

See instructions in 'https://github.com/kenshohara/3D-ResNets-PyTorch' for performing the following:

* Convert the student videos from avi to jpg files using utils/video_jpg.py

* Generate fps files using utils/fps.py


## Reference

* For extracting the pretrained model and inference
1) https://github.com/mzraghib/video-classification-3d-cnn-pytorch


* For training 
2) https://github.com/kenshohara/3D-ResNets-PyTorch
