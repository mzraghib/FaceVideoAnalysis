# Stage 4 
Classification for Chalearn label 'extraversion' via feature extraction, and via transfer learning. A pretrained 3D ResNeXt-101 model (on the Kinetics dataset) 

### main_stage4.py:
 Extrtact model and pretrain

### stage4helper.py
Helper functions


## Instructions

### 1
clone the repo from reference 1

Copy stage 4 code to the directory containing 'main.py'. 


### 2
clone the repo form reference 2

See instructions in 'https://github.com/kenshohara/3D-ResNets-PyTorch' for performing the following:

* Convert the student videos from avi to jpg files using utils/video_jpg.py

* Generate fps files using utils/fps.py


### 3 Fine Tuning
### 3.1) Convert Chalearn dataset to UCF-101 format for fine tuning  (use modified 3D-ResNets-PyTorch repo) using convert_to_ucf_format.py

#### consider the following directories after conversion:

avi_video_directory = /scratch/mzraghib/stage4/chalearn/fine_tune/ChalearnFT
jpg_video_directory = /scratch/mzraghib/stage4/chalearn/fine_tune/ChalearnFTjpg
annotation_dir_path = /scratch/mzraghib/stage4/chalearn/fine_tune/annotations

#### convert to jpeg	(changed '.avi' to '.mp4' in video_jpg_ucf101_hmdb51.py)
```
python utils/video_jpg_ucf101_hmdb51.py /scratch/mzraghib/stage4/chalearn/fine_tune/ChalearnFT /scratch/mzraghib/stage4/chalearn/fine_tune/ChalearnFTjpg
```

#### Generate n_frames
```
python utils/n_frames_ucf101_hmdb51.py /scratch/mzraghib/stage4/chalearn/fine_tune/ChalearnFTjpg
```

#### Generate annotation file in json format similar to ActivityNet (changed range to single loop in ucf101_json.py and changed one line to split at second '.')

```
python utils/ucf101_json.py /scratch/mzraghib/stage4/chalearn/fine_tune/annotations
```

### 3.2) Fine tuning conv5_x and fc layers 

``` 
cd /scratch/mzraghib/stage4/3D-ResNets-PyTorch 
```

``` 
python main.py --root_path /scratch/mzraghib/stage4/chalearn/fine_tune/ --video_path ChalearnFTjpg --annotation_path annotations/ucf101_01.json \
--result_path results --dataset ucf101 --n_classes 400 --n_finetune_classes 2 \
--pretrain_path models/resnext-101-kinetics.pth --ft_begin_index 4 \
--model resnext --model_depth 101 --resnet_shortcut B --batch_size 10 --n_threads 4 --checkpoint 1 
```
### 3.3) Extract features and scores:

``` 
python main.py --input ./input --video_root /scratch/mzraghib/stage4/chalearn/test \
 --output ./output_save_40_features.json --model /scratch/mzraghib/stage4/chalearn/fine_tune/results/save_40.pth \
 --model_name resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --mode feature
```

## Reference

* For extracting the pretrained model and inference
1) https://github.com/mzraghib/video-classification-3d-cnn-pytorch


* For training 
2) https://github.com/kenshohara/3D-ResNets-PyTorch
