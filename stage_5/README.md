# INSTRUCTIONS

## 1) Folder stucture

    .
    ├── stage5_main.py             # Crop videos into segments
    ├── /videos_cropped            # output folder
    ├── /videos_uncropped          # input folder containing original videos
    ├── test_logs                  # summary of results
    └──  Classs1451ProblemHistory.xlsx   # Excel file with start and end time info

## 2) Place uncropped videos in /videos_uncropped directory

 * Prepend the session ID to the beginning of the video file name
  e.g - Webcam_wo_watermark_KameaM_4-19-2018.mp4  --> 69217_Webcam_wo_watermark_KameaM_4-19-2018.mp4
  
 * Update the start_times and begin_times dictionaries in the code manually. Refer to the comments in the code
  
## 3) run stage5_main.py
 ```
 python stage5_main.py
 ```
 
## 4) After completion, all cropped videos will be in the /videos_cropped folder.
Manually delete videos that are too long (e.g > 2 mins)  or short (e.g < 5s)
