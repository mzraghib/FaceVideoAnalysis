import numpy as np
import cv2
import os
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from os.path import expanduser
import math

home = expanduser("~")

curr_dir =  os.getcwd()
prev_dir = os.path.normpath(curr_dir + os.sep + os.pardir)

df = pd.read_excel(home + '\\Pictures\\DS_VIDEOS\\Millbury Sync Files\\problemHistorylogClasses1310&1342FORSYNCH.xlsx', sheetname='Sheet2')



""" crop video from original video and save """

def cropVideo(videoName, t1, t2, newName):
    ffmpeg_extract_subclip(videoName, t1, t2, targetname= curr_dir + str(newName) + ".mov")
    print(".")
    print(".")
    print("made := " + str(newName) + ".mov")

  


def grade10():
        
    originalVideoA1 = home + "\\Pictures\\DS_VIDEOS\\Millbury Sync Files\\Millbury7grade10_A1.mov"        
    originalVideoA2 = home + "\\Pictures\\DS_VIDEOS\\Millbury Sync Files\\Millbury7grade10_A2.mov"
    originalVideoA3 = home + "\\Pictures\\DS_VIDEOS\\Millbury Sync Files\\Millbury7grade10_A3.mov"        
   
    subclips_id = 0

    for i in range(3,22):  
        t1 = df['t1'][i]
        t2 =  df['t2'][i]
        if(math.isnan(t2) or math.isnan(t1)):
            print("continue")
            subclips_id += 1
            continue
        
        if (i <= 6 ): 
            newName = "Millbury7grade10_A1_" + str(subclips_id)
            cropVideo(originalVideoA1, t1, t2, newName)
            subclips_id += 1

        elif (6 < i < 15 ):
            newName = "Millbury7grade10_A2_" + str(subclips_id)
            cropVideo(originalVideoA2, t1, t2, newName)
            subclips_id += 1
        else:
            newName = "Millbury7grade10_A3_" + str(subclips_id)
            cropVideo(originalVideoA3, t1, t2, newName)
            subclips_id += 1

            
def grade4():
        
    originalVideoA1 = home + "\\Pictures\\DS_VIDEOS\\Millbury Sync Files\\Millbury7grade4_A1.mov"        
    subclips_id = 0
    for i in range(22, 38):  # 29 entries
        
        t1 = df['t1'][i]
        t2 =  df['t2'][i]

        print(subclips_id)
        if(math.isnan(t2) or math.isnan(t1)):
            print("continue1")
            subclips_id += 1
            continue
        
        if (subclips_id >20):
            print("continue2")
            subclips_id += 1
            continue
        else:
            newName = "Millbury7grade4_A1_" + str(subclips_id)
            cropVideo(originalVideoA1, t1, t2, newName)
            subclips_id += 1

def grade7():
        
    originalVideoA1 = home + "\\Pictures\\DS_VIDEOS\\Millbury Sync Files\\Millbury7grade7_A2.mov"        
    subclips_id = 0
    for i in range(73, len(df['w_student_main::userName'])):  # 34 entries
        
        t1 = df['t1'][i]
        t2 =  df['t2'][i]

        print(subclips_id)
        if(math.isnan(t2) or math.isnan(t1)):
            print("continue1")
            subclips_id += 1
            continue
        
        newName = "Millbury7grade7_A2_" + str(subclips_id)
        cropVideo(originalVideoA1, t1, t2, newName)
        subclips_id += 1
    
if __name__ == '__main__':
     grade4()


