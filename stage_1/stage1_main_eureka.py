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

df = pd.read_excel('problemHistorylogClasses1310&1342-someTestuUsersRemoved.xlsx', sheetname='Sheet1')



""" crop video from original video and save """

def cropVideo(videoName, t1, t2, newName):
    ffmpeg_extract_subclip(videoName, t1, t2, targetname= curr_dir + str(newName) + ".mp4")
    print(".")
    print(".")
    print("made := " + str(newName) + ".mov")

  


def eureka51():
        
    originalVideoA1 = home + "\\Pictures\\DS_VIDEOS\\Camp Eureka - GRIT - July 13, 2017\\Eureka51.mp4"        
    subclips_id = 0
    for i in range(1492, 1507):  # 15 entries
        
        t1 = df['t1'][i]
        t2 =  df['t2'][i]

        print(subclips_id)
        if(math.isnan(t2) or math.isnan(t1)):
            print("continue1")
            subclips_id += 1
            continue
        
        newName = "Eureka51_" + str(subclips_id)
        cropVideo(originalVideoA1, t1, t2, newName)
        subclips_id += 1
        
def eureka54():
        
    originalVideoA1 = home + "\\Pictures\\DS_VIDEOS\\Camp Eureka - GRIT - July 13, 2017\\Eureka54.mp4"        
    subclips_id = 0
    for i in range(1556, 1581):  # 25 entries
        
        t1 = df['t1'][i]
        t2 =  df['t2'][i]

        print(subclips_id)
        if(math.isnan(t2) or math.isnan(t1)):
            print("continue1")
            subclips_id += 1
            continue
        
        newName = "Eureka54_" + str(subclips_id)
        cropVideo(originalVideoA1, t1, t2, newName)
        subclips_id += 1
        
def eureka55():
        
    originalVideoA1 = home + "\\Pictures\\DS_VIDEOS\\Camp Eureka - GRIT - July 13, 2017\\Eureka55.mp4"        
    subclips_id = 0
    for i in range(1581, 1606):  # 25 entries
        
        t1 = df['t1'][i]
        t2 =  df['t2'][i]

        print(subclips_id)
        if(math.isnan(t2) or math.isnan(t1)):
            print("continue1")
            subclips_id += 1
            continue
        
        newName = "Eureka55_" + str(subclips_id)
        cropVideo(originalVideoA1, t1, t2, newName)
        subclips_id += 1
        
def eureka56():
        
    originalVideoA1 = home + "\\Pictures\\DS_VIDEOS\\Camp Eureka - GRIT - July 13, 2017\\Eureka56.mp4"        
    subclips_id = 0
    for i in range(1606, 1637):  # 25 entries
        
        t1 = df['t1'][i]
        t2 =  df['t2'][i]

        print(subclips_id)
        if(math.isnan(t2) or math.isnan(t1)):
            print("continue1")
            subclips_id += 1
            continue
        
        newName = "Eureka56_" + str(subclips_id)
        cropVideo(originalVideoA1, t1, t2, newName)
        subclips_id += 1
        
def eureka57():
        
    originalVideoA1 = home + "\\Pictures\\DS_VIDEOS\\Camp Eureka - GRIT - July 13, 2017\\Eureka57.mp4"        
    subclips_id = 0
    for i in range(1637, 1650):  # 25 entries
        
        t1 = df['t1'][i]
        t2 =  df['t2'][i]

        print(subclips_id)
        if(math.isnan(t2) or math.isnan(t1)):
            print("continue1")
            subclips_id += 1
            continue
        
        newName = "Eureka57_" + str(subclips_id)
        cropVideo(originalVideoA1, t1, t2, newName)
        subclips_id += 1
        
def eureka59():
        
    originalVideoA1 = home + "\\Pictures\\DS_VIDEOS\\Camp Eureka - GRIT - July 13, 2017\\Eureka59.mp4"        
    subclips_id = 0
    for i in range(1680, 1711):  # 25 entries
        
        t1 = df['t1'][i]
        t2 =  df['t2'][i]

        print(subclips_id)
        if(math.isnan(t2) or math.isnan(t1)):
            print("continue1")
            subclips_id += 1
            continue
        
        newName = "Eureka59_" + str(subclips_id)
        cropVideo(originalVideoA1, t1, t2, newName)
        subclips_id += 1
    
if __name__ == '__main__':
     eureka59()