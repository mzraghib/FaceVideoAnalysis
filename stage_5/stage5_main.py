'''
Script to crop longer videos into shorter segments

Set directories for uncropped videos and resulting cropped videos

To crop all videos in the uncropped_dir, simply run the script which
will output all cropped videos in the cropped_dir directory

'''

import numpy as np
import os
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import logging

logging.basicConfig(filename= 'test_logs.log', 
                 level=logging.DEBUG,format='%(message)s')

curr_dir =  os.getcwd()
uncropped_dir = curr_dir + '\\videos_uncropped\\'
cropped_dir = curr_dir + '\\videos_cropped\\'
start_times = {'69202': 315,'69224': 214,'69225': 149,
         '69227': 247,'69239': 272}
begin_times = {'69202': 44126,'69224': 47447,'69225': 48082,
         '69227': 51078,'69239': 53252}

""" crop video from original video and save """

def cropVideo(videoName, t1, t2, newName):
    ffmpeg_extract_subclip(videoName, t1, t2, targetname= cropped_dir + str(newName) + ".mp4")
    
    
def get_times(video_id, t1, t2):
    start = begin_times[str(video_id)] # start time (offset) in seconds (24hr format)
    start_in_vid = start_times[str(video_id)] # start time for video playback
    #convert from timestamp to seconds
    t1_seconds = t1.hour*60*60 + t1.minute*60 + t1.second
    t2_seconds = t2.hour*60*60 + t2.minute*60 + t2.second
    
    delta_t1_start = t1_seconds - start # delta from t1 to start time
    delta_t1_t2 = t2_seconds - t1_seconds  # delta from t2 to t1
    
    new_t1 = delta_t1_start + start_in_vid
    new_t2 = new_t1 + delta_t1_t2
    
    return new_t1, new_t2

def list_file_ids(path):
    '''
    returns a list of ids (with extension, without full path) of all files 
    in folder path
    '''
    files = {}
    ids = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            #first few numbers before '_' are the ID            
            files[(name.split("_", 1)[0])] = name
            ids.append(int(name.split("_", 1)[0]))
    return files, ids
    
def processVids(table, uncropped_vids_ids):
    #loop through each index/question segment and crop into different videos
    for i in range(len(table)):
        #ignoring bad data
        if int(table['sessionId'][i]) not in uncropped_vids_ids:
            continue
        if table['mode'][i] == 'demo':
            continue
        
        newName = int(table['ID'][i])
        t1 = table['problemBeginTime'][i] # timestamp format
        t2 = table['newEndTime'][i]


        # calculate times with offset start time
        video_id = table['sessionId'][i]
        newt1, newt2 = get_times(video_id, t1, t2)        
        time_delta = newt2 - newt1

        
        #ignoring bad data
        if np.isnan(time_delta) or time_delta <= 0:
            logging.info('IGNORED {}.mp4 since time_delta = {}'.format(str(newName),(str(time_delta))))
            continue
#        print('Generating vid ', str(newName),'.mp4', 'of duration',
#                     str(time_delta), 'seconds from', str(newt1), 'to ', str(newt2))
        
        logging.info('Generating vid {}.mp4 of duration {}s, from {} to {}'.format(str(newName),
                     str(time_delta),str(newt1),str(newt2)))

        

        videoName = uncropped_dir + uncropped_vids[str(table['sessionId'][i])]
        cropVideo(videoName, newt1, newt2, newName)
        
        
        
if __name__ == '__main__':
    
    uncropped_vids, uncropped_vids_ids = list_file_ids(uncropped_dir)
    df = pd.read_excel('Classs1451ProblemHistory.xlsx', sheetname='Sheet1')


    table = df[['ID','sessionId','student_userName','mode','mastery','emotionAfter','emotionLevel',
                'effort','numHintsBeforeSolve','isSolved','timeInSession',
                'problemBeginTime','timeInTutor','timeToFirstAttempt','timeToFirstHint','timeToSolve',
                'newEndTime']]
  
    processVids(table, uncropped_vids_ids)       
    print('END')

