# -*- coding: utf-8 -*-
"""
Author: Sharri Zamore zamore@vt.edu

This script takes raw color files and saves them as a singular raw pickle file.
"""

import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import pdb
import KF

red = pd.read_csv('~/BehaviorVids/94/94_170208_0.5rpm-red2.csv',delimiter=',',header=0)
yellow = pd.read_csv('~/BehaviorVids/94/94_170208_0.5rpm-yellow2.csv',delimiter=',',header=0)
len(red),len(yellow)

red_pred = KF.fill_in_data(red)
yellow_pred = KF.fill_in_data(yellow)
len(red_pred[0]),len(yellow_pred[0])

if len(red_pred[0])!=len(yellow_pred[0]):
    print("Lengths don't match. Aborting!")
else:

#determine frames and times (all videos are 25 fps)
    frames = red_pred[1]
    frames = np.linspace(frames[0],frames[-1],frames[-1]-frames[0]+1)
    frames = frames[:len(frames)-1]

    rot_dir = input('Enter rotation direction: CW or CCW. ')
    
    #TODO: improve this
    T = []
    rotation = []
    for row in frames:
        T.append(row/25)
        rotation.append(rot_dir)

    raw_data = np.column_stack((red_pred[0][:,0].T,red_pred[0][:,1].T,red_pred[0][:,2].T,red_pred[0][:,3].T,\
                       yellow_pred[0][:,0].T,yellow_pred[0][:,1].T,yellow_pred[0][:,2].T,yellow_pred[0][:,3].T,\
                       frames,rotation))
    raw_pd = pd.DataFrame(raw_data, columns = ['red_x','red_y','red x vel','red y vel','yellow_x',\
                                 'yellow_y','yellow x vel','yellow y vel','frames','rotation_direction'])

    filename = input('Enter entire file name with extension: ')

    raw_pd.to_pickle(filename)
    print('File saved!')