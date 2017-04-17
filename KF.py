#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:04:45 2017

@author: sazamore
"""

import numpy as np
from pykalman import KalmanFilter
import pdb

def fill_in_data(color):
    """This function adds rows of NaNs to missing gaps in with 
    tracking data. NaNs are then masked and a Kalman filter is used to inter-
    polate the missing data points. Input (color) has to be an n x 3 matrix, 
    where the columns are: x,y,frame.
    Output is the filled x, y, and frames as individual lists
    """
    color = color
    colormat = color.as_matrix()
    frameDiff = np.diff(colormat.T[2])
    locations = np.where(frameDiff!=1)[0]
	
    #Calculate number of frames skipped
    sample = []
    sample = colormat.T
    sample = sample[:2].T
    frames = colormat.T[2]
    #frames = np.linspace(frames[0],frames[-1],frames[-1]-frames[0]+1)
    #frames = frames[:len(frames)-1]

    numfill = []
    missing = []
    for i in locations:
		numfill.append(frames[i+1]-frames[i]-1)
		missing.append(np.linspace(i+1,i+1+numfill[-1],numfill[-1]))

    missing = np.concatenate(missing)

    missing = missing[:len(missing)-1]
    missing = missing.astype(int)

    for j in reversed(missing):
        sample = np.insert(sample,j,(np.nan,np.nan),axis = 0)
		#frames = np.insert(frames,j,j,axis=0)

    color_x,color_y=KFilt(sample)
    color_mat = np.column_stack((color_x[:,0],color_y[:,0],color_x[:,1],color_y[:,1]))
    return color_mat,frames

def KFilt(sample):
    """Input (sample) is fill_in_data output. Outputs two lists of color data
    """
	#kalman filter inputs
    n_timesteps = len(sample)
    trans_mat = []
    trans_cov = []
    init_cond = [],[]
	#TODO: set up specific parameters (observation matrix, etc)
		
	#mask missing values
    observations = np.ma.array(sample,mask=np.zeros(sample.shape))
    missing_loc = np.where(np.isnan(sample))
    observations[missing_loc[0][:],missing_loc[1][:]] = np.ma.masked
	
	#Import Kalman filter, inerpolate missing points and get 2nd, 3rd orde kinematics
    dt = 1./25	#Length of each frame (should be iether 1/25 or 1/30)	
	#trans_mat = np.array([[1, 0 ,1, 0],[0, 1, 0, 1],[0,0,1,0],[0,0,0,1]])	
	#trans_cov = 0.01*np.eye(4)
    if not trans_mat:
        #if there's not a global variable defining tranisiton matrices and covariance, make 'em and optimize
        trans_mat = np.array([[1,1],[0,1]])
        trans_cov = 0.01*np.eye(2)
        kf = KalmanFilter(transition_matrices = trans_mat, transition_covariance=trans_cov)

        kf = kf.em(observations.T[0],n_iter=5)	#optimize parameters
        
        trans_mat = kf.transition_matrices
        trans_cov = kf.transition_covariance
        init_mean = kf.initial_state_mean
        init_cov = kf.initial_state_covariance

    else:
        kf = KalmanFilter(transition_matrices = trans_mat, transition_covariance=trans_cov,\
                         initial_state_mean = init_mean,initial_state_covariance = init_cov)
        
    global trans_mat, trans_cov, init_cond
    
    color_x = kf.smooth(observations.T[0])[0]
    color_y = kf.smooth(observations.T[1])[0]
	
    return color_x,color_y #np.column_stack((color_x[:,0],color_y[:,0],color_x[:,1],color_y[:,1])),frames
