#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:04:45 2017

@author: sazamore
"""

import numpy as np
from pykalman import KalmanFilter
import pdb

def fill_in_data(color,frames,fs=25):
    """This function adds rows of NaNs to missing gaps in with 
    tracking data. NaNs are then masked and a Kalman filter is used to inter-
    polate the missing data points. 
    Inputs: 
    color-  an n x 2 matrix, where the columns are: x,y. 
    frames- an n x 1 vector of frames.
    fs - sample frequency (fps)
    Output is the filled x, y, and frames as individual lists
    """
    color = color
    colormat = color.as_matrix()
    frameDiff = np.diff(colormat.T[2])
    locations = np.where(frameDiff!=1)[0]

    #Calculate number of frames skipped
    #sample = []
    #sample = colormat.T
    sample = sample[:2].T
    #frames = range(100,len(colormat.T[2])+100)
    #frames = np.linspace(frames[0],frames[-1],frames[-1]-frames[0]+1)
    #frames = frames[:len(frames)-1]
    
    #if locations is empty, try looking for a row of nans
    if np.all(locations):
        for i in range(len(sample)):
            if np.all(sample[i] == 0):
                sample[i]=[np.nan, np.nan]
        missing = list(np.where(np.isnan(sample.T[0])))

    else:
        numfill = []
        missing = []
        for i in locations:
            numfill.append(frames[i+1]-frames[i])#-1)
            #pdb.set_trace()
            missing.append(np.linspace(i+1,i+1+numfill[-1],numfill[-1]))

        missing = np.concatenate(missing)

        missing = missing[:len(missing)-1]
        missing = missing.astype(int)

        pdb.set_trace()

        for j in reversed(missing):
            sample = np.insert(sample,j,(np.nan,np.nan),axis = 0)
            #frames = np.insert(frames,j,j,axis=0)

    color_x,color_y,x_filt=KFilt(sample,fs)
    color_mat = np.column_stack((color_x[:,0],color_y[:,0],color_x[:,1],color_y[:,1]))
    return color_mat,frames,x_filt


def KFilt(sample,fs=25):
    """Input (sample) is fill_in_data output. Outputs two lists of color data
    """
	#kalman filter inputs
    
        # Dimensions of parameters:
    # 'transition_matrices': 2,
    # 'transition_offsets': 1,
    # 'observation_matrices': 2,
    # 'observation_offsets': 1,
    # 'transition_covariance': 2,
    # 'observation_covariance': 2,
    # 'initial_state_mean': 1,
    # 'initial_state_covariance': 2,
    
    n_timesteps = len(sample)
    trans_mat = []

	#mask missing values
    observations = np.ma.array(sample,mask=np.zeros(sample.shape))
    missing_loc = np.where(np.isnan(sample))
    observations[missing_loc[0][:],missing_loc[1][:]] = np.ma.masked
	
	#Import Kalman filter, inerpolate missing points and get 2nd, 3rd orde kinematics
    dt = 1./25	#Length of each frame (should be iether 1/25 or 1/30)	
    n_timesteps = len(sample)
    
    observation_matrix = np.array([[1,0,0,0],
                                   [0,1,0,0]])#np.eye(4) 
    t = np.linspace(0,len(observations)*dt,len(observations))
    q = np.cov(observations.T[:2,:400])
    qdot = np.cov(np.diff(observations.T[:2,:400]))#np.cov(observations[:1,:400])

    h=(t[-1]-t[0])/t.shape[0]
    A=np.array([[1,0,h,.5*h**2], 
                     [0,1,0,h],  
                     [0,0,1,0],
                     [0,0,0,1]]) 

    init_mean = [sample[0],0,0] #initial mean should be close to the first point, esp if first point is human-picked and tracking starts at the beginning of a video
    observation_covariance = q*500  #ADJUST THIS TO CHANGE SMOOTHNESS OF FILTER
    init_cov = np.eye(4)*.001#*0.0026
    transition_matrix = A
    transition_covariance = np.array([[q[0,0],q[0,1],0,0],
                                      [q[1,0],q[1,1],0,0],
                                      [0,0,qdot[0,0],qdot[0,1]],
                                      [0,0,qdot[1,0],qdot[1,1]]])

    kf = KalmanFilter(transition_matrix, observation_matrix,transition_covariance,observation_covariance,n_dim_obs=2)

    kf = kf.em(observations,n_iter=1,em_vars=['transition_covariance','transition_matrix','observation_covariance'])

    #pdb.set_trace()
    
    global trans_mat, trans_cov, init_cond
    x_filt = kf.filter(observations[0])[0]#observations.T[0])[0]
    kf_means = kf.smooth(observations[0])[0]
	
    return kf_means,x_filt #np.column_stack((color_x[:,0],color_y[:,0],color_x[:,1],color_y[:,1])),frames
