# -*- coding: utf-8 -*-
"""
Updates video settings using opencv (or matplotlib?). for each video, user selects ROI (non-white region of video), color 
values (red and yellow), starting location of colors and center.
Created on Wed Aug 17 11:19:18 2016

@author: sazamore
"""

import cv2,os,glob,csv,time
import numpy as np
import pyqtgraph as qt
import pickle

global color, upperPt, lowerPt, frameCenter, refPt, color_data, scaled, frame
upperPt = []
lowerPt = []
frameCenter = []
refPt = []
diameter = []
count = 0
scaled = False

color_data = {
'red_pos': [],\
'red_min': [],\
'red_max': [],\
'yellow_pos': [],\
'yellow_min': [],\
'yellow_max': [],\
'lowerPt': [],\
'upperPt': [],\
'frameCenter': [],\
'refPt': [],\
'diameter': [],\
}

color_set = ['red','yellow']

def histNorm(frame,plotting=False):	
    """this function does some shit
    Frame must be cropped (tracking area only)"""

    hist, bins = np.histogram(frame.flatten(),256,[0,256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/cdf.max()
    #TODO: save cdf from example frame to make normalized hist for subsequent frames
    
    if plotting:
        plt.plot(cdf_normalized, color = 'b')
        plt.hist(frame.flatten(),256,[0,256])
        plt.xlim([0,256])
        plt.legend(('cdf','histogram'), loc = 'upper left')
        plt.show()
        #TODO: delete image with keystroke

    #mask and scale histogram
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    
    new_image = cdf[frame]
    
    return new_image

def get_center(event,x,y,flags,param):			
    """Finds center of an image using mouse click"""
    global frameCenter, frame	
    if event == cv2.EVENT_LBUTTONDOWN:
        frameCenter = (x,y)
        cv2.circle(frame,(x,y),10,(0,0,0),-1)
        cv2.imshow('Click on center', frame)
        print('Center is '+ str(frameCenter))
        cv2.destroyAllWindows()
    
        color_data["frameCenter"] = frameCenter

def get_circ(event,x,y,flags,param):
    """Finds edge of drum for distance estimations"""	
    global frame,frameCenter#,diameter
    click = 0
    if event == cv2.EVENT_LBUTTONDOWN:
        #draw circles where user clicks
        cv2.circle(frame,(x,y),1,(0,0,0),-1)
        cv2.imshow('Click 6 points on circle', frame)
        diameter.append(np.sqrt((frameCenter[0]-x)**2 + (frameCenter[1]-y)**2))
    return diameter

def get_color(event,x,y,flags,param):
    """gets location of color from button click, derives color value parameters for tracking"""
    global color_loc,setColor,setColorMax, setColorMin,frame	#define these
    color_loc = []	
    setColorMin = []
    setColorMax = []	
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if event == cv2.EVENT_LBUTTONDOWN:		#check 
        color_loc = (x,y)		
        setColor = hsv[y-2:y+2,x-2:x+2,:]
        setColor = np.uint8(setColor)		
        setColorMin = (np.min(setColor[:,:,0]),np.min(setColor[:,:,1]),np.min(setColor[:,:,2]))	 
        #not sure this will gake min of each column
        setColorMax = (np.max(setColor[:,:,0]),np.max(setColor[:,:,1]),np.max(setColor[ :,:,2]))
        print hsv[color_loc[0],color_loc[1],:]
    
        #export data
        color_data[color+'_pos']=color_loc
        color_data[color+'_min']=setColorMin
        color_data[color+'_max']=setColorMax
    
        cv2.destroyAllWindows()

def get_reference(event, x,y,flags, param):
    """Selects ROI (100 x 100 rectangle) on frame from which the grating rotation speed can be calculated.
    ROI is a rectangle whos bounds are determined by clicking the position of the upper
    left corner of the rectangle. Image window name must be 'image'. """
    global refPt,frame
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        refPt.append((x+80,y+80))
        #accessing the values within the rectange would use: image[refPt[0][0:1],refPt[1][0:1]]
        cv2.rectangle(frame,refPt[0],refPt[1],(255,255,0),2)
        cv2.imshow('Reference region made',frame)
        cv2.destroyAllWindows()
    
        color_data["refPt"] = refPt

def selectROI(camera):
    global sel,upperPt, lowerPt
    
    def selectFrame(event, x, y, flags, param): 
        """Click and drag to select entire probable search area starting in upper left corner."""
        global drag_start, sel, patch
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start = x, y
            sel = 0,0,0,0
        elif event == cv2.EVENT_LBUTTONUP:
            if sel[2] > sel[0] and sel[3] > sel[1]:
                patch = frame[sel[1]:sel[3],sel[0]:sel[2]]
                cv2.imshow("result", patch)
                color_data["upperPt"] = sel[:1]
                color_data["lowerPt"] = sel[2:]
            drag_start = None
        elif drag_start:
            #print flags
            if flags & cv2.EVENT_FLAG_LBUTTON:
                minpos = min(drag_start[0], x), min(drag_start[1], y)
                maxpos = max(drag_start[0], x), max(drag_start[1], y)
                sel = minpos[0], minpos[1], maxpos[0], maxpos[1]
                #img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                img = frame.copy()
                cv2.rectangle(img, (sel[0], sel[1]), (sel[2], sel[3]), (0,255,255), 1)
                cv2.imshow("select ROI", img)
            else:
                print("selection is complete")
                print(sel)
                drag_start = None


    cv2.namedWindow("select ROI", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("select ROI", selectFrame)
    sel = (0,0,0,0)
    drag_start = None
    patch = None
    
    camera.set(cv2.CAP_PROP_POS_FRAMES, 300) #set video to beginning
    #for i in range(int(nframes)):
    r, frame = camera.read()
    if r == True:
        #frame = f
        scale = 1300./frame.shape[1]
        frame = cv2.resize(frame,None,fx=scale,fy=scale,interpolation = cv2.INTER_CUBIC)
        cv2.imshow('select ROI', frame)
        
        k = cv2.waitKey()
        if k == 27:
            cv2.destroyAllWindows()
            cv2.namedWindow("select ROI", cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback("select ROI", selectFrame)
            #sel = (0,0,0,0)
            drag_start = None
            patch = None
            #break # exit
        #if k == ord('r'):
            #continue #skip frame without selecting anything
        if k == ord('a') or k == ord('x'):
            #print('Sel is {}'.format(sel))
            upperPt,lowerPt = sel[:2],sel[2:]
            color_data["upperPt"] = upperPt
            color_data["lowerPt"] = lowerPt
            print upperPt, lowerPt
            cv2.destroyAllWindows()
            #break
    #upperPt,lowerPt = sel[:2],sel[2:]
    #color_data["upperPt"] = upperPt
    #color_data["lowerPt"] = lowerPt
    #print upperPt, lowerPt
    return sel, frame
    
def resize_frame(event,x,y,flags,param):
    """Click and drag to select entire probable search area starting in upper left corner."""		
    global upperPt, lowerPt, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        upperPt = [(x,y)]
        print upperPt
    if event == cv2.EVENT_LBUTTONUP:
        lowerPt = [(x,y)]
        print lowerPt
        cv2.rectangle(frame, upperPt[0], lowerPt[0],(0,0,0),1)
        cv2.destroyWindow('Select region of interest')			
        #crop frame
        frame = frame[upperPt[0][1]:lowerPt[0][1],upperPt[0][0]:lowerPt[0][0]]
        cv2.imwrite('resized.jpg',frame)
        frame = histNorm(frame)
        print('Resize successful')
        cv2.imshow('Select region of interest', frame)	

        color_data["upperPt"] = upperPt
        color_data["lowerPt"] = lowerPt

if __name__ == '__main__':
    #Navigate to video filepath and search for videos by animal number
    os.chdir('/Users/sazamore/BehaviorVids')
    ID = ["81_170426-clip.mp4"] #str(raw_input('Enter animal ID number: ')) + '*.mp4'

    #Iterate through videos with animal number. Create and save settings.
    for filename in ID: # glob.glob(ID):
        #set/reset values
        upperPt = []
        lowerPt = []
        frameCenter = []
        refPt = []
        scaled = False

        color_data = {
        'red_pos': [],\
        'red_min': [],\
        'red_max': [],\
        'yellow_pos': [],\
        'yellow_min': [],\
        'yellow_max': [],\
        'lowerPt': [],\
        'upperPt': [],\
        'frameCenter': [],\
        'refPt': [],\
        'diameter': [],\
        }

        color_set = ['red','yellow']

        #set up camera, grab 200th frame (lighting issues)
        camera = cv2.VideoCapture(filename)
        camera.set(cv2.CAP_PROP_POS_FRAMES, 300)
        (grabbed, frame) = camera.read()

        if grabbed:
            print('Frame captured successfully')

        #scale all videos to same size
        scale = 1300./frame.shape[1]
        frame = cv2.resize(frame,None,fx=scale,fy=scale,interpolation = cv2.INTER_CUBIC)
        scaled = True

        filename = filename[:-4] + '_data.csv'		

        #crop video
        frame.shape
        if upperPt == [] or lowerPt ==[]:	
            k,frame = selectROI(camera)

        frame= frame[sel[1]:sel[3],sel[0]:sel[2],:]
        print(frame.shape)
        print('Uppper pt is {}, lower pt is {}'.format(upperPt, lowerPt))
            
        newframe = histNorm(frame)
        
        cv2.imshow('Keep color correction? (y/n)',newframe)
        k = cv2.waitKey()
        if k == ord('y'):
            frame=newframe
            cv2.destroyAllWindows()
        elif k == ord('n'):
            cv2.destroyWindow('Keep color correction? (y/n)')
        
        #get center
        if frameCenter == [] : 
            cv2.imshow('Click on center',frame)
            cv2.setMouseCallback('Click on center',get_center)
            cv2.waitKey(5000)
            cv2.destroyWindow('Click on center')

        #get diam
        while count < 6:
            cv2.imshow('Click 6 points on circle',frame)
            cv2.setMouseCallback('Click 6 points on circle', get_circ)
            cv2.waitKey(4000)
            print(diameter)
            count += 1
        cv2.destroyWindow('Click 6 points on circle')
        radius = int(np.mean(diameter))
        cv2.circle(frame,frameCenter,radius,(0,0,0),2)
        color_data['diameter'] = radius*2

        #find reference region
        if refPt ==[]:
            #get stimulus motion reference
            cv2.imshow('Click pattern for reference',frame)	
            cv2.setMouseCallback('Click pattern for reference',get_reference)
            cv2.waitKey(5000)

        #get color values
        for color in color_set:
            if color_data[color+'_pos'] == []:
                cv2.imshow(color,frame)			
                cv2.setMouseCallback(color,get_color)
                cv2.waitKey(5000)		#-1 and 0 values seem to hang up. 5000 works.

        #write settings to file
        #import pdb; pdb.set_trace()
        picklename = filename[:-3]+'p'
        pickle.dump(color_data, open(picklename,"wb"))
        print("Successfully saved to {}".format(picklename))


#        time.sleep(5)
