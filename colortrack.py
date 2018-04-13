import pandas as pd
import numpy as np
#import argparse
import cv2, csv, glob, os, sys
import pickle
#import scipy.spatial as sps
import pdb
import time
from matplotlib import pyplot as plt

COLOR_BOUND = {
'blue':((0,50,0), (130,225,136)),\
'red': ((0,100,70), (11,255,181)),\
'yellow': ((12,100,100),(24,255,255))
}

global camera,hsv, upperPt, lowerPt, kernel

#brightness = 1.5 #1.65
#contrast = -2. #-56.

def histNorm(frame,plotting=False):	
    """this function does some shit
    Frame must be cropped (tracking area only)"""

    hist, bins = np.histogram(frame.flatten(),256,[0,256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/cdf.max()

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
	if np.shape(frame)[1]==1300:
		frame = frame[upperPt[1]:lowerPt[1],upperPt[0]:lowerPt[0]]
	if event == cv2.EVENT_LBUTTONDOWN:
		frameCenter = (x,y)
		cv2.circle(frame,(x,y),10,(0,0,0),-1)
		cv2.imshow('Click on center', frame)
		print('Center is '+ str(frameCenter))
		cv2.destroyAllWindows()
		
		color_data["frameCenter"] = frameCenter

def calibrate(x,frame):
    """This function adjusts the (x,y) outputs of all tracked data and moves the origin to the center determined
    by get_center()"""
    convers = (44.45/2.)/param_data['diameter'] #convert pixels to cm
    new_x = []#np.zeros(np.size(x))
    #new_y = np.zeros(np.size(y))
    
    #default origin is the upper left corner of an image.
    
    if frameCenter == []:
        print ('Center not found')
        cv2.imshow('Click on center',frame)
        cv2.setMouseCallback('Click on center',get_center)
        cv2.waitKey(5000)
    for i in range(len(x)):
        if len(x[i])>1:
                new_x[i].extend([(x[i][0] - param_data['frameCenter'][0])*convers, 
                                 (param_data['frameCenter'][1] -x[i][1])*convers])
        elif len(x[i])==1:
            # pdb.set_trace()
            new_x[i] = (x[i] - param_data['frameCenter'][0])*convers
            #new_y[i] = (param_data['frameCenter'][1] - y[i])*convers
    return new_x

def savedata(reddata, yellowdata):
    save_data = {'red':[],'yellow':[]}

    if len(red.position) > len(yellow.position):
        numpts = len(yellow.position)-1
    elif len(red.position) < len(yellow.position):
        numpts = len(red.position)-1
    else:
        numpts = len(red.position)-1

    for i in range(numpts):
        save_data['red'].append([red_data[0][i],red_data[1][i],color_data['red'][i][1]]) 
        save_data['yellow'].append([yellow_data[0][i],yellow_data[1][i],color_data['yellow'][i][1]])
    #yellow.savefilenameadjBrtCont
    key_names = ['red','yellow']
    data_pd = pd.DataFrame(save_data, columns = save_data.keys())
    check_name = input('Is filename {} acceptable? Y/N '.format(yellow.savefilename))
    if check_name.lower == 'y':
        data_pd.to_csv(yellow.savefilename)
    else:
        check_name = input("Enter filename. Suggested format: 'ID_SPDrpm_DATE.p'")
        data_pd.to_csv(check_name)

class colortracker():#, numcolor=1, colornames='blue', tracers = 'off'):
    """Finds object of specified color(s), tracks pixel position and 
        radius of the largets object in the frame.

    Attributes:
        colornames: user-specified colors to track (red,green,blue)
        tracers: draw trails of tracked points on image (may slow tracking)
        position: (x,y) pixel position of center of largest centroid
        radius: pixel radius of largest centroid
    """

    def __init__(self, color, camera, flag, filename, draw='off',tracers='off'):
        #Thread.__init__(self)
        global brightness, contrast
        
        self.color = color
        self.draw = draw
        self.tracers = tracers
        self.position = []
        self.radius = []
        self.hsv_lower = np.uint8(COLOR_BOUND[color][0])
        self.hsv_upper = np.uint8(COLOR_BOUND[color][1])
        self.found = []
        self.camera = camera
        self.flag = flag
        self.csvname = []
        self.loadfilename = filename[:-4] + '_data.p'
        self.ID = filename[:2]
        self.draw = draw

        self.refPt = []
        self.sizePt = []

        self.imported = False

        if tracers != 'off':
            cv2.cvNamedWindow('Tracking video',1)

        #Make output file
        self.savefilename = str(filename[:-4]) + '.csv'  #changing files to .csv 12/1/2017
        saveit = open(self.savefilename, 'wb')

    def import_settings(self):
        """Asks user if image settings should be updated for specific color (determined by Thread). If yes, updates color
        center, reference region and crops image. If no, looks for save file of setting data then bypasses all setting 
        updates. BYPASSING SETTING UPDATES IS NOT RECOMMENDED. YOUR SHIT WILL EXPLODE."""
        global param_data, upperPt, lowerPt, frameCenter, refPt, frame, frameNum,global_data		

        filepath = input("Enter full path to parent folder. Videos in folder titled mp4, parameters in folder titled vidParam.. Use quotes. Ex: '/Users/name/folder' ")
        os.chdir(filepath +'/vidParam')#'/Volumes/ADIPOSE/'+self.ID+'/vidParam/')#'/Users/sazamore/BehaviorVids/90/')
        globalname = self.loadfilename[:3]+'00_'+ self.loadfilename[3:9] +'-global.p'
        print(self.loadfilename, globalname)
        param_data = pickle.load(open(self.loadfilename,"rb"))  #import video-specific data
        global_data = pickle.load(open(globalname,"rb"))  #import batch-specific data
        #Navigate parent folder
        os.chdir("..")
        os.chdir(filepath+'/mp4')
        #os.chdir('/Volumes/ADIPOSE/'+self.ID+'/Videos')

        #TODO: only reference param_data, get rid of these variables
        upperPt = global_data['upperPt']#(390,2) #param_data['upperPt']
        lowerPt = global_data['lowerPt']#(1085,650) #param_data['lowerPt']
        param_data['diameter']=global_data['diameter']
        frameCenter = param_data['frameCenter']
        refPt = param_data['refPt']

        self.hsv_lower = np.uint8(COLOR_BOUND[self.color][0])
        self.hsv_upper = np.uint8(COLOR_BOUND[self.color][1])

        #check for error in finding colors, set to canonical values
        if self.hsv_lower[0]==self.hsv_upper[0]:
            self.hsv_lower = np.uint8(COLOR_BOUND[self.color][0])
            self.hsv_upper = np.uint8(COLOR_BOUND[self.color][1])

        self.imported = True 
        print('Settings imported')
        return global_data, param_data	
        
    def search_region(self,frame, color, camera, regionSize=70):
        """Creates region in which colors will be searched based on last-found position. 
        Outputs adjusted frame (frame) and top left corner of the search region (topleft)"""
        global searchCenter, upperPt, lowerPt#, camera
        topleft = []

        #resize video and adjust color
        if len(upperPt)==1:
            upperPt = upperPt[0]
            lowerPt = lowerPt[0]
        
        if frame.shape[0] != lowerPt[1]-upperPt[1]:
            frame = frame[upperPt[1]:lowerPt[1],upperPt[0]:lowerPt[0]]

        frame = adjBrtCont(frame,self.brightness,self.contrast)

        searchCenter = self.position[-1]

        if isinstance(searchCenter, int):
            searchCenter = list(self.position)
        elif isinstance(searchCenter, tuple):
            searchCenter = list(self.position[-1])
        elif not isinstance(searchCenter[0],int) and not np.isnan(searchCenter[0]):            
            #make sure values are int
            searchCenter = [int(i) for i in self.position[-1]]
        
        #If the point wasn't found in the previous frame, search the whole frame
        if np.all(self.position) == 0 or np.isnan(searchCenter[0]) or not searchCenter:
            #First try to find the last found point
            for row in reversed(self.position):
                regionSize += 20
                if not np.isnan(row[0]) and not np.all(row)==0:
                    searchCenter = row
                    regionSize+=50
                    break
        if self.camera.get(cv2.CAP_PROP_POS_FRAMES) >= 217:
            time.sleep(.1)
        x = int(searchCenter[0]) - regionSize/2
        y = int(searchCenter[1]) - regionSize/2

        if x <= 0:
            x = 1
        elif y <= 0:
            y = 1

        topleft = (x, y)
        frame = frame[y:int(searchCenter[1])+regionSize/2, \
                      x:int(searchCenter[0])+regionSize/2]
        return frame,topleft

    def find_color(self, frame, camera, frameNum, color_data):
        global brightness, contrast
        #TODO:Somewhere outside of this function, determine if from camera or file

        self.camera = camera
        self.brightness = brightness = color_data['brightness']
        self.contrast = contrast = color_data['contrast']
        #print(brightness,contrast)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        
        if not self.imported: 
            #print('importing settings')
            self.import_settings()
            self.found = list(param_data[self.color+'_pos'])
            searchCenter = list(param_data[self.color+'_pos'])
            firstFrame = True
            other = (np.nan,np.nan)
            self.imported = True

            param_data['upperPt']=(390,2)
            param_data['lowerPt']=(1085, 657)

        if not self.position:
            if self.color == 'yellow':
                self.position = [list(param_data['yellow_pos'])]
            else:
                self.position = [list(param_data['red_pos'])]
                if not param_data['red_pos']:
                    self.position = [list(param_data['yellow_pos'])]
                    print(self.position)
            #print('No position data!')

        #Define search region
        frame,cropRef = self.search_region(frame,camera,self.color)

        #Convert color values to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        try:
            hsv = np.uint8(hsv)
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        except:
            print(type(frame))

        #filter image for specific color
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    #this function erodes and dilates based on kernel

        #find contours in mask
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(contours) > 0:
            #Find center of largest contour
            filtered = max(contours, key=cv2.contourArea)

            ((x, y), radius) = cv2.minEnclosingCircle(filtered)

            if radius > 1:#3: 
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
                if self.color =='yellow':
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                        (0, 255,255), 2)
                else:
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                        (0,0,255),2)
                if camera.get(cv2.CAP_PROP_POS_FRAMES)%1000 == 0:
                    print('Another 1000 frames tracked successfully!')

                self.position.append([x+cropRef[0],y+cropRef[1]])
                self.found = self.position[-1]
                self.radius.append(radius)
            else:
                print('{} object too small'.format(self.color))
                self.position.append([np.nan, np.nan])
                self.radius.append(np.nan)
        else:
            print('{} object not found'.format(self.color))
            x = y = 0 #np.nan
            radius = 0# np.nan								
            self.position.append([0,0])#(np.nan, np.nan))
            self.radius.append(0) #previously np.nan
        if self.draw.lower == 'on':
            cv2.imshow('frame',frame)
            cv2.waitKey(200)
        #Dump data to pickle 
        print(frameNum)
        color_data[self.color].append(self.position[-1]) 
        color_data[self.color+'_frames'].append(int(frameNum))
        color_data[self.color+'_r'].append(self.radius[-1])
        # pdb.set_trace()
        savepd = pd.DataFrame.from_dict(color_data,orient='index')
        savepd.transpose()
        # pickle.dump(color_data,open(self.savefilename,'wb'))
        savepd.to_csv(self.savefilename)
        
        if (cv2.waitKey(10) >= 0):
            camera.release()
            cv2.destroyAllWindows()

def adjBrtCont(img,brightness,contrast):
    mod_img = cv2.add(img,np.array([contrast]))
    mod_img = cv2.multiply(mod_img,np.array([brightness]))
    cv2.destroyAllWindows()
    return mod_img

def drawMatchColor(img,brightness,contrast):
    """Draws circles on image for color matching. Color values are BGR. To make conversions, 
    use http://http://colorizer.org/. Remember to update the image (run this function)
    after every whole-image modification, or color values will also change."""
    
    x1,y1 = param_data['red_pos']
    cv2.circle(img, (int(x1)+10, int(y1)-10), int(5),(21.32, 36.47,203),-2)#(2.5, 206.5, 204), 2)
    
    x2,y2 = param_data['yellow_pos']
    cv2.circle(img, (int(x2)+10, int(y2)-10), int(5),(0, 183, 255),-2)#(21.5, 255, 127.5), 2)

def img_adj(frame):
    """On-screen live adjustment of brightness and contrast."""
    global brightness, contrast
    
    brightness = 1.5
    contrast = -2.
    sat = 0.
    inc = 0.5

    finished = False
    scale = 1300./frame.shape[1]
    frame = cv2.resize(frame,None,fx=scale,fy=scale,interpolation = cv2.INTER_CUBIC)

    mod_img = frame[upperPt[1]:lowerPt[1],upperPt[0]:lowerPt[0]]
    prime_img = frame[:]
        
    cv2.imshow('mod',mod_img)
    
    while not finished:
        mod_img = adjBrtCont(prime_img,brightness, contrast)
        drawMatchColor(mod_img,brightness,contrast)
        cv2.imshow('mod',mod_img)

        k = cv2.waitKey(0)
        if k == ord('w'): #wincreases contrast
            contrast -= 2.
            mod_img = cv2.add(prime_img,np.array([contrast])) 
            
        elif k == ord('s'): #s decreases contrast
            contrast += 2.
            mod_img = cv2.add(prime_img,np.array([contrast])) 
            
        elif k == ord('a'): #a decreases brightness
            brightness -= 0.05
            mod_img = cv2.multiply(prime_img,np.array([brightness]))
            
        elif k == ord('d'): #d increases brightness
            brightness += 0.05
            mod_img = cv2.divide(prime_img,np.array([brightness]))

        elif k == ord('q'): #q decreases saturation
            sat += inc
            mod_img = mod_img[:,:,1]+inc   
            
        elif k == ord('e'): #e decreases saturation
            sat -= inc
            mod_img = mod_img[:,:,1]-inc     

        elif k == ord('o'): #show original
            cv2.imshow('original',frame)
            #k = cv2.waitKey()
            
        elif k == ord('x'): #x exits adjustment
            finished = True
            cv2.destroyAllWindows()

    cv2.destroyAllWindows()
    return brightness, contrast

def update_colors(frame, brightness,contrast):
    """Updates upper and lower HSV values for masking"""
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    x1,y1 = param_data['red_pos']
    red = hsv[y1-2:y1+2,x1-2:x1+2,:]
    red = np.uint8(red)
    redMin = (np.min(red[:,:,0]),np.min(red[:,:,1]),np.min(red[:,:,2]))	 	#not sure this will take min of each column
    redMax = (np.max(red[:,:,0]),np.max(red[:,:,1]),np.max(red[ :,:,2]))
    
    x2,y2 = param_data['yellow_pos']
    yellow = hsv[y2-2:y2+2,x2-2:x2+2,:]
    yellow = np.uint8(yellow)
    yellowMin = (np.min(yellow[:,:,0]),np.min(yellow[:,:,1]),np.min(yellow[:,:,2]))	 	#not sure this will take min of each column
    yellowMax = (np.max(yellow[:,:,0]),np.max(yellow[:,:,1]),np.max(yellow[ :,:,2]))
    
    return np.uint8(redMin), np.uint8(redMax), np.uint8(yellowMin), np.uint8(yellowMax)

def mask_check(mod_img):
    """Filters for yellow and red colors in augmented image ("mod_img"). Comparison to original ("frame") can be 
    made, but modifications to script are necessary. Displays augmented image with locations of "found" colors, 
    and images of masks for yellow and red colors. If the colors are well defined, there should be only small circle
    for red and yellow in the mask, and appropriately sized circles around the found colors on the input image 
    (mod_img)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    
    hsv_mod = cv2.cvtColor(mod_img, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv_mod, np.uint8(COLOR_BOUND['red'][0]),np.uint8(COLOR_BOUND['red'][1]))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel) 

    yellow_mask = cv2.inRange(hsv_mod, np.uint8(COLOR_BOUND['yellow'][0]),np.uint8(COLOR_BOUND['yellow'][1]))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel) 

    r_contours = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    if r_contours:
        filtered = max(r_contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(filtered)
        M = cv2.moments(filtered)
        #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.circle(mod_img, (int(x), int(y)), int(radius),
                                        (0, 0, 255), 2)
    else:
        print('No red found')

    y_contours = cv2.findContours(yellow_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    if y_contours:
        filtered = max(y_contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(filtered)
        M = cv2.moments(filtered)
        cv2.circle(mod_img, (int(x), int(y)), int(radius),(0, 255, 255), 2)
        print ()
    else:
        print('No yellow found')
        
    cv2.imshow('red mask',red_mask)
    cv2.imshow('yellow mask',yellow_mask)
    cv2.imshow('image',mod_img)
    cv2.waitKey(3000)
    
    k = cv2.waitKey(0)
    if k == ord('x'): #x exits adjustment
        cv2.destroyAllWindows()


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
        print hsv[color_loc[1],color_loc[0],:]
        print('Min value is {}.'.format(setColorMin))
        print('Max value is {}.'.format(setColorMax))
    
        #export data
        #color_data[color+'_pos']=color_loc
        #color_data[color+'_min']=setColorMin
        #color_data[color+'_max']=setColorMax
        cv2.destroyAllWindows()