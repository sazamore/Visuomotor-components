#USAGE
#Because this script may use the camera, you need to call the script from root
# sudo python ball_tracking.py --video VIDEO_TITLE.mp4
# sudo python ball_tracking.py

#Ideas borrowed from https://github.com/bipul21/Colored-Ball-Tracking
#and 

import pandas as pd
import numpy as np
#import argparse
import cv2, csv, glob, os, sys
import pickle
import scipy.spatial as sps


COLOR_BOUND = {
'blue':((0,50,0), (130,225,136)),\
'red': ((0,100,70), (12,255,255)),\
'yellow': ((12,100,100),(31,255,255))
}
#'yellow'20:30
color_loc = {
'red': [],\
'blue': [],\
'yellow': [],\
}

color_data = {
'yellow_output': [],\
'red_output':[],\
}

frameCenter = []	
setColor = []
setColorMin = []
setColorMax = []
refPt = np.zeros((4))
position = []

firstFrame = True

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

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


class colortracker():#, numcolor=1, colornames='blue', tracers = 'off'):
	"""Finds object of specified color(s), tracks pixel position and 
		radius of the largets object in the frame.

	Attributes:
f		colornames: user-specified colors to track (red,green,blue)
		tracers: draw trails of tracked points on image (may slow tracking)
		position: (x,y) pixel position of center of largest centroid
		radius: pixel radius of largest centroid
	"""
	
#	if numcolor is None:
#		numcolor = raw_input('Specify number of colors to track. ')
#	elif len(colornames) != numcolor:
#		print('Please specify the correct number of color names to track')
#		colornames = str(raw_input('Enter % color names: ') %numcolor) 

	def __init__(self, color, camera, flag, draw='on',tracers='off'):
		#Thread.__init__(self)
		#global color		
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

		self.refPt = []
		self.sizePt = []
		
		
		self.velocity = (0,0)		
		self.next = (0,0)
		self.factor = 1

		if tracers != 'off':
			cv2.cvNamedWindow('Tracking video',1)

		#Make output file
		self.savefilename = str(filename[:-4]) + '.p'
		saveit = open(self.savefilename, 'wb')
				
	def import_settings(self):
		"""Asks user if image settings should be updated for specific color (determined by Thread). If yes, updates color
		center, reference region and crops image. If no, looks for save file of setting data then bypasses all setting 
		updates. BYPASSING SETTING UPDATES IS NOT RECOMMENDED. YOUR SHIT WILL EXPLODE."""
		global param_data, upperPt, lowerPt, frameCenter, refPt, frame		
				
		param_data = {
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
		}
					
		param_data = pickle.load(open(self.loadfilename,"rb"))

		#TODO: only reference param_data, get rid of these variables
		upperPt = param_data['upperPt'][0]
		lowerPt = param_data['lowerPt'][0]
		frameCenter = param_data['frameCenter']
		refPt = param_data['refPt']
		
		#TODO: make this another class?
		
		self.hsv_lower = np.uint8(COLOR_BOUND[self.color][0])
		self.hsv_upper = np.uint8(COLOR_BOUND[self.color][1])
		
#		self.hsv_lower = np.uint8(param_data[self.color+'_min'])
#		self.hsv_upper = np.uint8(param_data[self.color+'_max'])
		
		#check for error in finding colors, set to canonical values
		if self.hsv_lower[0]==self.hsv_upper[0]:
			self.hsv_lower = np.uint8(COLOR_BOUND[self.color][0])
			self.hsv_upper = np.uint8(COLOR_BOUND[self.color][1])
		
				
		print('Settings imported')
		return frame	
			
	def search_region(self,frame, color, regionSize=100):	#make this a hidden method?
		"""Creates region in which colors will be searched based on last-found position. 
		Outputs adjusted frame (frame) and top left corner of the search region (topleft)"""
		global searchCenter		
		topleft = []		
		#resize video and adjust color
		frame = frame[upperPt[1]:lowerPt[1],upperPt[0]:lowerPt[0]]
		frame = histNorm(frame)

#		if self.position == [] or np.isnan(self.position[-1][0]):	
#			print "first frame business"
#			searchCenter = param_data[color+'_pos']
#			#searchCenter = (searchCenter[1], searchCenter[0])
#			topleft = (searchCenter[1] - regionSize/2, searchCenter[0]-regionSize/2)
#			#frame = frame[searchCenter[1]-regionSize/2:searchCenter[1]+regionSize/2, searchCenter[0]-regionSize/2:searchCenter[0]+regionSize/2]
#
#		else:
		searchCenter = self.found#self.position[-1]
		if searchCenter == (0,0) or searchCenter==(np.nan, np.nan):
			#If the point wasn't found in the previous frame, search the whole frame
			#TODO: predict position based on velocity			
			topleft = (0,0)
		else:
			topleft = (searchCenter[0] - regionSize/2, searchCenter[1]-regionSize/2)					
			frame = frame[searchCenter[1]-regionSize/2:searchCenter[1]+regionSize/2, searchCenter[0]-regionSize/2:searchCenter[0]+regionSize/2]

		return frame,topleft
	
	def find_color(self,frame):
		global hsv, color_data, upperPt, lowerPt		
		#TODO:Somewhere outside of this function, determine if from camera or file
		firstFrame = False	
		
		#frame = cv2.medianBlur(frame, 9) #check that this doesn't make finding a small bead all screwy
		if camera.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) <= 1.0:
			if frameCenter == []:
				frame = self.import_settings()
			self.found = param_data[self.color+'_pos']	
			firstFrame = True
			self.factor = 1
			other = (np.nan,np.nan)
		
			if self.color == 'yellow':
				self.position = [param_data['yellow_pos']]
			else: #self.color == 'red':
				self.position = [param_data['red_pos']]
				
		#Predict location of object
		self.next = (self.found[0] + self.velocity[0], self.found[1] + self.velocity[1])

		#define search region
		if self.factor <= 5:
			frame,cropRef = self.search_region(frame,self.color)
		else:
			frame,cropRef = self.search_region(frame,self.color,regionSize=150)
				
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	
		#filter image for specific color
		mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    #this function erodes and dilates based on kernel

		#find contours in mask
		contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
		#center = None
		all_circles = []
		radii = []
		#areas = []
		
        	if len(contours) > 0:
			# find the largest contour in the mask, then use
			# it to compute the minimum enclosing circle and centroid
									
			#TODO: set default to predicted!!
									
#			filtered = max(contours, key=cv2.contourArea)
#			((x, y), radius) = cv2.minEnclosingCircle(filtered)										
			
			for i in range(len(contours)):
				all_circles.append(cv2.minEnclosingCircle(contours[i])[0])
				radii.append(cv2.minEnclosingCircle(contours[i])[1])				
			
			#find 2 circles closest to last found point. 
			#Find contour cloest to predicted location	.
			
			#TODO: set these points relavent to scaled frame, not search region
			tree = sps.cKDTree(all_circles)
			closest_found = tree.query((self.found[0]-cropRef[0],self.found[1]-cropRef[1]), k=2)#[1]	
			closest_next = tree.query((self.next[0]-cropRef[0],self.next[1]-cropRef[1]))#[1]
					
			last = (self.found[0]-cropRef[0],self.found[1]-cropRef[1])
			current = (cropRef[0]+x,cropRef[1]+y)
			nearest = all_circles[closest_found[1][0]]
			Next = all_circles[closest_next[1]]
			
			if self.color == 'yellow' and camera.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) != 1.0:
				other = red.found

			elif self.color == 'red' and camera.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) != 1.0:
				other = yellow.found
			#dist = np.sqrt((current[0]-other[0])**2 + (current[1]-other[1])**2)

			
			#Debugging info
#			print(len(contours))
#			print('Closest to last found is {}'.format(last))
#			print('Closest to current found is {}'.format(current))
#			print('Closest to predicted location is {}'.format(Next))
#			print('Other found color is {}'.format(other))
			
			if camera.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) >=298.0:
				import pdb;pdb.set_trace()

			#if the current point is not closest to the last found point, 
			#pick the another point 
			if other[0]+3 > current[0] > other[0]-3 or other[1]+3 > current[1] > other[1]-3:
				#current point is too close to other color point				

				if len(contours) == 2:
					#try looking for the second-closest point
					if (x,y) == last:
						#choose the other found point, if 2
						((x,y),radius) = cv2.minEnclosingCircle(contours[closest_found[1][1]]) 
					else:
						((x,y),radius) = cv2.minEnclosingCircle(contours[closest_found[1][0]]) 
					self.factor = 1
				elif len(contours) > 2:
					if (x,y) == Next or nearest == Next:
						#choose nearest to last found point
						((x,y),radius) = cv2.minEnclosingCircle(contours[closest_found[1][0]])
					elif (x,y) == nearest:
						#choose predicted
						((x,y),radius) = cv2.minEnclosingCircle(contours[closest_next[1][0]])
					else:
						((x,y),radius) = cv2.minEnclosingCircle(contours[closest_found[1][0]]) 					
						
				elif closest_found[0][0] < 8: # (round(x),round(y))==(round(Next[0]),round(Next[1])):	
					#confirm that current point is in reasonable proximity
					((x,y),radius) = cv2.minEnclosingCircle(contours[closest_found[1][0]]) 					
					self.factor = 1
				else:
					radius = 0
					self.factor += 1
			elif closest_found[0][0] > 30:
				#makes some arbitrarily large jump, don't trust it
				radius = 0
				self.factor += 1
				
			self.velocity = (self.factor*(x+cropRef[0]-self.position[0][0]),self.factor*(y+cropRef[1]-self.position[0][1]))
			#M = cv2.moments(contours[largest_contour[0]])
                	if 0 < radius < 12: #try addding radus > 3 													
			      # draw the circle and centroid on the frame,
                        	# then update the list of tracked points
				if self.color =='yellow':															
	                        	cv2.circle(frame, (int(x), int(y)), int(radius),
                                	(0, 255, 255), 2)
				else:
					cv2.circle(frame, (int(x), int(y)), int(radius),
						(0,0,255),2)

				#if self.color==yellow and np.all(np.round(x+cropRef[0]),np.round(y+cropRef[y])==np.round(red.found)):
					#import pdb;pdb.set_trace()

				self.position.append((x+cropRef[0],y+cropRef[1]))
				self.found = self.position[-1]
				self.radius.append(radius)
				
                	else:
                        	print('{} object too large'.format(self.color))
				self.position.append((np.nan, np.nan))
				self.radius.append(np.nan)
        	else:
			x = y = 0 #np.nan
			radius = 0# np.nan								
			self.position.append((0,0))#(np.nan, np.nan))
			self.radius.append(0) #previously np.nan
                	print('{} object not found'.format(self.color))

		#Dump data to pickle stream
		#TODO: this is not fast, augment for live tracking ish
		color_data[self.color+'_output'].append((self.position[-1], self.radius[-1]))
		pickle.dump(color_data,open(self.savefilename,'wb'))

		cv2.imshow("tracking...",frame)
		if (cv2.waitKey(10) >= 0):
		#alternate:
#		if cv2.waitKey(10) & 0xFF == ord('q'):
			camera.release()
			cv2.destroyAllWindows()
			
	def find_nearest(self,array,value):
		idx = (np.abs(array - value)).argmin()
		return [idx]#, array[idx]]
		
	def recenter(self,x,y):
		"""This function adjusts the x,y outputs of all tracked data and moves the origine to the center determined
		by get_center()"""
		#default origin is the upper left corner of an image.
		if frameCenter == []:
			print ('Center not found')
			cv2.imshow('Click on center',frame)
			cv2.setMouseCallback('Click on center',self.get_center)
			cv2.waitKey(5000)
		new_x = x - frameCenter[0] 	#x_center,y_center are global
		new_y = frameCenter[1] - y
		return new_x,new_y

if __name__ == '__main__':
	os.chdir('/Users/sazamore/BehaviorVids/')

	ID = str(raw_input('Enter animal ID number: ')) + '_28nov16_1rpm.mp4'
	for filename in glob.glob(ID):	
		global filename
		camera = cv2.VideoCapture(filename)
		
#		for i in range(200):
#			(grabbed, frame) = camera.read()
	
		#TODO:figure out live camera capture business. NEEDED?
		#TODO: toggle tracking "preview" (showing the tracking as it's happening)
		cv2.namedWindow("tracking", cv2.CV_WINDOW_AUTOSIZE)
		red = colortracker("red",camera,1)
		yellow = colortracker("yellow",camera,1)

		while True:
			#grab and scale frame
			(grabbed, frame) = camera.read()	#check args for filename
			scale = 1300./frame.shape[1]
			frame = cv2.resize(frame,None,fx=scale,fy=scale,interpolation = cv2.INTER_CUBIC)
			
			yellow.find_color(frame)		
			red.find_color(frame)
			#blue.find_color(frame)
			if cv2.waitKey(10) >= 0:
				camera.release()
				cv2.destroyAllWindows()
				print('Quitting tracker..')
				sys.exit(1)