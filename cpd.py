#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:16:49 2017

@author: sazamore
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 11:49:55 2016

Ray casting algorithm based on http://philliplemons.com/2014/09/14/ray-casting-algorithm/

@author: sazamore
"""
import sys
import numpy as np
from numpy import linalg as LA

#input
#head = (-2,-3)#(2,0)
#tail = (0,0)
#radius = 9
diam = 44.45    #diameter in cm

def make_circ(radius,res=100):
    t = np.linspace(0,2*np.pi,res)
    x = radius * np.sin(t)
    y = radius * np.cos(t)
    return x,y

class Points:
    def __init__(self,x,y):
        '''
        A series of points specified by (x,y) coordinates on the Cartesian plane
        '''
        self.x = x
        self.y = y
        
#use ray casting to find point where ray crosses enclosing polygon (circle) ONCE
#this script will loop through the edges of the circle
class Polygon:
    def __init__(self,points):
        '''
        points: a list of polygon verticies in a CW order. A zipped tuple of x 
        and y coordinates works best. Ex: [(x1,y1), (x2,y2),...,(xn,yn)]
        vect_x: a list of x positions along vector
        vect_y: a list of y postitions of vectors 
        
        '''
        self.points = points
        #self.edge_list = []
        #@property
        def edges(self):
            '''Returns a list of tuples that each contain 2 points of an edge'''
            self.edge_list = []
            #for i,p in enumerate(self.points):
            for i,p in enumerate(self.points):
                p1 = p
                p2 = self.points[(i+1) % len(self.points)]
                self.edge_list.append((p1,p2))
            return self.edge_list

        self.edge_list = edges(self)

    def contains(self, point):
        """When using this function, use the Points method for each coordinate.
        self.contains(Points(xn,yn))
        """
        #_huge is used to act as infinity if we divide by 0
        _huge = sys.float_info.max
        #_eps is used to make sure points are not on the same line as vertexes
        _eps = 1e-5

        #We start on the outside of the polygon
        inside = False
        for edge in self.edge_list:
            #Make sure A is te lower point of the edge
            A,B = edge[0],edge[1]

            if A[1] > B[1]:
                A, B = B, A
            #import pdb; pdb.set_trace()
                #Make sure the point is not at the same height as vertex
            if point.y == A[1] or point.y == B[1]:
                point.y += _eps
                
            if (point.y > B[1] or point.y < A[1] or point.x > max(A[0],B[0])):
                #The horizontal ray does not intersect with the edge
                continue
                
            if point.x < min(A[0], B[0]):
                #The ray instersects with the edge
                inside = not inside
                #print 'point is inside the circle'
                continue

            try:
                m_edge = (B[1] - A[1])/(B[0] - A[0])
            except ZeroDivisionError:
                m_edge = _huge

            try:
                m_point = (point.y - A[1])/(point.x - A[0])
            except ZeroDivisionError:
                m_point = _huge
            
            if m_point >= m_edge:
                #The ray intersects with the edge
                inside = not inside
                #print ('point is inside the circle')
                continue

        return inside#,A,B
    
    def dist_calc(self,vect_x,vect_y,data_mat,i):
        """Finds distance between head of vector (vect,data_mat) 
        and edge of polygon. 
        Returns distance between intercept point and
        vector head.
        vect should be an n x 2 array (try zipping x and y vectors)"""
        
        incircle = []
        #Vect = zip(vect_x,vect_y)#Points(vect_x,vect_y)
        #import pdb; pdb.set_trace()
        #for row in Vect:
        #    incircle.append(circle.contains(Points(row[0],row[1])))

        #find first point along extended vectior that is outsid the circle
        intercept = next(i for i,v in enumerate(incircle) if v==False)
        
def calc_view_dist(data_mat,vector_mag,circle,theta,grating,dt):
    """Calculates 2D viewing distances for all points in trajectory.
    Inputs:
        data_mat - m x 2 matrix of x and y points (columns) from which 
            distance to circumference is calculated. m is number of timesteps
        vector_mag - magnitude of vectors to determine angles
        
    """
    #make a vector of 1000 points and a polygon of 1000 points (~1/3 inch side lengths)
    #new mag adds the diameter, to ensure it leaves the polygon
    view_dist = []
    itcpt_dist = []
    cpd = []
    
    for k in range(len(data_mat)):
        vector_x,vector_y = [],[]
        #incircle = []
        for row in dt:
            vector_x = data_mat[k,0]+(row * np.cos(theta[k]))
            vector_y = data_mat[k,1]+(row * np.sin(theta[k]))
            
        #vect = zip(vector_x,vector_y)
            incircle = []
        
        #for row in vect:
            check = circle.contains(Points(vector_x,vector_y))
            if check:
                incircle.append(check)
            else:
                break
        intercept = len(incircle)
        #pdb.set_trace()
        itcpt_coord = (vector_x,vector_y)  #intercept coordinates on circumference
        #calculate distance of coordinate to vector head
        dxdy = (vector_x+data_mat[k,0],itcpt_coord[1]-data_mat[k,1])
        view_dist.append(LA.norm(dxdy)+2.54) #dist in cm
        #pdb.set_trace()
        if view_dist[-1]<2.54:
            view_dist[-1]=2.54
       
        cpd.append(round(1./(np.arctan(grating[0]/view_dist[-1]))/57.295779513,5))

    return cpd,view_dist