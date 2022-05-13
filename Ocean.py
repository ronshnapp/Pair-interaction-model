#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 10:40:34 2021

@author: ron
"""

import numpy as np
from scipy.spatial import cKDTree


class ocean(object):
    '''
    a class that holds the entire simulation
    
    plankton_lst - a list of plankter objects
    R - the perception radius
    M - the memory length of previous encounters
    L - Domain box size. If L is a number the the domain is between
        the boundaries (0,L) in each axis and they are periodic
        boundaries. If L is None (default) then the domain is infinite.
    '''
    
    def __init__(self, plankton_lst, R, M, L=None):
        self.plankton = plankton_lst
        self.R = R
        self.M = M
        self.dt = self.plankton[0].dt
        
        if L is None:
            self.boundedDomain = False
        
        else:
            self.boundedDomain = True
            self.L = L
        
    def __repr__(self):
        return 'An Ocean instance with %i partcles'%(len(self.plankton))
        
    
    def step(self):
        
        # first, advance all plankters one time step:
        for p in self.plankton: p.step()
           
        # get indexes of pairs that make encounters:
        pairs_indexes = self.find_close_plankters()
        N_pairs = len(pairs_indexes[0])
        
        # make the pair meet by changing their velocity, and mark the encounter:
        for k in range(N_pairs):
            i,j = pairs_indexes[0][k], pairs_indexes[1][k]
            pi, pj = self.plankton[i], self.plankton[j]
            meeting_point = (pi.p[-1] + pj.p[-1])/2
            vi = (meeting_point - pi.p[-1])/self.dt
            vj = (meeting_point - pj.p[-1])/self.dt
            pi.v[-1] = vi
            pj.v[-1] = vj
            pi.encounters[-1] = 1
            pj.encounters[-1] = 1
    
        # if bounded domain reflect all the plankers that
        # are out of (0,L):
        if self.boundedDomain:
            for plankter in self.plankton:
                plankter.p[-1] = plankter.p[-1]%self.L
            
    
    def find_close_plankters(self):
        '''
        returns indexes of plankton pairs that should bepaired in the next step
        '''
        '''
        # Old and brute force method for NN calculation:
        N = len(self.plankton)
        dM = int(self.M / self.dt)
        encounter_matrix = np.ones((N, N)) * -1
        for i in range(N):
            for j in range(i+1, N):
                social_index_i = 1 in self.plankton[i].encounters[-dM:] 
                social_index_j = 1 in self.plankton[j].encounters[-dM:]
                
                # both plankters are not ready for an encounter:
                if social_index_i or social_index_j: 
                    continue
                  
                d = self.distance(self.plankton[i], self.plankton[j])
                
                # if plankters are too far away:
                if d>self.R: 
                    continue
                
                else:
                    encounter_matrix[i,j] = d
                    
        # in each row and column keep only the pair minimum distance plankton:
        for i in range(N):
            w = np.where(encounter_matrix[i,:]>0)[0]
            if len(w)>1:
                mn = np.amin(encounter_matrix[i,:][w])
                encounter_matrix[i,:][encounter_matrix[i,:]>mn] = -1
                        
        for j in range(N):
            w = np.where(encounter_matrix[:,j]>0)[0]
            if len(w)>1:
                mn = np.amin(encounter_matrix[:,j][w])
                encounter_matrix[:,j][encounter_matrix[:,j]>mn] = -1
        
        return np.where(encounter_matrix > 0)
        
        '''
        # New method for NN calculation:
        # 1) get coordinates of plankton that are ready to interact:
        dM = int(self.M / self.dt)
        active_plankton_coords = []
        active_plankton_index = []
        for i, plnk in enumerate(self.plankton):
            if 1 in plnk.encounters[-dM:]: continue
            pos = plnk.p[-1]
            active_plankton_coords.append(pos)
            active_plankton_index.append(i)
        
        
        # 2) find the nearest pairs within the interaction radius 
        #    using the cKDTree method
        if len(active_plankton_coords) > 1:
            NN_searcer = cKDTree(active_plankton_coords)
            pairs = ([], [])
            for i in range(len(active_plankton_coords)):
                if i in pairs[1]: continue
                qi = NN_searcer.query(active_plankton_coords[i], k=[2])
                if qi[0][0]<self.R:
                    qj = NN_searcer.query(active_plankton_coords[qi[1][0]],
                                          k=[2])
                    if qj[1][0]==i:
                        pairs[0].append(active_plankton_index[i])
                        pairs[1].append(active_plankton_index[qi[1][0]])
                        
            return pairs
        
        else:
            return ([],[])

            
        
    
    
    def distance(self, plankter_i, plankter_j):
        '''returns the distance between a pair of plankters'''
        return np.linalg.norm(plankter_i.p[-1] - plankter_j.p[-1])
    
    
    def get_positions_at_step(self, i):
        '''returns the positions of planktons at timestep i'''
        pos_list = [plnktr.p[i] for plnktr in self.plankton]
        return pos_list
    
    
    def plot_at_step(self, step_i, fig, ax):
        '''will plot as a scatter plot the position of plankters at time step_i'''
        dM = int(self.M/self.dt)
        for i in range(len(self.plankton)):
            p = self.plankton[i]
            if 1 in p.encounters[step_i - dM:step_i]: c = 'y'
            else: c = 'r'
            ax.plot(p.p[step_i][0], p.p[step_i][1], 'o', ms=3, color=c)
            
    
    def calc_bbox(self):
        '''determines the boundaries of the plankton trajectories'''
        x_lst = []
        y_lst = []
        for p in self.plankton:
            x_lst += list(np.array(p.p)[:,0])
            y_lst += list(np.array(p.p)[:,1])
            
        self.bbox = (np.amin(x_lst), np.amax(x_lst), np.amin(y_lst), np.amax(y_lst))
        
    
    def get_distance_from_center_at_step_i(self, step_i):
        '''will return a list of the distances of all
        plankters from their center of the distribution at
        time step_i'''
        
        x_lst = []
        y_lst = []
        
        for p in self.plankton:
            x_lst.append(p.p[step_i][0])
            y_lst.append(p.p[step_i][1])
        
        cx = np.mean(x_lst)
        cy = np.mean(y_lst)
        
        return ((np.array(x_lst) - cx)**2 + (np.array(y_lst) - cy)**2)**0.5
        
        
    def get_distance_between_neighbours_at_step_i(self, step_i):
        '''will return a list of the distances between plankton at
        time step_i'''
        d_lst = []
        N = len(self.plankton)
        for i in range(N):
            for j in range(i+1, N):
                r = self.plankton[i].p[step_i] - self.plankton[j].p[step_i]
                d_lst.append(np.linalg.norm(r))
        return d_lst
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    