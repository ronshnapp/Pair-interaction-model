#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 10:34:44 2021
@author: ron


The PLankter class of the pair-interaction model

"""

import numpy as np


class plankter(object):
    '''
    A class that holds the position and velocity
    of each plankter in the simulation. Note that this
    basic class supports any number of dimensions, and that 
    this is determined by the dimension of the initial position.
    
    p0 - initial position (D dimensional vector)
    V - swimming speed (constant scalar)
    dt - time step     (constant scalar)
    encounters - a list of boolian values, 
                =0 if there was no encounter in this time step
                and =1 if there was.
    '''
    
    def __init__(self, p0, V, dt):
        
        self.dt = dt
        self.V = V
        self.p = [p0]
        self.v = []
        self.encounters = [0]
        self.D = len(p0)
        
        self.generate_new_velocity()
        
        
    def generate_new_velocity(self):
        
        r = np.random.uniform(-1, 1, self.D)
        self.v.append( r / np.linalg.norm(r) * self.V )
    
    
    def step(self):
        self.p.append( self.p[-1] + self.v[-1] * self.dt )
        self.generate_new_velocity()
        self.encounters.append(0)

        
        
        
class plankter_bias(object):
    '''
    A class that holds the position and velocity
    of each plankter in the simulation. Note that this
    basic class supports any number of dimensions, and that 
    this is determined by the dimension of the initial position.
    
    p0 - initial position (D dimensional vector)
    V - swimming speed (constant scalar)
    dt - time step     (constant scalar)
    encounters - a list of boolian values, 
                =0 if there was no encounter in this time step
                and =1 if there was.
    '''
    
    def __init__(self, p0, V, dt):
        
        self.dt = dt
        self.V = V
        self.p = [p0]
        self.v = []
        self.encounters = [0]
        self.D = len(p0)
        
        self.generate_new_velocity()
        
        
    def generate_new_velocity(self):
        
        r = np.random.uniform(-1, 1, self.D)
        vi = r / np.linalg.norm(r) * self.V
        
        x1 = self.p[-1][1]
        if x1>0.5:
            sign=-1.
        else:
            sign=1.
        vi[1] += x1*(1-x1) * self.V/10 * sign
        self.v.append( vi )
    
    
    def step(self):
        self.p.append( self.p[-1] + self.v[-1] * self.dt )
        self.generate_new_velocity()
        self.encounters.append(0)