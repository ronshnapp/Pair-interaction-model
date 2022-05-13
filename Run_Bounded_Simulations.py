#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 11:01:13 2021

@author: ron


This script is used to run a simulation of the pair-interaction models
with a bounded domain size and periodic boundary conditions.

"""



import numpy as np
from Plankter import plankter, plankter_bias
from Ocean import ocean
from Analysis_tools import save_Ocean



# ======= Parameters ======== #
    
D = 3            # dimensions
Np = 1000        # num. of particles
L = 1.0          # domain side length
R = 0.05         # interaction rad.
dt = 1.0         # time step size
V =  R / 16.0    # velocity
M = 10.0         # memory time
steps = 1000     # number of steps to run


# the file name used to save the results
save_name = 'results.npy'


# ====== Print summary ======= # 

S = V*dt
print('S = %.3f'%S)
print('rho = %.3f'%(R/S))
print('mu = %.3f'%(S/R*M**0.5))

step_spheres = L**D / S**D 
print('particles per step sphere = %.3f'%(Np/step_spheres))

interaction_spheres = L**D / R**D 
print('particles per interaction sphere = %.3f'%(Np/interaction_spheres))





# ======= Setup ======== #

 
def init_pos_generator():
    x0 = np.random.uniform(0.0, L, size=D)
    #x0[1] = np.random.normal(loc=0.5, scale=0.1)
    return x0


    

#for ii in range(20):
    
P_lst = []
for i in range(Np):
    plnk = plankter(init_pos_generator(), V, dt) 
    #plnk = plankter_bias(init_pos_generator(), V, dt) 
    
    #set initial encounters list:
    enc0 = []
    for j in range(int(M/dt)):
        if np.random.uniform(0,1) <= 1.0/(M/dt):
            enc0.append(1)
            for k in range(j+1,int(M/dt)): enc0.append(0)
            break
        else:
            enc0.append(0)
    plnk.encounters = enc0
    
    P_lst.append(plnk)
    
    # make up initial encounters:
O = ocean(P_lst, R, M, L=L)




# ======= Run several steps of the simulation ======== #


if steps > 0:
    
    import sys
    import time
    
    t0=time.time()

    print('')
    for i in range(steps):
        O.step()
        sys.stdout.write('\r')
        sys.stdout.write('%d / %d  '%(i, steps))
        sys.stdout.flush()
        
    run_time = time.time() - t0
    print('\n')
    print('finished at %.1f seconds'%run_time)
    print('%.3f seconds per step'%(run_time/steps))
    
    #save_Ocean(save_name, O)

    

