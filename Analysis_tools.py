#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 12:20:08 2021

@author: ron
"""

import numpy as np
import matplotlib.pyplot as plt
from Plankter import plankter
from Ocean import ocean



def save_Ocean(fname, Ocean):
    '''
    will save the data in an Ocean instance
    as a numpy .npz file
    '''
    Np = len(Ocean.plankton) 
    dt = Ocean.plankton[0].dt 
    V = Ocean.plankton[0].V
    S = V * dt
    pos_lst = [np.array(Ocean.plankton[i].p) for i in range(Np)]
    encounters = [np.array(Ocean.plankton[i].encounters) for i in range(Np)]
    L = Ocean.L ; M = Ocean.M ; R = Ocean.R
    rho = R / S  ;  mu = M**0.5 * S / R

    params_string = ('Np, S, L, R, M, rho, mu, dt, V')
    params = (Np, S, L, R, M, rho, mu, dt, V)
    
    np.savez(fname, pos_lst = pos_lst,
             params = params, params_string = params_string,
             encounters = encounters)
    


def load_Ocean(fname):
    data = np.load(fname)
    Np, S, L, R, M, rho, mu, dt, V = data['params']
    D = data['pos_lst'].shape[-1]
    p_lst = []
    for i in range(int(Np)):
        plnk = plankter(np.zeros(D), V, dt) 
        plnk.p = list(data['pos_lst'][i])
        plnk.encounters = list(data['encounters'])
        p_lst.append(plnk)
    O = ocean(p_lst, R, M, L=L)
    return O
    


def load_pos_list(fname):
    data = np.load(fname)
    param_strings = str(data['params_string']).split(', ')
    for i in range(len(data['params'])):
        print(param_strings[i]+': ', data['params'][i])
    return data['pos_lst']
    
    
    


def concentration_at_scale(pos_list, limits, number_of_devisions):
    '''
    given a list of coordinates (pos_list) in D dimensional space,
    that in each direction is found in between the limits 
    (limits[0], limits[1]), and a number of devisions (bins). 
    Then, this function will calculate the number of coordinates in 
    each of space boxes of volume (L/bins)**D, and will return 
    it as a list.
    
    returns:
        concentration_list - list of concentrations
        scale = L/bins
    '''
    D = len(pos_list[0])
    rng = [(limits[0], limits[1]) for i in range(D)] 
    
    h, bins = np.histogramdd( np.array(pos_list),
                             bins=number_of_devisions,
                             range=rng)
    
    scale = (limits[1] - limits[0])/number_of_devisions
    concentration_list = np.ravel(h)
    
    #-----------------------------------------
    # repeat at translations of 1/2 cell size:
    d = (limits[1] - limits[0]) / number_of_devisions / 2
    rng2 = [(limits[0]+d, limits[1]-d) for i in range(D)] 
    h, bins = np.histogramdd( np.array(pos_list),
                             bins=number_of_devisions-1,
                             range=rng2)
    concentrations2 = np.ravel(h)
    concentration_list = np.append(concentration_list, concentrations2)
    #-----------------------------------------
    
    return concentration_list, scale





def concentration_PDF(pos_list, limits, number_of_devisions,
                      kmin=None, kmax=None):
    '''
    Will return the x, y values of the PDF of the concentration field
    of the positions pos_list, estimated with number_of_devisions devisions
    of space.
    
    kmin and kmax are the optionally given limits for the distribution.
    '''
    c, s = concentration_at_scale(pos_list, limits, number_of_devisions)
    
    if kmin is None: kmin = np.amin(c)
    if kmax is None: kmax = np.amax(c)
    values = np.arange(kmin, kmax+1)
    
    bins = [v-0.5 for v in values]
    bins.append(bins[-1]+1)
    h = np.histogram(c, bins=bins, density = True)
    x, y = 0.5*(h[1][:-1]+h[1][1:]), h[0]
    return x, y


def Pois(k, lmbda):
    '''the Poisson distribution PDF'''
    return lmbda**k / np.math.factorial(k) * np.exp(-lmbda)






#=============================================================================
# concentration measurements from experiments:

def concentration_from_experiments(pos_list, x_lim, y_lim, z_lim, cell_size):
    '''
    This is used to calculate a concentration field from experimental 
    pos_list. The difference with concentration_at_scale() is that here
    the space can be anisotropic so the cells are defined specifically
    for each coordinate axis.
    
    x_lim (tuple) - the limits, (xmin, xmax) for the domain in x direction.
    cell_size (float) - the side length of cubical cells for calculation.
    
    *** In each direction e.g. (xmax - xmin)/cell_size must be a whole number!!
    
    returns:
        concentration_list - list of concentrations
    '''
    
    Nx = int((x_lim[1]-x_lim[0])/cell_size )
    Ny = int((y_lim[1]-y_lim[0])/cell_size )
    Nz = int((z_lim[1]-z_lim[0])/cell_size )
    
    binsX = np.linspace(x_lim[0], x_lim[1], num=Nx+1)
    binsY = np.linspace(y_lim[0], y_lim[1], num=Ny+1)
    binsZ = np.linspace(z_lim[0], z_lim[1], num=Nz+1)
    
    
    h, bins = np.histogramdd( np.array(pos_list),
                             bins=(binsX, binsY, binsZ))
   
    concentration_list = np.ravel(h)
    
    #-----------------------------------------
    # repeat at translations of 1/2 cell size:
    if Nx>1 and Ny>1 and Nz>1:
        binsX = np.linspace(x_lim[0] + cell_size/2, x_lim[1] - cell_size/2, num=Nx)
        binsY = np.linspace(y_lim[0] + cell_size/2, y_lim[1] - cell_size/2, num=Ny)
        binsZ = np.linspace(z_lim[0] + cell_size/2, z_lim[1] - cell_size/2, num=Nz)
        
        h, bins = np.histogramdd( np.array(pos_list),
                             bins=(binsX, binsY, binsZ))
        
    concentrations2 = np.ravel(h)
    concentration_list = np.append(concentration_list, concentrations2)
    #-----------------------------------------
    
    return concentration_list






def concentration_PDF_from_experiment(pos_list, x_lim, y_lim, z_lim, cell_size,
                                      kmin=None, kmax=None):
    
    c = concentration_from_experiments(pos_list, x_lim, y_lim, z_lim, cell_size)
    if kmin is None: kmin = np.amin(c)
    if kmax is None: kmax = np.amax(c)
    values = np.arange(kmin, kmax+1)
    
    bins = [v-0.5 for v in values]
    bins.append(bins[-1]+1)
    h = np.histogram(c, bins=bins, density = True)
    x, y = 0.5*(h[1][:-1]+h[1][1:]), h[0]
    return x, y



#=============================================================================





def voidFraction_vs_scale(pos_list, limits, bins_list = range(1,10)):
    '''
    given a list of coordinates (pos_list), this function will
    calculate the voidFraction as a function of the scale.
    '''
    voidFraction_lst = []
    scale_lst = []
    for nod in bins_list:
        concentration_list, scale =  concentration_at_scale(pos_list, 
                                                            limits, 
                                                            nod)   
        vf = float(len(concentration_list[concentration_list==0])) / len(concentration_list)
        voidFraction_lst.append(vf)
        scale_lst.append(scale)
    return voidFraction_lst, scale_lst

    



def clustering_index_vs_scale(pos_list, limits, bins_list = range(1,10)):
    '''
    The clustering index is defined as the variance of the 
    concentration devided by the mean of the concentration.

    Given a list of coordinates (pos_list), this function will
    calculate the clustering index as a function of the scale.
    '''
    clusterIndex_lst = []
    scale_lst = []
    for nod in bins_list:
        c, scale =  concentration_at_scale(pos_list, limits, nod)   
        cs = np.var(c) / np.mean(c)
        clusterIndex_lst.append(cs)
        scale_lst.append(scale)

    return clusterIndex_lst, scale_lst





def clustering_index_at_scale(pos_list, limits, N_bins):
    '''
    The clustering index is defined as the variance of the 
    concentration devided by the mean of the concentration.

    Given a list of coordinates (pos_list), this function will
    calculate the clustering index at a fixed scale. The scale
    is determined by the size of the doiman, devided by N_bins,
    an input integer.
    '''
    c, scale =  concentration_at_scale(pos_list, limits, N_bins)   
    ci = np.var(c) / np.mean(c)
    return ci




def dispersion(pos_list, box_size, max_time = None, skip = 1):
    '''
    will calculate the dispersion as a function of time for 
    particles in an ocean instance
    D = <|x_i - x_0|^2> 
    
    Since this is can be a heavy calculation for many particles and
    long simulations, use max time to limit the duration over which
    the dispersion is estimated, and use skip to skip time steps, i.e.
    estimate D every, say, 5 time steps.
    '''
    
    # detect first reflection and mark it for each particles:
    L = box_size
    stop_lst = []
    for i in range(pos_list.shape[0]):
        pos = pos_list[i,:,:]
        ds = np.linalg.norm(pos[1:] - pos[:-1], axis = 1)
        for i in range(len(ds)):
            if ds[i]>= L/2:
                break
        stop = i+1
        stop_lst.append(stop)
    
    # calcualte dispersion for each time step individually:
    N = max(stop_lst)
    D = [0.0]
    count_traj = [pos_list.shape[0]]
    
    if max_time is None:
        max_time = N
    
    for i in range(1, max_time, skip):
        translations_squared = []
        c = 0
        for j in range(pos_list.shape[0]):
            if i < stop_lst[j]:
                pos_ = pos_list[j,:stop_lst[j],:]
                d = np.linalg.norm(pos_[i:,:] - pos_[:-i,:], axis=1)**2
                translations_squared += list(d)
                c+=1
        
        D.append(np.mean(translations_squared))
        count_traj.append(c)
        
    D = np.array(D)
    tm = np.array([0] + list(range(1, max_time, skip)))
    
    return tm, D, count_traj




def diffusivity(ocean , max_dist_from_center=0.25):
    disp = dispersion(ocean, max_dist_from_center=max_dist_from_center)
    D = []
    for i in range(1, len(disp)):
        D.append(disp[i] / i / 2)
    return np.array(D)
    


def clustering_index_vs_time(pos_lst, limits, N_bins):
    '''
    will return the clustering index vs. time for 
    coordinates in a position list array.
    
    Works when the pos_lst is loaded using the load_pos_list()
    '''
    n = pos_lst.shape[1]
    ci_lst = []
    
    for i in range(n):
        ci_lst.append( 
            clustering_index_at_scale(pos_lst[:,i,:], 
                                      limits, 
                                      N_bins) 
                     )
    return np.array(ci_lst)




# fname = 'S_M8d00.npz'
# pos_lst = pos_lst = load_pos_list(fname)
# ci_lst = clustering_index_vs_time(pos_lst, (0,1), 10)
# print(np.mean(ci_lst[1000:]))
# print(np.std(ci_lst[1000:]))
# plt.plot(ci_lst)


#np.savez('S_M_ci.npz', rho=rho, mu=mu, av_ci = av_ci, std_ci = std_ci)


#np.savez('dispersion_NoInt.npz', tm = tm , dispersion=d, count_traj=count_traj)

