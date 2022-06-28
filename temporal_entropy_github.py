#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 15:35:22 2022

@author: erinnsun
"""

import numpy as np
from scipy import io, fftpack
import pywt # Wavelet Transform
import matplotlib.pyplot as plt


# suppress future warning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



# Construct State Matrix
def state_matrix(A, Q, transform = True):
    
    if np.any(np.iscomplex(A)): # if any element of A is complex (e.g. A_dft)
        A  = np.abs(A) 
    
    A = np.asarray(A, dtype = 'float64')
    
    if transform:
        A = np.reshape(A, (A.shape[0], -1)) # (epochs, nodes)
        A = A.T # (nodes, epochs)
    
    A_min = np.min(A, axis = -1)
    A_max = np.max(A, axis = -1)
    intvl = (A_max - A_min) / Q
    
    # calculate states
    if isinstance(A_min, np.ndarray):   # reshape to enable broadcasting across the array  
        A_min = A_min[:, np.newaxis]
        intvl = intvl[:, np.newaxis]
    intvl +=  1e-6 # handle the zero case
    S = (A - A_min) / intvl
    S = np.trunc(S).astype('int64')
    
    return S



# Temporal Marginal Entropy
def TME(S, Q):
    
    counts = np.vstack((np.bincount(row, minlength = Q) for row in list(S))) # count the frequency of each state
    P = counts / S.shape[1] # calculate probabilities
    logP = np.log2(P + 1e-12) # handle log2(0)
    H = - np.sum(np.multiply(P, logP), axis = 1) # calculate marginal entropy
    
    return H
   


# Temporal Conditional Entropy
def TCE(S, Q):
    
    N = S.shape[0] # number of nodes
    counts = np.zeros((N,  Q * Q))
    
    # count the frqeuency of each pair of states
    for i in range(N):
        for j in range(S.shape[1] - 1):
            s1 = S[i][j] # state 1
            s2 = S[i][j+1] # state 2
            counts[i][s1 * Q + s2] += 1
    
    P = counts / (S.shape[1] - 1) # calculate probabilities   
    logP = np.log2(P + 1e-12) # handle log2(0)
    tje = - np.sum(np.multiply(P, logP), axis = 1) # calculate joint entropy
    
    tme = TME(S, Q)
    # print('TJE:', tje.mean())
    # print('TME:', tme.mean())
    H = tje - tme # calculate conditional entropy
    return H




def load_mat_data(path = 'nasdaq100.mat', var_name = 'X'):
    '''
    Load '.mat' files
    '''

    data = io.loadmat(path)
    data = list(data[var_name].flatten())
    A = np.asarray(data) # shape = (epochs, stocks, features)
    print(path + 'shape', A.shape)
    
    return A


def add_cdf(entropy, ax, idx, n_pts, label):
    '''
    Calculate the CDF corresponding to the entropy distribution 
    '''
    
    S = state_matrix(entropy, n_pts, transform = False)
    counts = np.bincount(S, minlength = n_pts) 
    CDF = np.cumsum(counts)/np.sum(counts)
    X = np.linspace(entropy.min(), entropy.max(), n_pts)
    ax[idx[0], idx[1]].plot(X, CDF, label = label)
    ax[idx[0], idx[1]].legend()
    
        
    
      
    
    
    
    
    
    
if __name__ == '__main__':
    A = load_mat_data('nasdaq100/nasdaq100.mat', 'X')
    A_dft = load_mat_data('nasdaq100/nasdaq100_dft.mat', 'A')
    A_dwt = load_mat_data('nasdaq100/nasdaq100_dwt.mat', 'B')
    A_dct = load_mat_data('nasdaq100/nasdaq100_dct.mat', 'C')
    
    
    Q = 10 # number of states
    fig, ax = plt.subplots(2, 2, figsize = (15, 10))
    
    for i in range(A.shape[-1]):
        print('Feature %d'%(i+1))
        
        # select the i_th feature
        U = A[:, :, i] 
        U_dft = A_dft[:, :, i] 
        U_dwt = A_dwt[:, :, i] 
        U_dct = A_dct[:, :, i]
        
    
        S = state_matrix(U, Q)
        S_dct = state_matrix(U_dct, Q)
        # S_dft = state_matrix(U_dft, Q)
        # S_dwt = state_matrix(U_dwt, Q)
        
        
        
        print('Calculating original domain...')
        tce = TCE(S, Q)
        tme = TME(S, Q)
        print('Calculating transform domains...')
        tce_dct = TCE(S_dct, Q)
        tme_dct = TME(S_dct, Q)
        # tce_dwt = TCE(S_dwt, Q)
        # tme_dwt = TME(S_dwt, Q)
        # tce_dft = TCE(S_dft, Q)
        # tme_dft = TME(S_dft, Q)
        
        
        n_pts = 1000 
        idx = [int(i%2), int(i/2)]
        add_cdf(tce, ax, idx, n_pts, 'conditional')
        add_cdf(tce_dct, ax, idx, n_pts, 'conditional-TD')
        add_cdf(tme, ax, idx, n_pts, 'marginal')
        add_cdf(tme_dct, ax, idx, n_pts, 'marginal-TD')
        

    
    
    
    
    
    
    
        