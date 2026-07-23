import numpy as np
import numba
from numba import njit , prange, jit
import progressbar
import math
import cmath
from cmath import pi
from tkinter import W
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import time
import joblib
from joblib import parallel_backend, Parallel, delayed
from scipy import signal
import scipy
import csv
from operator import mod
from turtle import shape
from IPython.display import display, Math
from operator import mul
from functools import reduce 
from scipy.optimize import fsolve
import sdeint
import multiprocessing
import os.path as path
import shutil
from tqdm import tqdm
import glob
import warnings
import os
from tempfile import mkdtemp
warnings.filterwarnings("ignore")




path1='/Users/tabar/Desktop/Abolfazl/JD_Paper/res/'
# path1="D:/U/M/M-P/res/"

@njit
def MY_JUMPY_SDEINT(N, F, G, Ksi2, Lambda, dt, nsteps, x0):
    x = np.zeros((N, nsteps))
    x[:, 0] = x0
    sqrt_dt = np.sqrt(dt)
    
    for i in range(nsteps - 1):
        t = i * dt
        # Wiener increment and its square minus dt
        dW = np.random.normal(0.0, sqrt_dt, N)
        dZ = 0.5 * (dW**2 - dt)        
        # First intermediate step (K1)
        k1 = x[:, i] + F(x[:, i], t) * dt + np.dot(G(x[:, i], t), dW)
        # Second intermediate step (K2)
        # F_k1 = F(k1, t + dt)
        # G_k1 = G(k1, t + dt)
        # Update: Runge-Kutta style (Milstein-like correction)
        x[:, i + 1] = x[:, i] + 0.5 * (F(x[:, i], t) + F(k1, t + dt)) * dt \
                        + np.dot(G(x[:, i], t), dW) \
                        + np.dot((G(k1, t + dt) - G(x[:, i], t)), dZ)
        # Jump part
        rate = Lambda(x[:, i], t)
        J = Ksi2(x[:, i], t)
        for n in range(N):
            if np.random.rand() < rate[n] * dt:
                jump_noise = np.random.normal(0.0, 1.0, N)
                for row in range(N):
                    x[row, i + 1] += np.sqrt(J[row, n]) * jump_noise[row]
                    
    return x


@njit(fastmath=True)
def MY_EM_SDE(N, F, G, Ksi, dt, nsteps, x0):
    x = np.zeros((N, nsteps), dtype=np.float64)
    dw = np.random.normal(0, 1, (N, nsteps)) * np.sqrt(dt)
    x[:, 0] = x0
    for i in range(nsteps - 1):
        x[:, i + 1] = x[:, i] + F(x[:, i], i * dt) * dt + G(x[:, i], i * dt) * dw[:, i] + Ksi(x[:, i], i * dt)
    return x  

def caly(x,N):
    xshape=x.shape

    y=np.zeros((N,xshape[1])) 
    for i in range(N):
        y[i,:]=(np.roll(x[i,:],-1)-x[i,:])  
    x2=np.zeros((N,xshape[1]-1))  
    y2=np.zeros((N,xshape[1]-1))    
    for i in range(N):
        x2[i,:]=np.delete(x[i,:],-1)
        y2[i,:]=np.delete(y[i,:],-1) 
    return x2,y2   

def calpowercoef(N,M,mode):
    if mode=='lim':
        powercoef=np.zeros((1+sum(N**M for M in range(1,M+1)),N))
        counter=0
        for j in range(1,M+1):
            for k in range(1,N**j+1):
                counter+=1
                for l in range(1,j+1):
                    x=math.ceil(k/N**(j-l))
                    if (x==0): x=N
                    k=mod(k,N**(j-l))
                    powercoef[int(counter),x-1]+=1
        powercoef2,ind=np.unique(powercoef,axis=0,return_index=True)
        powercoef2=powercoef2[np.argsort(ind)]
    elif mode=='list':
        powercoef=np.zeros((sum(N**M for M in M),N))
        counter=0
        for j in M:
            for k in range(1,N**j+1):
                counter+=1
                for l in range(1,j+1):
                    x=math.ceil(k/N**(j-l))
                    if (x==0): x=N
                    k=mod(k,N**(j-l))
                    powercoef[int(counter)-1,x-1]+=1
        powercoef2,ind=np.unique(powercoef,axis=0,return_index=True)
        powercoef2=powercoef2[np.argsort(ind)]        
    else:
        raise ValueError("mode can only be 'lim' or 'list' ")
    return powercoef2

@njit()  # parallel=True
def calM2(x,N,powercoef):
    mdim=len(powercoef)
    M2=np.zeros((mdim,mdim))
    xshape=x.shape
    for i1 in range(mdim):
        for i2 in range(i1,mdim):
            a=np.ones(xshape[1])
            pc=powercoef[i1,:]+powercoef[i2,:]
            for i in range(N):
                a=a*x[i,:]**pc[i]
            M2[i1,i2]=np.mean(a)
            M2[i2,i1]=M2[i1,i2]             
    return M2

@njit()  # parallel=True
def calM1(l,ilist,x,y,N,powercoef):
    mdim=len(powercoef)
    M1=np.zeros(mdim)
    xshape=x.shape
    for i1 in range(mdim):
        a=np.ones(xshape[1])
        for i in range(l):
            a=a*y[ilist[i],:] 
        for i in range(N):
            a=a*x[i,:]**powercoef[i1,i]
        M1[i1]=np.mean(a)        
    return M1

def calDlcoef(l,x,y,dt,N,M,mode):
    powercoef=calpowercoef(N,M,mode)

    C_keys = []
    for i in range(len(powercoef)):
        temp = []
        for j in range(N):
            if powercoef[i, j] != 0:
                temp.append(f"x{j+1}^{int(powercoef[i, j])}")
        C_keys.append("*".join(temp))  # Join terms with '*'
    if C_keys[0] == '':
        C_keys[0] = '1'


    Dcoeffs=pd.DataFrame()
    M2=calM2(x,N,powercoef)


    if l==1:
        for i in range(N):   
            ilist=np.array([i])
            M1=calM1(1,ilist,x,y,N,powercoef)
            Dcoef=np.linalg.solve(M2,M1)/dt
            Dcoeffs[f"M\u2081{i+1}"]=Dcoef 

    elif l==2:
        for i in range(N):
            for j in range(i,N):   
                ilist=np.array([i,j])
                M1=calM1(2,ilist,x,y,N,powercoef)
                Dcoef=np.linalg.solve(M2,M1)/dt
                Dcoeffs[f"M\u2082{i+1}{j+1}"]=Dcoef  

    elif l==4:
        for i in range(N):
            for j in range(i,N):   
                ilist=np.array([i,i,j,j])
                M1=calM1(4,ilist,x,y,N,powercoef)
                Dcoef=np.linalg.solve(M2,M1)/dt
                Dcoeffs[f"M\u2084{i+1}{i+1}{j+1}{j+1}"]=Dcoef 

    elif l==6:
        for i in range(N):
            for j in range(i,N): 
                for k in range(j,N):  
                    ilist=np.array([i,i,j,j,k,k])
                    M1=calM1(6,ilist,x,y,N,powercoef)
                    Dcoef=np.linalg.solve(M2,M1)/dt
                    Dcoeffs[f"M\u2086{i+1}{i+1}{j+1}{j+1}{k+1}{k+1}"]=Dcoef 


    elif l==246:
        for i in range(N):
            for j in range(i,N):   
                ilist=np.array([i,j])
                M1=calM1(2,ilist,x,y,N,powercoef)
                Dcoef=np.linalg.solve(M2,M1)/dt
                Dcoeffs[f"M\u2082{i+1}{j+1}"]=Dcoef 
        for i in range(N):
            for j in range(i,N):   
                ilist=np.array([i,i,j,j])
                M1=calM1(4,ilist,x,y,N,powercoef)
                Dcoef=np.linalg.solve(M2,M1)/dt
                Dcoeffs[f"M\u2084{i+1}{i+1}{j+1}{j+1}"]=Dcoef 
        for i in range(N):
            for j in range(i,N): 
                for k in range(j,N):  
                    ilist=np.array([i,i,j,j,k,k])
                    M1=calM1(6,ilist,x,y,N,powercoef)
                    Dcoef=np.linalg.solve(M2,M1)/dt
                    Dcoeffs[f"M\u2086{i+1}{i+1}{j+1}{j+1}{k+1}{k+1}"]=Dcoef         

    elif l==46:
        for i in range(N):
            for j in range(i,N):   
                ilist=np.array([i,i,j,j])
                M1=calM1(4,ilist,x,y,N,powercoef)
                Dcoef=np.linalg.solve(M2,M1)/dt
                Dcoeffs[f"M\u2084{i+1}{i+1}{j+1}{j+1}"]=Dcoef 
        for i in range(N):
            for j in range(i,N): 
                for k in range(j,N):  
                    ilist=np.array([i,i,j,j,k,k])
                    M1=calM1(6,ilist,x,y,N,powercoef)
                    Dcoef=np.linalg.solve(M2,M1)/dt
                    Dcoeffs[f"M\u2086{i+1}{i+1}{j+1}{j+1}{k+1}{k+1}"]=Dcoef         






    Dcoeffs.index = C_keys
    col=Dcoeffs.columns
    ind=Dcoeffs.index
    return Dcoeffs,ind,col

def D1_func(x, df):
    """
    Evaluate the D1 function for a given x vector.
    :param x: A numpy array of N dimensions (e.g., [x1, x2, ...]).
    :param df: The DataFrame containing the coefficients.
    :return: An N-dimensional array of D1(x).
    """
    N = len(x)  # Dimensionality of x
    results = np.zeros(N)

    for dim in range(N):
        # Evaluate for each dimension M_1, M_2, ..., M_N
        column_name = f"M\u2081{dim+1}"
        terms = df.index  # Get terms like "1", "x1^1", "x1*x2^1", etc.
        coefficients = df[column_name].values  # Coefficients for this dimension
        
        for term, coeff in zip(terms, coefficients):
            if coeff != 0:  # Ignore zero terms
                # Parse term like "x1^1*x2^1" and evaluate it
                factors = term.split('*')
                product = 1.0
                for factor in factors:
                    if '^' in factor:
                        var, power = factor.split('^')
                        idx = int(var[1:]) - 1  # Convert "x1" to index 0, etc.
                        product *= x[idx] ** int(power)
                    else:
                        product *= float(factor)  # Handle constants like "1"
                results[dim] += coeff * product
    return results

def calculate_newy(x, df, dt):
    """
    Calculate newx = (x[i+1] - x[i]) - D1(x[i]).
    :param x: A numpy array of shape (N, L), where N is dimensions and L is data points.
    :param df: The DataFrame containing the coefficients.
    :return: A numpy array of newx.
    """
    N, L = x.shape
    newy = np.zeros((N, L - 1))  # Result array
    for i in range(L - 1):
        newy[:, i] = (x[:, i + 1] - x[:, i]) - D1_func(x[:, i], df)*dt

    return newy

def ansacalallDcoef(N,F,G,Ksi2,Lambda,dt,nsteps,x0,method,nansa,M,mode,exn,path1,n_parallel, dxf, histsave, dx, momentssave):
    D1shape=[int(np.sum(  [  np.prod(  [(N+i) for i in range(mi)]  ) / math.factorial(mi)  for mi in range(M[0]+1)  ]  ))  ,  int(N)]
    D2shape=[int(np.sum(  [  np.prod(  [(N+i) for i in range(mi)]  ) / math.factorial(mi)  for mi in range(M[1]+1)  ]  ))  ,  int(N*(N+1)/2)]
    D4shape=[int(np.sum(  [  np.prod(  [(N+i) for i in range(mi)]  ) / math.factorial(mi)  for mi in range(M[2]+1)  ]  ))  ,  int(N*(N+1)/2)]
    D6shape=[int(np.sum(  [  np.prod(  [(N+i) for i in range(mi)]  ) / math.factorial(mi)  for mi in range(M[3]+1)  ]  ))  ,  int(N*(N+1)*(N+2)/6)]

    filename1 = path.join(mkdtemp(), 'newfile1.dat')
    filename2 = path.join(mkdtemp(), 'newfile2.dat')
    filename3 = path.join(mkdtemp(), 'newfile3.dat')
    filename4 = path.join(mkdtemp(), 'newfile4.dat')
    filename6 = path.join(mkdtemp(), 'newfile6.dat')

    D1array=np.memmap(filename1, dtype='float32', mode='w+', shape=((int(nansa),int(D1shape[0]),int(D1shape[1]))))
    D2array=np.memmap(filename2, dtype='float32', mode='w+', shape=((int(nansa),int(D2shape[0]),int(D2shape[1]))))
    D4array=np.memmap(filename3, dtype='float32', mode='w+', shape=((int(nansa),int(D4shape[0]),int(D4shape[1]))))
    D6array=np.memmap(filename4, dtype='float32', mode='w+', shape=((int(nansa),int(D6shape[0]),int(D6shape[1]))))
    Moments = np.memmap(filename6, dtype='float32', mode='w+', shape=(int(nansa), int(N), 7))

    temp_hist_dir = mkdtemp()  # Create a temp dir to store histograms

    def calansa(i):
        xdata=MY_JUMPY_SDEINT(N,F,G,Ksi2,Lambda,dt,nsteps,x0)
        if histsave:
            for j in range(N):
                maxabs = np.max(np.abs(xdata[j, :]))
                half_bins = int(np.ceil(maxabs / dx))
                if half_bins % 2 == 0:
                    half_bins += 1  # force odd
                nbins = 2 * half_bins - 1
                hist_range = (-half_bins * dx, half_bins * dx)
                hist, _ = np.histogram(xdata[j, :], bins=nbins, range=hist_range, density=True)
                temp_filename = os.path.join(temp_hist_dir, f"hist_i{i}_j{j}.npz")
                np.savez(temp_filename, hist=hist, nbins=nbins, hist_range=hist_range)

        if momentssave==True:
            for j in range(N):
                Moments[i,j,0] = np.mean(xdata[j,:]**1)
                Moments[i,j,1] = np.mean(xdata[j,:]**2)
                Moments[i,j,2] = np.mean(xdata[j,:]**4)
                Moments[i,j,3] = np.mean(xdata[j,:]**6)
                Moments[i,j,4] = np.mean(xdata[j,:]**8)
                Moments[i,j,5] = np.mean(xdata[j,:]**10)
                Moments[i,j,6] = np.mean(xdata[j,:]**12)


        x,y=caly(xdata,N)    
        Dcoeffs,ind,col=calDlcoef(1,x,y,dt,N,M[0],mode)
        D1array[i]=Dcoeffs.to_numpy()

        if dxf==True: y=calculate_newy(xdata, Dcoeffs, dt)
        
        Dcoeffs,ind2,col2=calDlcoef(2,x,y,dt,N,M[1],mode)
        D2array[i]=Dcoeffs.to_numpy()

        Dcoeffs,ind4,col4=calDlcoef(4,x,y,dt,N,M[2],mode)
        D4array[i]=Dcoeffs.to_numpy()

        Dcoeffs,ind6,col6=calDlcoef(6,x,y,dt,N,M[3],mode)
        D6array[i]=Dcoeffs.to_numpy()

    Parallel(n_jobs=n_parallel)(delayed(calansa)(i) for i in range(nansa))
    # Parallel(n_jobs=n_parallel)(delayed(calansa)(i) for i in tqdm(range(nansa), desc="Processing", ncols=100, dynamic_ncols=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}]'))
    
    # Find all temp hist files
    hist_files = glob.glob(os.path.join(temp_hist_dir, "hist_i*_j*.npz"))
    # Store the ranges to find the global max range
    ranges = {}
    for fname in hist_files:
        data = np.load(fname)
        nbins = data['nbins'].item()
        hist_range = data['hist_range']
        ranges[fname] = hist_range
    # Get the largest absolute value across all hists
    max_abs_list = [max(abs(r[0]), abs(r[1])) for r in ranges.values()]
    global_maxabs = max(max_abs_list)
    # Compute final nbins (always odd)
    half_bins = int(np.ceil(global_maxabs / dx))
    if half_bins % 2 == 0:
        half_bins += 1
    final_nbins = 2 * half_bins - 1

    # Now build the final big array
    Histarray_final = np.zeros((nansa, N, final_nbins))
    for fname in tqdm(hist_files, desc="Padding histograms", disable=True):
        # Extract i, j from filename
        base = os.path.basename(fname)
        i_str = base.split('_')[1][1:]
        j_str = base.split('_')[2][1:].split('.')[0]
        i = int(i_str)
        j = int(j_str)
        # Load data
        with np.load(fname) as data:
            hist = data['hist']
            nbins = data['nbins'].item()

        # Pad to final_nbins
        pad_each_side = (final_nbins - nbins) // 2
        hist_padded = np.pad(hist, (pad_each_side, pad_each_side), mode='constant')

        Histarray_final[i, j, :] = hist_padded

    # Remove all temp files + dir
    shutil.rmtree(temp_hist_dir)
    # print(f"Deleted temp hist dir: {temp_hist_dir}")

    finalmeanD1=np.zeros((D1shape[0],D1shape[1]))
    finalmedD1=np.zeros((D1shape[0],D1shape[1]))
    finalsemD1=np.zeros((D1shape[0],D1shape[1]))

    finalmeanD2=np.zeros((D2shape[0],D2shape[1]))
    finalmedD2=np.zeros((D2shape[0],D2shape[1]))
    finalsemD2=np.zeros((D2shape[0],D2shape[1]))

    finalmeanD4=np.zeros((D4shape[0],D4shape[1]))
    finalmedD4=np.zeros((D4shape[0],D4shape[1]))
    finalsemD4=np.zeros((D4shape[0],D4shape[1]))

    finalmeanD6=np.zeros((D6shape[0],D6shape[1]))
    finalmedD6=np.zeros((D6shape[0],D6shape[1]))
    finalsemD6=np.zeros((D6shape[0],D6shape[1]))

    for i in range(D1shape[0]):
        for j in range(D1shape[1]):
            D1array[:,i,j]=np.sort(D1array[:,i,j])
            finalmedD1[i,j]=np.median(D1array[:,i,j])
            finalmeanD1[i,j]=np.mean(D1array[:,i,j])
            finalsemD1[i,j]=np.std(D1array[:,i,j])/np.sqrt(nansa)


    for i in range(D2shape[0]):
        for j in range(D2shape[1]):
            D2array[:,i,j]=np.sort(D2array[:,i,j])
            finalmedD2[i,j]=np.median(D2array[:,i,j])
            finalmeanD2[i,j]=np.mean(D2array[:,i,j])
            finalsemD2[i,j]=np.std(D2array[:,i,j])/np.sqrt(nansa)


    for i in range(D4shape[0]):
        for j in range(D4shape[1]):
            D4array[:,i,j]=np.sort(D4array[:,i,j])
            finalmedD4[i,j]=np.median(D4array[:,i,j])
            finalmeanD4[i,j]=np.mean(D4array[:,i,j])
            finalsemD4[i,j]=np.std(D4array[:,i,j])/np.sqrt(nansa)

    for i in range(D6shape[0]):
        for j in range(D6shape[1]):
            D6array[:,i,j]=np.sort(D6array[:,i,j])
            finalmedD6[i,j]=np.median(D6array[:,i,j])
            finalmeanD6[i,j]=np.mean(D6array[:,i,j])
            finalsemD6[i,j]=np.std(D6array[:,i,j])/np.sqrt(nansa)


    # xdata=MY_JUMPY_SDEINT(N,F,G,Ksi2,Lambda,dt,nsteps=100,x0=x0,method=method)
    xdata=MY_JUMPY_SDEINT(N,F,G,Ksi2,Lambda,dt,nsteps=100,x0=x0)
    x,y=caly(xdata,N)    
    Dcoeffs,ind,col=calDlcoef(1,x,y,dt,N,M[0],mode)
    Dcoeffs,ind2,col2=calDlcoef(2,x,y,dt,N,M[1],mode)
    Dcoeffs,ind4,col4=calDlcoef(4,x,y,dt,N,M[2],mode)
    Dcoeffs,ind6,col6=calDlcoef(6,x,y,dt,N,M[3],mode)

    dfD=pd.DataFrame(finalmedD1)
    dfD.index=ind
    dfD.columns=col+"_median"
    dfD2=pd.DataFrame(finalmeanD1)
    dfD2.index=ind
    dfD2.columns=col+"_mean"
    dfSEM=pd.DataFrame(finalsemD1)
    dfSEM.index=ind
    dfSEM.columns=col+"_error"
    dfD=pd.concat([dfD, dfD2, dfSEM],axis=1)    
    fp=path1+exn+"-T="+str(int(nsteps*dt))+"-M1-data.csv"
    dfD.to_csv(fp, index=True) 

    dfD=pd.DataFrame(finalmedD2)
    dfD.index=ind2
    dfD.columns=col2+"_median"
    dfD2=pd.DataFrame(finalmeanD2)
    dfD2.index=ind2
    dfD2.columns=col2+"_mean"
    dfSEM=pd.DataFrame(finalsemD2)
    dfSEM.index=ind2
    dfSEM.columns=col2+"_error"
    dfD=pd.concat([dfD, dfD2, dfSEM],axis=1)   
    fp=path1+exn+"-T="+str(int(nsteps*dt))+"-M2-data.csv"
    dfD.to_csv(fp, index=True) 


    dfD=pd.DataFrame(finalmedD4)
    dfD.index=ind4
    dfD.columns=col4+"_median"
    dfD2=pd.DataFrame(finalmeanD4)
    dfD2.index=ind4
    dfD2.columns=col4+"_mean"
    dfSEM=pd.DataFrame(finalsemD4)
    dfSEM.index=ind4
    dfSEM.columns=col4+"_error"
    dfD=pd.concat([dfD, dfD2, dfSEM],axis=1)   
    fp=path1+exn+"-T="+str(int(nsteps*dt))+"-M4-data.csv"
    dfD.to_csv(fp, index=True) 


    dfD=pd.DataFrame(finalmedD6)
    dfD.index=ind6
    dfD.columns=col6+"_median"
    dfD2=pd.DataFrame(finalmeanD6)
    dfD2.index=ind6
    dfD2.columns=col6+"_mean"
    dfSEM=pd.DataFrame(finalsemD6)
    dfSEM.index=ind6
    dfSEM.columns=col6+"_error"
    dfD=pd.concat([dfD, dfD2, dfSEM],axis=1)   
    fp=path1+exn+"-T="+str(int(nsteps*dt))+"-M6-data.csv"
    dfD.to_csv(fp, index=True) 

    return Histarray_final, Moments

def timelistansacalallDcoef(N,F,G,Ksi2,Lambda,dt,x0, method,nansa,M,mode,exn,path1,n_parallel,timelist,dxf, histsave, dx, momentssave):
    print("EXAMPLE=",exn)    
    steplist=[int(time/dt) for time in timelist]
    AllMoments = np.zeros((len(steplist), int(nansa), int(N), 7))
    for nsteps in steplist:
        start=time.perf_counter()
        Histarray, Moments = ansacalallDcoef(N,F,G,Ksi2,Lambda,dt,nsteps,x0,method,nansa,M,mode,exn,path1,n_parallel, dxf, histsave, dx, momentssave)
        AllMoments[steplist.index(nsteps)] = Moments
        np.save(path1+exn+f"_AllHistarray_{int(nsteps*dt)}.npy", Histarray)
        finish=time.perf_counter()
        print("T="+str(int(nsteps*dt)) + f" Finished in {round(finish-start)} s")

    np.save(path1+exn+"_AllMoments.npy", AllMoments)



method="itoSRI2"
nansa=30 ; n_parallel=15


timelist=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536] 



"""   1D#1   """
@njit
def F(x, t):
    return np.array([-5*x[0]**3])
@njit
def G(x, t):
    return np.array([np.sqrt(1+1*x[0]**2+0.5*x[0])])
@njit
def Ksi2(x,t):
    return np.array([[0.1]])
@njit
def Lambda(x,t):
    return np.array([50*x[0]**2])

N=1 ; dt=0.0005 ; x0 = np.array([0]) ; M=[3,3,3,3] ; mode="lim" ; dxf=True ; histsave=True ; momentssave=True
dx=0.01
exn="1D_#1_itoSRI2_5E4_dxfTrue"
timelistansacalallDcoef(N,F,G,Ksi2,Lambda,dt,x0, method,nansa,M,mode,exn,path1,n_parallel,timelist,dxf, histsave, dx, momentssave)
#####################################################################################################################################################
"""   2D#1   """
@njit
def F(x, t):
    return np.array([-5*x[0]**3 - 2*x[1],
                     -5*x[1]**3 - 3*x[0]])

@njit
def G(x, t):
    return np.array([[ np.sqrt(0.7  + x[0]**2 + 2*x[1]**2)     , 0                                  ],
                     [ 0                                     , np.sqrt(0.3 + 2*x[0]**2 + x[1]**2)]])

@njit
def Ksi2(x,t):
    return np.array([ [0.1    , 0.045  ] ,
                      [0.075  , 0.155  ] ])

@njit
def Lambda(x,t):
    return [50*x[0]**2, 25*x[1]**2]

N=2 ; dt=0.0005 ; x0 = np.array([0,0]) ; M=[3,3,3,3] ; mode="lim" ; dxf=True ; histsave=True ; momentssave=True
dx=0.01
exn="2D_#1_itoSRI2_5E4_dxfTrue"
timelistansacalallDcoef(N,F,G,Ksi2,Lambda,dt,x0, method,nansa,M,mode,exn,path1,n_parallel,timelist,dxf, histsave, dx, momentssave)
#####################################################################################################################################################
"""   2D#2   """
@njit
def F(x, t):
    I = 0.5
    return np.array([x[0] - x[0]**3/3 - x[1] + I,
                     0.08*( x[0] + 0.7 - 0.75*x[1] )])

@njit
def G(x, t):
    return np.array([[  0.2  , 0   ],
                     [  0    , 0.2 ] ])

@njit
def Ksi2(x,t):
    return np.array([ [0.2  , 0.09  ] ,
                      [0     , 0.15   ] ])

@njit
def Lambda(x,t):
    return [1, 2]

N=2 ; dt=0.0005 ; x0 = np.array([0,0]) ; M=[3,3,3,3] ; mode="lim" ; dxf=True ; histsave=True ; momentssave=True
dx=0.01
exn="2D_#2_itoSRI2_5E4_dxfTrue"
timelistansacalallDcoef(N,F,G,Ksi2,Lambda,dt,x0, method,nansa,M,mode,exn,path1,n_parallel,timelist,dxf, histsave, dx, momentssave)
#####################################################################################################################################################
"""   3D#1   """
@njit
def F(x, t):
    return np.array([-7.5*x[0]**3 - 2*x[1] - x[2],
                     -7.5*x[1]**3 - 3*x[0] + x[2],
                     -7.5*x[2]**3 -   x[0] - x[1]])

@njit
def G(x, t):
    return np.array([[ np.sqrt(0.2  + 0.5*x[0]**2), 0                                  , 0                                 ],
                     [ 0                                  , np.sqrt(0.4  + 0.75*x[1]**2 ), 0                                 ],
                     [ 0                                  , 0                                  , np.sqrt(0.6 + x[2]**2)]    ])

@njit
def Ksi2(x,t):
    return np.array([ [0.1     , 0.05, 0   ],
                      [0       , 0.15, 0.13],
                      [0.07    , 0.11 , 0   ] ])

@njit
def Lambda(x,t):
    return [40*x[0]**2 + 10   ,   35*x[1]**2 + 15  ,  50*(x[2]**2) + 20 ]

N=3 ; dt=0.0005 ; x0 = np.array([0,0,0]) ; M=[3,3,3,3] ; mode="lim" ; dxf=True ; histsave=True ; momentssave=True
dx=0.01

method="itoSRI2"
exn="3D_#1_itoSRI2_5E4_dxfTrue"
timelistansacalallDcoef(N,F,G,Ksi2,Lambda,dt,x0, method,nansa,M,mode,exn,path1,n_parallel,timelist,dxf, histsave, dx, momentssave)
#####################################################################################################################################################
"""   3D#2   """
@njit
def F(x, t):
    mu = 1
    alpha = 0.1
    beta = 0.5
    return np.array([  -x[0]**3 + x[1]                                ,
                       mu*(1-x[0]**2)*x[1] -x[0] + x[2]    ,
                       -alpha * x[2] - beta*x[0]      ]  )

@njit
def G(x, t):
    return np.array([[ 0.1     , 0      , 0      ],
                     [ 0       , 0.2    , 0      ],
                     [ 0       , 0      , 0.1    ]    ])
 
@njit
def Ksi2(x,t):
    return np.array([ [0.3     , 0    , 0      ],
                      [0       , 0.001   , 0   ],   # it is really (0,0,0)
                      [0        , 0    , 0.2   ] ])

@njit
def Lambda(x,t):
    return [ 8   , 0   ,   4 ]

N=3 ; dt=0.0005 ; x0 = np.array([0,0,0]) ; M=[3,3,3,3] ; mode="lim" ; dxf=True ; histsave=True ; momentssave=True
dx=0.01

method="itoSRI2"
exn="3D_#2_itoSRI2_5E4_dxfTrue"
timelistansacalallDcoef(N,F,G,Ksi2,Lambda,dt,x0, method,nansa,M,mode,exn,path1,n_parallel,timelist,dxf, histsave, dx, momentssave)
#####################################################################################################################################################
"""   3D#3   """
@njit
def F(x, t):
    sigma = 10
    ro = 28
    beta = 8/3
    s = 0.01
    return np.array([  sigma * ( x[1] - x[0] )   ,
                       x[0] * ( ro - x[2]/s ) - x[1]    ,
                       x[0]*x[1]/s - beta*x[2]     ]  )

@njit
def G(x, t):
    s = 0.01
    return np.array([[x[0]*0 + 5  ,     0                  ,    0                    ]      ,
                     [0                  ,     x[1]*0 + 15  ,    0                    ]      ,
                     [0                  ,     0                  ,    x[2]*0 + 25    ]]     )*s

@njit
def Ksi2(x,t):
    return np.array([ [0.3     , 0     , 0      ],
                      [0       , 0.2   , 0      ],
                      [0.2     , 0     , 0.1   ] ])

@njit
def Lambda(x,t):
    return [ 5   ,   20   ,  15*(x[2]**2) + 5 ]

N=3 ; dt=0.0005 ; x0 = np.array([0,0,0]) ; M=[3,3,3,3] ; mode="lim" ; dxf=True ; histsave=True ; momentssave=True
dx=0.01

method="itoSRI2"
exn="3D_#3_itoSRI2_5E4_dxfTrue"
timelistansacalallDcoef(N,F,G,Ksi2,Lambda,dt,x0, method,nansa,M,mode,exn,path1,n_parallel,timelist,dxf, histsave, dx, momentssave)
#####################################################################################################################################################
timelist=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] 
"""   4D#1   """
@njit
def F(x, t):
    return np.array([   -10*x[0]**3 + x[1] + x[2] + x[3]      ,
                        -10*x[1]**3 + x[0] + x[2] + x[3]     ,
                        -10*x[2]**3 + x[0] + x[1] + x[3]     ,
                        -10*x[3]**3 + x[0] + x[1] + x[2]     ]  )

@njit
def G(x, t):
    return np.array([[0.2  ,  0.3  ,  0.4  ,  0.5],
                     [0    ,  0.6  ,  0.7  ,  0.8],
                     [0    ,  0    ,  0.9  ,  0.9  ],
                     [0    ,  0    ,  0    ,  1.1]       ]     )

@njit
def Ksi2(x,t):
    return np.array([ [0.1  , 0    , 0.1   , 0      ],
                      [0    , 0.4  , 0     , 0.1    ],
                      [0.2  , 0    , 0.3   , 0      ],
                      [0    , 0.1  , 0     , 0.2    ] ])

@njit
def Lambda(x,t):
    return [40*x[0]**2    ,   60*x[1]**2  ,  20*x[2]**2  ,  50*x[3]**2 ]


N=4 ; dt=0.0005 ; x0 = np.array([0,0,0,0]) ; M=[3,3,3,3] ; mode="lim" ; dxf=True ; histsave=True ; momentssave=True
dx=0.01
exn="4D_#1_itoSRI2_5E4_dxfTrue"
timelistansacalallDcoef(N,F,G,Ksi2,Lambda,dt,x0, method,nansa,M,mode,exn,path1,n_parallel,timelist,dxf, histsave, dx, momentssave)
#####################################################################################################################################################
timelist=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] 
"""   4D#2   """
@njit
def F(x, t):
    F = 2.5
    return np.array([   (x[1] - x[2] ) * x[3] - x[0] + F     ,
                        (x[2] - x[3] ) * x[0] - x[1] + F     ,
                        (x[3] - x[0] ) * x[1] - x[2] + F     ,
                        (x[0] - x[1] ) * x[2] - x[3] + F     ]  )

@njit
def G(x, t):
    return np.array([[0.2  ,  0.3  ,  0.4  ,  0.5],
                     [0    ,  0.6  ,  0.7  ,  0.8],
                     [0    ,  0    ,  0.9  ,  1.1],
                     [0    ,  0    ,  0    ,  1.1]       ]     )

@njit
def Ksi2(x,t):
    return np.array([ [0.1  , 0.05 , 0     , 0      ],
                      [0    , 0    , 0.06  , 0.11   ],
                      [0    , 0.08 , 0.04  , 0.12   ],
                      [0.06 , 0    , 0     , 0      ] ])

@njit
def Lambda(x,t):
    return [25   ,   35  ,  35  , 25 ]

N=4 ; dt=0.0005 ; x0 = np.array([0,0,0,0]) ; M=[3,3,3,3] ; mode="lim" ; dxf=True ; histsave=True ; momentssave=True
dx=0.01
method="itoSRI2"
exn="4D_#2_itoSRI2_5E4_dxfTrue"
timelistansacalallDcoef(N,F,G,Ksi2,Lambda,dt,x0, method,nansa,M,mode,exn,path1,n_parallel,timelist,dxf, histsave, dx, momentssave)
#####################################################################################################################################################
timelist=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] 
"""   4D#3   """
@njit
def F(x, t):
    F = 8
    return np.array([   (x[1] - x[2] ) * x[3] - x[0] + F     ,
                        (x[2] - x[3] ) * x[0] - x[1] + F     ,
                        (x[3] - x[0] ) * x[1] - x[2] + F     ,
                        (x[0] - x[1] ) * x[2] - x[3] + F     ]  )

@njit
def G(x, t):
    return np.array([[0.2  ,  0    ,  0.4  ,  0  ],
                     [0    ,  0.6  ,  0    ,  0.8],
                     [0    ,  0    ,  0.9  ,  0  ],
                     [0    ,  0    ,  0    ,  1.1]       ]     )

@njit
def Ksi2(x,t):
    return np.array([ [0.1  , 0    , 0.1   , 0      ],
                      [0    , 0.4  , 0     , 0.1    ],
                      [0.2  , 0    , 0.3   , 0      ],
                      [0    , 0.1  , 0     , 0.2    ] ])

@njit
def Lambda(x,t):
    return [10  , 10  ,  10  ,  10 ]

N=4 ; dt=0.0005 ; x0 = np.array([0,0,0,0]) ; M=[3,3,3,3] ; mode="lim" ; dxf=True ; histsave=True ; momentssave=True
dx=0.1
method="itoSRI2"
exn="4D_#3_itoSRI2_5E4_dxfTrue"
timelistansacalallDcoef(N,F,G,Ksi2,Lambda,dt,x0, method,nansa,M,mode,exn,path1,n_parallel,timelist,dxf, histsave, dx, momentssave)


