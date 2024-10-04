# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:06:53 2024

@author: Quentin Ploussard
"""

import numpy as np
import pandas as pd
from ortools.math_opt.python import mathopt
import time
import matplotlib.pyplot as plt


def clean_linreg_data(data, p, keepMinMaxOnly=True, reverseOrder=False):
    
    # Remove nan values, round numbers, and only keep y_min and y_max for same x values
    data2 = data[~np.isnan(data[:,0]) & ~np.isnan(data[:,1]),:]
    x = data2[:,0]
    y = data2[:,1]
    x2 = np.array([(0. if xelem==0. else round(xelem, -int(np.floor(np.log10(abs(xelem))))+p-1)) for xelem in x])
    y2 = np.array([(0. if yelem==0. else round(yelem, -int(np.floor(np.log10(abs(yelem))))+p-1))for yelem in y])    
    df = pd.DataFrame({'x':x2,'y':y2})
    df_min = df.groupby('x').min().reset_index()
    df_max = df.groupby('x').max().reset_index()
    if keepMinMaxOnly:
        df = pd.concat([df_min, df_max], axis=0)
    # dfUnique = df.drop_duplicates(subset=['x','y']).sort_values('y').sort_values('x')
    dfUnique = df.drop_duplicates(subset=['x','y'])
    data2 = np.zeros(len(dfUnique), dtype=[('x',float),('y',float)])
    data2['x'] = dfUnique.x.values
    data2['y'] = dfUnique.y.values
    if reverseOrder:
        data2['y'] = - data2['y']
    data2 = np.sort(data2, order=['x', 'y'])
    if reverseOrder:
        data2['y'] = - data2['y']
    # x = dfUnique.x.values
    # y = dfUnique.y.values
    maxDiff = max((df_max-df_min).y)
    minErr = maxDiff/2    
    
    # return minErr, np.array([x,y]).T
    return minErr, np.array([data2['x'],data2['y']]).T


def rescale_piecewise_linear_problem(data, p):
    
    x = data[:,0]
    y = data[:,1]
    xMin = min(x)
    xMax = max(x)
    dx = xMax - xMin
    yMin = min(y)
    yMax = max(y)
    dy = yMax - yMin
    
    dataBounds = np.zeros((2,2))
    dataBounds[0,0] = xMin
    dataBounds[1,0] = xMax
    dataBounds[0,1] = yMin
    dataBounds[1,1] = yMax
    
    xx = (x - xMin)/dx
    yy = (y - yMin)/dy
    N = len(xx)
    for i in range(N):
        if xx[i]>0:
            xx[i] = round(xx[i], -int(np.floor(np.log10(abs(xx[i]))))+p-1)
        if yy[i]>0:
            yy[i] = round(yy[i], -int(np.floor(np.log10(abs(yy[i]))))+p-1)
    
    dataResc = np.array([xx,yy]).T
    
    return dataBounds, dataResc



def optimal_piecewise_linear_approximation_binary_search2(data, err, crossSeg=False, reverseOrder=False, commonPoint = False, p=7):    

    start = time.time()

    data0 = np.array(data)

    if reverseOrder:
        data0[:,0] = -data0[:,0]
        data0 = np.flip(data0,0)

    # Remove nan values, round numbers, and only keep y_min and y_max for same x values
    minErr, data2 = clean_linreg_data(data0, p)
    x = data2[:,0]
    y = data2[:,1]   
    
    N = len(x)
    LinSeg = np.zeros((N,4))
    LinSeg[:] = np.nan
    
    sizeLP = np.zeros(10*N)
    sizeLP[:] = np.nan
    idxLP = 0

    print(f"Reduced to {len(x)} datapoints")
    
    if err<minErr:
        print("The input error is too low. Please provide an input error value of at least "+str(minErr))
    else:        
        idxLS = 0
        
        # group x values that are identical
        uniqVal, idxUniq, invUniq, uniqCounts = np.unique(x, return_index=True, return_inverse=True, return_counts=True)
        NbUniq = len(idxUniq)
        
        print(str(NbUniq)+" datapoints with unique x values")
        
        kp = 0
        
        kqMax = int(NbUniq)
        if crossSeg:
            kqMin = int(kp)
        else:
            kqMin = int(kp+1)
        
        while kp < NbUniq:
            
            print("Segment "+str(idxLS+1))
            
            p = idxUniq[kp]
            
            LinSeg[idxLS,0] = x[p]
            
            # if p==N-1
            if idxUniq[kp] + uniqCounts[kp]==N:
                
                if crossSeg and idxLS>0:
                    LinSeg[idxLS,1] = x[N-1] 
                    LinSeg[idxLS,2] = (y[N-1] - LinSeg[idxLS-1,2]*x[N-2] - LinSeg[idxLS-1,3])/(x[N-1] - x[N-2])
                    LinSeg[idxLS,3] = y[N-1] - LinSeg[idxLS,2]*x[N-1]
                    kp=NbUniq
                    continue
                else:
                    LinSeg[idxLS,1] = x[N-1] 
                    LinSeg[idxLS,2] = 0
                    LinSeg[idxLS,3] = y[N-1]
                    kp=NbUniq
                    continue

            if crossSeg and idxLS>0: 
                
                # first binary search using LP1
                while kqMax>kqMin:

                    kq = int((kqMax+kqMin)/2)                    
                    q = idxUniq[kq] + uniqCounts[kq] - 1                    

                    subIdx = list(range(p-1,q+1))
                   
                    # calculate the analytical solution if p==q
                    if p==q:
                        slope1 = (y[q] - LinSeg[idxLS-1,2]*x[q-1] - LinSeg[idxLS-1,3])/(x[q] - x[q-1])
                        intercept1 = y[q] - slope1*x[q]
                        lastX1 = float(x[q]) 
                        kqMin = kq+1
                    else:
                        
                        slope, intercept, currErr = linear_regression_max_error(x[subIdx],y[subIdx],crossSeg=1,akp=LinSeg[idxLS-1,2],bkp=LinSeg[idxLS-1,3])
                        sizeLP[idxLP] = q-p+2
                        idxLP = idxLP+1
                        
                        if currErr>err:
                            kqMax = int(kq)
                        else:
                            slope1 = float(slope)
                            intercept1 = float(intercept)
                            lastX1 = float(x[q])   
                            kqMin = kq+1        
                    
                kqMin1 = int(kqMin)

                # second binary search using LP2
                kqMin = int(kp)
                kqMax = int(NbUniq)
            
                while kqMax>kqMin:

                    kq = int((kqMax+kqMin)/2)                    
                    q = idxUniq[kq] + uniqCounts[kq] - 1                       

                    subIdx = list(range(p-1,q+1))

                    if p==q:
                        slope2 = (y[q] - LinSeg[idxLS-1,2]*x[q-1] - LinSeg[idxLS-1,3])/(x[q] - x[q-1])
                        intercept2 = y[q] - slope1*x[q]
                        lastX2 = float(x[q]) 
                        kqMin = kq+1
                    else:
                        slope, intercept, currErr = linear_regression_max_error(x[subIdx],y[subIdx],crossSeg=2,akp=LinSeg[idxLS-1,2],bkp=LinSeg[idxLS-1,3])
                        sizeLP[idxLP] = q-p+2
                        idxLP = idxLP+1
                    
                        if currErr>err:
                            kqMax = int(kq)
                        else:
                            slope2 = float(slope)
                            intercept2 = float(intercept)
                            lastX2 = float(x[q])   
                            kqMin = kq+1      
                    
                kqMin2 = int(kqMin)

                if kqMin1>=kqMin2:
                    kqMin = int(kqMin1)
                    LinSeg[idxLS,2] = slope1
                    LinSeg[idxLS,3] = intercept1
                    LinSeg[idxLS,1] = lastX1
                else:
                    kqMin = int(kqMin2)
                    LinSeg[idxLS,2] = slope2
                    LinSeg[idxLS,3] = intercept2
                    LinSeg[idxLS,1] = lastX2                   

            else:
                
                while kqMax>kqMin:
                    
                    kq = int((kqMax+kqMin)/2)                    
                    q = idxUniq[kq] + uniqCounts[kq] - 1
                    
                    subIdx = list(range(p,q+1))
                    slope, intercept, currErr = linear_regression_max_error(x[subIdx],y[subIdx],crossSeg=0)
                    sizeLP[idxLP] = q-p+1
                    idxLP = idxLP+1
                    
                    if currErr>err:
                        kqMax = int(kq)
                    else:
                        LinSeg[idxLS,2] = slope
                        LinSeg[idxLS,3] = intercept
                        LinSeg[idxLS,1] = x[q]   
                        kqMin = kq+1        
            
            if commonPoint:
                kp = kqMin-1
                if kp == NbUniq-1:
                    break
            else:
                kp = int(kqMin)
                
            
            if crossSeg:
                kqMin = int(kp)
            else:
                kqMin = kp+1
            
            kqMax = int(NbUniq)
            
            idxLS = idxLS+1
            
        LinSeg = LinSeg[~np.isnan(LinSeg[:,0])]  
        sizeLP = sizeLP[~np.isnan(sizeLP)]
        
        if reverseOrder:
            LinSeg = np.flip(LinSeg,0)
            LinSeg[:,0:2] = np.flip(LinSeg[:,0:2],1)
            LinSeg[:,0:3] = -LinSeg[:,0:3]
            x = np.flip(-x)
        
        end = time.time()
        print(end - start)
        print(str(len(sizeLP))+" LP problems computed of average size "+str(np.mean(sizeLP)))
        print("Theoritical max: "+str(len(LinSeg)*(1+np.log2(NbUniq-1-len(LinSeg)))))
        
        C = np.zeros(N)
        for k in range(len(LinSeg)):
            C[(x>=LinSeg[k,0]) & (x<=LinSeg[k,1])] = k+1
    
    return C, LinSeg


def make_piecewise_linear_continuous2(data, err, LinSeg, p=7):
    
    minErr, data2 = clean_linreg_data(data, p, keepMinMaxOnly=True)
    
    C, LinSegGap = build_segment_index_vector(data2, LinSeg, p)
    
    N = len(LinSegGap)
    
    LinSeg2 = []
    prevBkPt = LinSegGap[0,0]

    for k in range(N-1):
        lk1 = prevBkPt
        uk1 = LinSegGap[k,1]
        lk2 = LinSegGap[k+1,0]
        uk2 = LinSegGap[k+1,1]
        ak1 = LinSegGap[k,2]
        bk1 = LinSegGap[k,3]
        ak2 = LinSegGap[k+1,2]
        bk2 = LinSegGap[k+1,3]
                   
        # crossVec[k] = ((ak2-ak1)*uk1 + (bk2-bk1))*((ak2-ak1)*lk2 + (bk2-bk1)) <= np.power(10.,-p)
        segmentsInterset = ((ak2-ak1)*lk1 + (bk2-bk1))*((ak2-ak1)*uk2 + (bk2-bk1)) <= np.power(10.,-p)
        
        addSegment = True
        
        if segmentsInterset:
            bkPt = (bk1 - bk2)/(ak2 - ak1)
            newSegk1 = (data2[:,0]>=lk1) & (data2[:,0]<=bkPt)
            newSegk2 = (data2[:,0]>=bkPt) & (data2[:,0]<=uk2)
            belowErr = max(max(abs(ak1*data2[newSegk1,0]+bk1 - data2[newSegk1,1])),max(abs(ak2*data2[newSegk2,0]+bk2 - data2[newSegk2,1]))) <= err
            if belowErr:
                addSegment = False
        
        if addSegment:
            # idxSegk1 = np.where(C==k+1)[0]
            # idxSegk2 = np.where(C==k+2)[0]
            # q1 = idxSegk1[-1]
            # p2 = idxSegk2[0]
            LinSeg2.append([lk1,uk1,ak1,bk1])
            aa = (ak1*uk1+bk1-ak2*lk2-bk2)/(uk1-lk2)
            bb = ak1*uk1+bk1 - uk1*aa
            LinSeg2.append([uk1,lk2,aa,bb])
            prevBkPt = lk2
        else:
            LinSeg2.append([lk1,bkPt,ak1,bk1])
            prevBkPt = bkPt

    LinSeg2.append([prevBkPt,LinSegGap[N-1,1],LinSegGap[N-1,2],LinSegGap[N-1,3]])
    
    LinSeg2 = np.array(LinSeg2)
        
    return LinSeg2




def build_segment_index_vector(data, LinSeg, p=7):
    
    minErr, data2 = clean_linreg_data(data, p)
    
    N = len(data2)
    Sn = np.zeros(N,dtype='int64')
    K = len(LinSeg)
    
    for k in range(K):
        Sn[(data2[:,0]>=LinSeg[k,0]) & (data2[:,0]<=LinSeg[k,1])] = k+1
        
    uniqVal = np.unique(Sn)
    LinSeg2 = np.copy(LinSeg[uniqVal-1,:])
    KK = len(uniqVal)
    
    for kk in range(KK):
        mask = (Sn==uniqVal[kk])
        Sn[mask] = kk+1
        LinSeg2[kk,0] = data2[np.where(mask)[0][0],0]
        LinSeg2[kk,1] = data2[np.where(mask)[0][-1],0]
            
    return Sn, LinSeg2


def bigM_parameter(data,err,p=7):
    
    minErr, data2 = clean_linreg_data(data, p)
    N = len(data2)
    
    dataBounds, dataResc = rescale_piecewise_linear_problem(data2, p)
    xx = dataResc[:,0]
    yy = dataResc[:,1]
    yMin = dataBounds[0,1]
    yMax = dataBounds[1,1]
    dy = yMax - yMin
    errResc = err/dy    
    
    df = pd.DataFrame(dataResc)
    x_unique = df.groupby(0,as_index=False).max()[0].to_numpy()
    y_min = df.groupby(0,as_index=False).min()[1].to_numpy()
    y_max = df.groupby(0,as_index=False).max()[1].to_numpy()
    z = np.maximum(abs(y_max[1:] - y_min[:-1]),abs(y_min[1:] - y_max[:-1]))
    maxSlope = max(np.divide(z+2*errResc,abs(x_unique[1:]-x_unique[:-1])))            
    bigMerr = np.max(abs(np.tile(yy.reshape((-1,1)),N)-np.tile(yy.reshape((-1,1)),N).T) + maxSlope*abs(np.tile(xx.reshape((-1,1)),N)-np.tile(xx.reshape((-1,1)),N).T),axis=0) + errResc
    bigMcont = np.max(maxSlope*(abs(np.tile(xx[:-1].reshape((-1,1)),N)-np.tile(xx.reshape((-1,1)),N-1).T) + abs(np.tile(xx[1:].reshape((-1,1)),N)-np.tile(xx.reshape((-1,1)),N-1).T)) + abs(np.tile((yy[:-1]-yy[1:]).reshape((-1,1)),N)),axis=0) + 2*errResc
    bigMslopeVar = 2*maxSlope

    return bigMerr, bigMcont, bigMslopeVar


def linear_regression_max_error(x, y, p=7, solverType=mathopt.SolverType.GLOP, crossSeg=0, akp=0, bkp=0):

    # rescale the problem 
    data = np.array([x,y]).T
    dataBounds, dataResc = rescale_piecewise_linear_problem(data, p)
    xx = dataResc[:,0]
    yy = dataResc[:,1]
    xMin = dataBounds[0,0]
    xMax = dataBounds[1,0]
    dx = xMax - xMin
    yMin = dataBounds[0,1]
    yMax = dataBounds[1,1]
    dy = yMax - yMin
    
    akpp = akp*dx/dy
    bkpp = (bkp - yMin + akp*xMin)/dy
    
    model = mathopt.Model(name="OPWLA")
    
    slope = model.add_variable()
    intercept = model.add_variable()
    bound = model.add_variable(lb=0.0)
    
    model.minimize(bound)
    
    N = len(x)

    if crossSeg==0: 
        for i in range(N):
            model.add_linear_constraint(slope*xx[i] + intercept - yy[i] - bound <= 0)
            model.add_linear_constraint(slope*xx[i] + intercept - yy[i] + bound >= 0)
    elif crossSeg==1:
        model.add_linear_constraint(slope*xx[0] + intercept - akpp*xx[0] - bkpp >= 0)
        model.add_linear_constraint(akpp*xx[1] + bkpp - slope*xx[1] - intercept >= 0)
        for i in range(1,N):
            model.add_linear_constraint(slope*xx[i] + intercept - yy[i] - bound <= 0)
            model.add_linear_constraint(slope*xx[i] + intercept - yy[i] + bound >= 0)
    elif crossSeg==2:
        model.add_linear_constraint(slope*xx[0] + intercept - akpp*xx[0] - bkpp <= 0)
        model.add_linear_constraint(akpp*xx[1] + bkpp - slope*xx[1] - intercept <= 0)       
        for i in range(1,N):
            model.add_linear_constraint(slope*xx[i] + intercept - yy[i] - bound <= 0)
            model.add_linear_constraint(slope*xx[i] + intercept - yy[i] + bound >= 0)
    
    params = mathopt.SolveParameters(enable_output=False)
    result = mathopt.solve(model, solverType, params=params)
    
    err = result.variable_values()[bound]*dy
    slopeVal = result.variable_values()[slope]*dy/dx
    interceptVal = yMin - result.variable_values()[slope]*xMin*dy/dx + result.variable_values()[intercept]*dy
    
    return slopeVal, interceptVal, err


    
def optimal_piecewise_linear_approximation_MILP(data, err, K, Slb = 0, crossSeg = 'unconstrained', solverType=mathopt.SolverType.GSCIP, coOpt=False, optErrorOnly=False, useSegIdx=False, gap = 0.01, bigM = 20, optBigM = False, rescale=True, p = 7, ws=False, wsLinSeg=[], applyBounds=False, lbLinSeg=[], ubLinSeg=[], dummyFix=False, fixWs=False):

    start = time.time()  
    
    # Remove nan values, round numbers, and only keep y_min and y_max for same x values
    minErr, data2 = clean_linreg_data(data, p)
    x = data2[:,0]
    # y = data2[:,1]
    
    N = len(x)
    dp = range(N)
    ls = range(K)
    
    LinSeg = np.zeros((K,4))
    LinSeg[:] = np.nan
    
    C = np.full((N,1), np.nan)
    C2 = np.full((N,1), np.nan)
    c = np.full((N,K), np.nan)
    c2 = np.full((N,K), np.nan)
    
    precision=3
    
    if err<minErr:
        print(f"The input error is too low. Please provide an input error value of at least {minErr}")
    else:        
       
        if rescale:
            # clean and rescale the data set 
            dataBounds, dataResc = rescale_piecewise_linear_problem(data2, p)
            xx = dataResc[:,0]
            yy = dataResc[:,1]
            xMin = dataBounds[0,0]
            xMax = dataBounds[1,0]
            dx = xMax - xMin
            yMin = dataBounds[0,1]
            yMax = dataBounds[1,1]
            dy = yMax - yMin
            errResc = err/dy
        else:
            xx = data[:,0]
            yy = data[:,1]
            dataResc = data
            xMin = min(xx)
            xMax = max(xx)
            yMin = min(yy)
            yMax = max(yy)
            dx = xMax - xMin
            dy = yMax - yMin
            errResc = err
            
        
        # optimize big-M
        if optBigM:
            df = pd.DataFrame(dataResc)
            x_unique = df.groupby(0,as_index=False).max()[0].to_numpy()
            y_min = df.groupby(0,as_index=False).min()[1].to_numpy()
            y_max = df.groupby(0,as_index=False).max()[1].to_numpy()
            z = np.maximum(abs(y_max[1:] - y_min[:-1]),abs(y_min[1:] - y_max[:-1]))
            maxSlope = max(np.divide(z+2*errResc,abs(x_unique[1:]-x_unique[:-1])))            
            bigMerr = np.max(abs(np.tile(yy.reshape((-1,1)),N)-np.tile(yy.reshape((-1,1)),N).T) + maxSlope*abs(np.tile(xx.reshape((-1,1)),N)-np.tile(xx.reshape((-1,1)),N).T),axis=0) + errResc
            bigMcont = np.max(maxSlope*(abs(np.tile(xx[:-1].reshape((-1,1)),N)-np.tile(xx.reshape((-1,1)),N-1).T) + abs(np.tile(xx[1:].reshape((-1,1)),N)-np.tile(xx.reshape((-1,1)),N-1).T)) + abs(np.tile((yy[:-1]-yy[1:]).reshape((-1,1)),N)),axis=0) + 2*errResc
            bigMslopeVar = np.ceil(2*maxSlope*10**(-int(np.floor(np.log10(abs(2*maxSlope))))+precision-1) )/(10**(-int(np.floor(np.log10(abs(2*maxSlope))))+precision-1))
            bigMerr = np.array([(0. if xelem==0. else np.ceil(xelem*10**(-int(np.floor(np.log10(abs(xelem))))+precision-1) )/(10**(-int(np.floor(np.log10(abs(xelem))))+precision-1))) for xelem in bigMerr])
            bigMcont = np.array([(0. if xelem==0. else np.ceil(xelem*10**(-int(np.floor(np.log10(abs(xelem))))+precision-1) )/(10**(-int(np.floor(np.log10(abs(xelem))))+precision-1))) for xelem in bigMcont])
        else:
            bigMslopeVar = bigM
            bigMerr = np.tile(bigM,N)
            bigMcont = np.tile(bigM,N)
            
        
        
        # midpoint values
        xxMid = (xx[1:] + xx[:-1])/2
        
        # count number of binary variables and those fixed
        NberBin = K*N
        NberBinFixed = 0
    
        model = mathopt.Model(name="CPWL_MILP")
    
        slope = [model.add_variable() for s in ls]
        intercept = [model.add_variable() for s in ls]
        cDec = [[model.add_binary_variable() for s in ls] for p in dp]
        segNber = [model.add_variable(lb=0.0) for p in dp]
        errVar = model.add_variable(lb=0.0)
        lastSegNber = model.add_variable(lb=Slb-1)

        
        if crossSeg=='inBetween':
            crossDir = [model.add_binary_variable() for s in ls]
            NberBin = NberBin + K-1
          
        
        if coOpt:
            model.minimize(lastSegNber + 1 + 0.5*errVar/errResc)
        else:
            if optErrorOnly:
                model.minimize(1*errVar/errResc)
            else:
                model.minimize(lastSegNber + 1)
        
        for i in dp:
            for j in ls:
                model.add_linear_constraint(slope[j]*xx[i] + intercept[j] - yy[i] - errVar - bigMerr[i]*(1 - cDec[i][j]) <= 0)
                model.add_linear_constraint(slope[j]*xx[i] + intercept[j] - yy[i] + errVar + bigMerr[i]*(1 - cDec[i][j]) >= 0)
        
        if not(optErrorOnly):
            model.add_linear_constraint(errVar - errResc <= 0)
        
        # prob += lastSegNber + 1 - Slb >= 0, ("SegNberLowBound") 
        
        for i in dp:
            model.add_linear_constraint(sum([cDec[i][j] for j in ls]) - 1 ==0)
        
        if not(optErrorOnly) or useSegIdx:
            for i in dp:
                model.add_linear_constraint(segNber[i] - sum([cDec[i][j]*j for j in ls]) == 0)
                
            for i in range(1,N):
                model.add_linear_constraint(segNber[i] - segNber[i-1] >= 0)
                
            for i in range(1,N):
                model.add_linear_constraint(segNber[i] - segNber[i-1] - 1 <= 0)

        for i in range(1,N):
            model.add_linear_constraint(cDec[i][0] - cDec[i-1][0] <= 0)
            for j in range(1,K):
                model.add_linear_constraint(cDec[i][j] - cDec[i-1][j-1] - cDec[i-1][j] <= 0)

        for i in range(1,N):
            for j in ls:
                if xx[i] == xx[i-1]:
                    model.add_linear_constraint(cDec[i][j] - cDec[i-1][j] == 0)

        if crossSeg=='midPoint':
            for i in range(0,N-1):
                for j in range(1,K):
                    model.add_linear_constraint(slope[j]*xxMid[i] + intercept[j] - slope[j-1]*xxMid[i] - intercept[j-1] - bigM*(2 - cDec[i+1][j] - cDec[i][j-1]) <= 0)
                    model.add_linear_constraint(slope[j]*xxMid[i] + intercept[j] - slope[j-1]*xxMid[i] - intercept[j-1] + bigM*(2 - cDec[i+1][j] - cDec[i][j-1]) >= 0)
        elif crossSeg=='inBetween':
            crossDir[0].lower_bound = 0
            crossDir[0].upper_bound = 0 
            # NberBinFixed = NberBinFixed + 1
            for j in range(1,K):
                model.add_linear_constraint(slope[j] - slope[j-1] + bigMslopeVar*crossDir[j] >= 0)
                model.add_linear_constraint(slope[j] - slope[j-1] - bigMslopeVar*(1 - crossDir[j]) <= 0)
            for i in  range(1,N):
                for j in range(1,K):
                    model.add_linear_constraint((slope[j] - slope[j-1])*xx[i] + (intercept[j] - intercept[j-1]) + bigMcont[i]*(2 + crossDir[j] - cDec[i][j] - cDec[i-1][j-1]) >= 0)
                    model.add_linear_constraint((slope[j] - slope[j-1])*xx[i] + (intercept[j] - intercept[j-1]) - bigMcont[i]*(3 - crossDir[j] - cDec[i][j] - cDec[i-1][j-1]) <= 0)
                    model.add_linear_constraint((slope[j] - slope[j-1])*xx[i-1] + (intercept[j] - intercept[j-1]) + bigMcont[i-1]*(3 - crossDir[j] - cDec[i][j] - cDec[i-1][j-1]) >= 0)
                    model.add_linear_constraint((slope[j] - slope[j-1])*xx[i-1] + (intercept[j] - intercept[j-1]) - bigMcont[i-1]*(2 + crossDir[j] - cDec[i][j] - cDec[i-1][j-1]) <= 0)
                                       
        # Border conditions
        if not(optErrorOnly) or useSegIdx:
            model.add_linear_constraint(lastSegNber - segNber[N-1] == 0)
        cDec[0][0].lower_bound = 1
        cDec[0][0].upper_bound = 1 
        # NberBinFixed = NberBinFixed + 1
        for j in ls:
            for l in range(j+1,K):
                cDec[j][l].lower_bound = 0
                cDec[j][l].upper_bound = 0
                # NberBinFixed = NberBinFixed + 1

        # Segment number bounds
        if applyBounds:
            lbS = len(lbLinSeg)
            for j in ls:
                if j < lbS:
                    for i in dp:
                        if (x[i] >= lbLinSeg[j,0] and x[i] <= lbLinSeg[j,1]):
                            if not(optErrorOnly) or useSegIdx:
                                model.add_linear_constraint(segNber[i] - j >= 0)
                            for l in range(j-1):
                                cDec[i][l].lower_bound = 0
                                cDec[i][l].upper_bound = 0
                                NberBinFixed = NberBinFixed + 1
            upS = len(ubLinSeg)
            for j in ls:
                if j < upS:
                    for i in dp:
                        if (x[i] >= ubLinSeg[j,0] and x[i] <= ubLinSeg[j,1]):
                            if not(optErrorOnly) or useSegIdx:
                                model.add_linear_constraint(segNber[i] - j - lastSegNber + upS-1 <= 0)
                            for l in range(upS-1-j):
                                cDec[i][K-1-l].lower_bound = 0
                                cDec[i][K-1-l].upper_bound = 0
                                NberBinFixed = NberBinFixed + 1
                                
            if lbS==K and upS==K:
                for j in ls:
                    if j < upS:
                        for i in dp:
                            if (x[i] >= ubLinSeg[j,0] and x[i] <= ubLinSeg[j,1] and x[i] >= lbLinSeg[j,0] and x[i] <= lbLinSeg[j,1]):
                                cDec[i][j].lower_bound = 1
                                cDec[i][j].upper_bound = 1
                                NberBinFixed = NberBinFixed + 1       
                                    

        # Warm Start
        model_params = mathopt.ModelSolveParameters()
        if ws:
            dictWS = {}
            S = len(wsLinSeg)
            dictWS[lastSegNber] = S-1
            for j in ls:
                if j < S:
                    dictWS[slope[j]] = round(wsLinSeg[j,2]*dx/dy,p)
                    dictWS[intercept[j]] = round((wsLinSeg[j,3] - yMin + wsLinSeg[j,2]*xMin)/dy,p)
                    if j>0 and crossSeg=='inBetween':
                        if wsLinSeg[j,2]>=wsLinSeg[j-1,2]:
                            dictWS[crossDir[j]] = 0
                        else:
                            dictWS[crossDir[j]] = 1
                    for i in dp:
                        if (x[i] >= wsLinSeg[j,0] and x[i] <= wsLinSeg[j,1]):
                            if not(optErrorOnly) or useSegIdx:
                                dictWS[segNber[i]] = j
                            dictWS[cDec[i][j]] = 1
                        else:
                            dictWS[cDec[i][j]] = 0
            model_params = mathopt.ModelSolveParameters(solution_hints=[mathopt.SolutionHint(variable_values=dictWS)])
            

        # Fixing warm start
        # if fixWs:
        #     S = len(wsLinSeg)
        #     lastSegNber.fixValue() 
        #     for j in ls:
        #         if j < S:
        #             slope[j].fixValue() 
        #             intercept[j].fixValue() 
        #             if j>0 and crossSeg=='inBetween':
        #                 crossDir[j].fixValue() 
        #             for i in dp:
        #                 if not(optErrorOnly) or useSegIdx:
        #                     segNber[i].fixValue() 
        #                 cDec[i][j].fixValue() 
        
        # dummy fix to check warm fixing variables work
        # if dummyFix:
        #     for i in dp:
        #         for j in ls:
        #             if j==0:
        #                 cDec[i][j].setInitialValue(1)
        #             else:
        #                 cDec[i][j].setInitialValue(0)
        #             cDec[i][j].fixValue() 

        print(f"Number of binary variables: {NberBin}")
        print(f"Number of binary variable fixed: {NberBinFixed}")

        params = mathopt.SolveParameters(enable_output=True, threads=16, relative_gap_tolerance=gap)
        result = mathopt.solve(model, solverType, params=params, model_params=model_params)

        if result.termination.reason == mathopt.TerminationReason.OPTIMAL:
            print("Status: Optimal")
            print(f"Objective function value: {result.objective_value()}")
        else:
            print(f"Status: {result.termination.reason}")

            
        for i in dp:
            for j in ls:
                c2[i][j] = result.variable_values()[cDec[i][j]]
                c[i][j] = round(result.variable_values()[cDec[i][j]])
        
        for i in dp:
            if optErrorOnly and not(useSegIdx):
                C2[i] = np.sum(c2[i]*np.arange(K))
                C[i] = np.round(C2[i])
            else:
                C2[i] = result.variable_values()[segNber[i]]
                C[i] = round(result.variable_values()[segNber[i]])
    
        for j in ls:
            if len(np.where(C==j)[0])>0:
                LinSeg[j,0] = x[np.where(C==j)[0][0]]
                LinSeg[j,1] = x[np.where(C==j)[0][-1]]
                LinSeg[j,2] = result.variable_values()[slope[j]]*dy/dx
                LinSeg[j,3] = yMin - result.variable_values()[slope[j]]*xMin*dy/dx + result.variable_values()[intercept[j]]*dy
    
        LinSeg = LinSeg[~np.isnan(LinSeg[:,0])] 
    
    end = time.time()
    print(end - start)
    
    return C2, c2, LinSeg, model


def rescale_PWL_solution(LinSeg, xMin, xMax, yMin, yMax):
    
    LinSegResc = np.copy(LinSeg)
    dx = xMax-xMin
    dy = yMax-yMin
    # LinSegResc[:,0] = xMin + dx*LinSeg[:,0]
    # LinSegResc[:,1] = xMin + dx*LinSeg[:,1]
    LinSegResc[:,3] = yMin - xMin*(dy/dx)*LinSeg[:,2] + dy*LinSeg[:,3]
    LinSegResc[:,2] = (dy/dx)*LinSeg[:,2]
    xCross = -np.divide(np.diff(LinSegResc[:,3]),np.diff(LinSegResc[:,2]))
    LinSegResc[0,0] = xMin
    LinSegResc[-1,1] = xMax
    LinSegResc[1:,0] = xCross
    LinSegResc[:-1,1] = xCross
    
    
    return LinSegResc


def test_piecewise_linear_approximation(data, LinSeg, p=7, axislim=[], plotGraph=True, loc='lower right'):

    N = len(LinSeg)    
    
    minErr, data2 = clean_linreg_data(data, p, keepMinMaxOnly=False)

    minX = min(data2[:,0])
    maxX = max(data2[:,0])
    dX = maxX - minX
    minY = min(data2[:,1])
    maxY = max(data2[:,1])
    dY = maxY - minY

    coveredPoints = np.tile(False,len(data2))

    if plotGraph:
        
        whereFillOdd = np.tile(False,N*3)
        whereFillEven = np.tile(False,N*3)
        whereFillOdd[0:N*3:6] = True
        whereFillOdd[1:N*3:6] = True
        whereFillEven[3:N*3:6] = True
        whereFillEven[4:N*3:6] = True    
        xFill = np.concatenate((LinSeg[:,0:2], np.zeros((N,1))),axis=1).reshape(N*3)
        fig, ax = plt.subplots()
        ax.fill_between(xFill, minY-0.1*dY, maxY+0.1*dY, where=whereFillOdd, facecolor='silver', alpha=0.5, label='segment domain')     
        ax.fill_between(xFill, minY-0.1*dY, maxY+0.1*dY, where=whereFillEven, facecolor='dimgrey', alpha=0.5, label='segment domain')     
    
        ax.scatter(data2[:,0],data2[:,1], color='k', facecolors='none', label='data points')
    
    errVec = np.zeros(N)
    crossVec = np.zeros(N)
    
    for k in range(N):
        lk = LinSeg[k,0]
        uk = LinSeg[k,1]
        segk = (data2[:,0]>=lk) & (data2[:,0]<=uk)
        coveredPoints[segk] = True
        ak = LinSeg[k,2]
        bk = LinSeg[k,3]

        if k>0:
            crossVec[k] = ((ak-LinSeg[k-1,2])*LinSeg[k-1,1] + (bk-LinSeg[k-1,3]))*((ak-LinSeg[k-1,2])*lk + (bk-LinSeg[k-1,3]))
        errVec[k] = max(abs(ak*data2[segk,0]+bk - data2[segk,1]))
        
        if plotGraph:
            if k==0:
                ax.plot(LinSeg[k,0:2],ak*LinSeg[k,0:2]+bk,'k', label='linear segment')
            else:
                ax.plot(LinSeg[k,0:2],ak*LinSeg[k,0:2]+bk,'k')
            if k>0:
                segkBackward = (data2[:,0]>=LinSeg[k-1,1]) & (data2[:,0]<=lk)
                ax.plot(data2[segkBackward,0],ak*data2[segkBackward,0]+bk,'k--')
            if k<N-1:
                segkForward = (data2[:,0]>=uk) & (data2[:,0]<=LinSeg[k+1,0])
                ax.plot(data2[segkForward,0],ak*data2[segkForward,0]+bk,'k--')
            ax.text(np.mean(LinSeg[k,0:2]),minY-0.1*dY,str(k+1),ha='center')
            ax.text(np.mean(LinSeg[k,0:2]),maxY+0.1*dY,"{:.3f}".format(errVec[k]),ha='center',va='top')

    if plotGraph:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        ax.text(minX-0.03*dX,minY-0.1*dY,'Segment:',ha='right')
        ax.text(minX-0.03*dX,maxY+0.1*dY,'Max error:',ha='right',va='top')
        
        if len(axislim)==0:
            ax.set_xlim([minX-0.02*dX, maxX+0.02*dX])
        else:
            ax.set_xlim([axislim[0,0], axislim[0,1]])
            ax.set_ylim([axislim[1,0], axislim[1,1]])
        
        ax.legend(loc=loc, framealpha=1)
    
    print("Max error: "+str(round(max(errVec),7)))
    print("Uncovered points: "+str(sum(~coveredPoints)))
    print("Discontinuity: "+str(sum(crossVec>np.power(10.,-p))))
    
    return errVec, crossVec



def fast_MILP(data, err, solverType=mathopt.SolverType.GUROBI, gap=1e-6, p=7):  
      
    minErr, data2 = clean_linreg_data(data, p)
    
    dataBounds, dataResc = rescale_piecewise_linear_problem(data2, p)
    xMin = dataBounds[0,0]
    xMax = dataBounds[1,0]
    yMin = dataBounds[0,1]
    yMax = dataBounds[1,1]
    dy = yMax - yMin
    
    data = dataResc
    err = err/dy
    
    C, LinSeg = optimal_piecewise_linear_approximation_binary_search2(data, err, crossSeg=False, reverseOrder=False)
    LinSegC = make_piecewise_linear_continuous2(data, err, LinSeg, p=7)
    
    C, LinSegR = optimal_piecewise_linear_approximation_binary_search2(data, err, crossSeg=False, reverseOrder=True)
    LinSegRC = make_piecewise_linear_continuous2(data, err, LinSegR, p=7)
    
    if len(LinSegC) <= len(LinSegRC):
        K = len(LinSegC)
        wsLinSeg = LinSegC
    else:
        K = len(LinSegRC)
        wsLinSeg = LinSegRC
    
    bigMerr, bigMcont, bigMslopeVar = bigM_parameter(data,err,p)
    bigM = max(max(bigMerr),max(bigMcont),bigMslopeVar) + 10
    
    CC, wsLinSeg2 = build_segment_index_vector(data, wsLinSeg, p)
    
    # tightening, WS
    C, c, LinSegMILP, model = optimal_piecewise_linear_approximation_MILP(data, err, K, Slb = 0, crossSeg='inBetween', solverType=solverType, coOpt=True, gap = gap, rescale=False, bigM = bigM, optBigM=True, p=p, ws=True, wsLinSeg=wsLinSeg2, applyBounds=True, lbLinSeg=LinSeg, ubLinSeg=LinSegR)
    LinSegMILPresc = rescale_PWL_solution(LinSegMILP, xMin, xMax, yMin, yMax)

    return LinSegMILPresc


def eval_piecewise_linear_approximation(data, LinSeg, p=7):
    N = len(data)
    yVal = np.zeros(N)
    for n in range(N):
        idxSeg = np.where((LinSeg[:,0]<=data[n,0]) & (LinSeg[:,1]>=data[n,0]))[0][0]
        yVal[n] = LinSeg[idxSeg,2]*data[n,0]+LinSeg[idxSeg,3]
    return yVal


def from_linseg_to_list_points(LinSeg,extend=[]):
    x = np.r_[LinSeg[0,0],LinSeg[:,1]]
    if len(extend)==2:
        x[0] = extend[0]
        x[-1] = extend[1]
    y = np.array(x)
    y[0] = LinSeg[0,2]*x[0] + LinSeg[0,3]
    y[1:] = LinSeg[:,2]*x[1:] + LinSeg[:,3]
    return x, y