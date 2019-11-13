# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:12:50 2017

@author: kirillveselkov
"""

import numpy as np
from scipy.spatial import KDTree


from basis.utils.typechecker import is_number


def conv2Dto3DX(X,xy):
    xy2D = conv_xy2grid(xy[:,0],xy[:,1]);
    nRows,nCols = xy2D.shape
    nObs,nmz = X.shape
                        
    # allocate spectral profiles
    X3d   = np.zeros([nmz, nCols, nRows])
    for i in np.arange(0, (nRows)):
        for j in np.arange(0, (nCols)):
            if not np.isnan(xy2D[i,j]):
                X3d[:,j,i] = X[int(xy2D[i,j]), :]
    return X3d,xy2D

def conv_xy2grid(x, y, tol=0.001):
    """
    Generates 2D grid layout from x and y coordinates (fast approach)        
    """
    #  round off coordinate values 
    x = np.round((x/tol));
    y = np.round((y/tol));
    # calculate minimum and maximum x & y coordinate values
    minx = np.min(x);
    maxx = np.max(x);
    miny = np.min(y);
    maxy = np.max(y);
    # step for x and y coordinates
    stepx  = np.max(np.diff(np.unique(x)));
    ycoord = np.unique(y);
    stepy  = np.max(np.diff(ycoord));
    gridy  = np.arange(miny, (maxy)+(stepy), stepy);
    gridx  = np.arange(minx, (maxx)+(stepx), stepx);
    nRows  = len(gridy);
    nCols  = len(gridx);
        
    xy2D    = np.empty([nRows, nCols],);
    xy2D[:] = np.nan; 
    # loop through all x and y coordinates
    iCount = 0;
    for iy in gridy:
        iCount = iCount+1
        if ycoord[iCount-1] == iy:
            try:       
                iScans = np.nonzero(y == iy);
                indcs  = np.where(np.in1d(gridx, x[iScans]))
                xy2D[int(nRows-iCount),indcs] = iScans
            except:
                print(str(iCount));
                #%self.resx = np.dot(stepx, tol)
                #%self.resy = np.dot(stepy, tol)
    return xy2D


def conv_xy2grid_nn(x, y, tol=[]):
    """
    Generates 2D grid layout from x and y coordinates using neareast neighbour approach (slow appraoch)        
    """      
    if is_number(tol):
        x = np.round((x/tol));
        y = np.round((y/tol));
    #% calculate minimum and maximum x & y coordinate values
    minx = np.min(x);
    maxx = np.max(x);
    miny = np.min(y);
    maxy = np.max(y);

    #% step for x and y coordinates
    xcoord = np.unique(x);
    stepx  = np.median(np.diff(xcoord));
    ycoord = np.unique(y);
    stepy  = np.median(np.diff(ycoord));
    gridy  = np.arange(miny, (maxy)+(stepy), stepy);
    gridx  = np.arange(minx, (maxx)+(stepx), stepx);
    nRows  = len(gridy);
    nCols  = len(gridx);
            
    xy2D    = np.empty([nRows, nCols],);
    xy2D[:] = np.nan;
    tree    = KDTree(gridy[:,None]);      
    yindcs  = np.unique(tree.query(ycoord[:,None],1)[1])
    for iy in yindcs:           
        try:       
            iScans = np.nonzero(y == ycoord[iy]);
            tree   = KDTree(gridx[:,None]);
            xindcs = np.unique(tree.query(x[iScans,None],1)[1]); 
            xy2D[nRows-iy-1,xindcs] = iScans;
        except:
            print(str(ycoord[iy]));
                #%self.resx = np.dot(stepx, tol)
                #%self.resy = np.dot(stepy, tol)
    xynan    = np.isnan(xy2D)        
    xynansum = xynan.sum(axis=1)
    rowindcs = xynansum!=nCols;
    xy2D     = xy2D[rowindcs,:];
    gridy    = gridy[rowindcs]
    xynan    = np.isnan(xy2D)
    xynansum = xynan.sum(axis=0)
    colindcs = xynansum!=nRows;
    xy2D     = xy2D[:,colindcs];
    gridx    = gridx[colindcs]
    
    return xy2D