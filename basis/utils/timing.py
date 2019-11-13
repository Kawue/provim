# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:48:32 2017

@author: Dr. Ivan Laponogov
"""

import time;

__dttime = time.time();

def tic():
    global __dttime;
    __dttime = time.time();
    
def toc():
    global __dttime;
    t = time.time() - __dttime;
    print('%s seconds'%t)