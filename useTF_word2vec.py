#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:07:58 2017

@author: hp
"""

import tensorflow as tf
import collections
import math
import numpy as np
import os
import datetime
import random
import zipfile
from matplotlib import pylab
from sklearn.manifold import TSNE


def Log(*context):
    '''
    for output some infomation
    '''
    outputlogo = "---->" + "[" + str(datetime.datetime.now()) + "]"
    string_print = ""
    for c in context:
        string_print += str(c)+"  "
    content = outputlogo +string_print + '\n'
    f = open("log.txt",'a')
    f.write(content)
    f.close()
    print outputlogo,string_print,'\n'
    return True


filename = './text8.zip'
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

