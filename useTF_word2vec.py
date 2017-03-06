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

words = read_data(filename)
Log("Data size " , len(words))

vocabulary_size = 50000

def build_dataset(words):
    count = [['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size -1))
    dictionary = dict()
    for word , _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values() , dictionary.keys()))
    return data , count , dictionary , reverse_dictionary

data , count , dictionary , reverse_dictionary = build_dataset(words)
del words
Log('Most common words (+UNK)', count[:5])
Log('Sample data', data[:10])

data_index = 0

def generate_batch(batch_size , num_skips , skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2*skip_window
    batch = np.ndarray(shape=(batch_size) , dtype = np.int32)
    labels = np.ndarray(shape = (batch_size , 1),dtype = np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen = span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index +1) % len(data)
        
    for i in range(batch_size//num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0,span-1)
            targets_to_avoid.append(target)
            batch[i*num_skips + j] = buffer[skip_window]
            labels[i*num_skips + j,0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch , labels

Log('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    Log('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    Log('    batch:', [reverse_dictionary[bi] for bi in batch])
    Log('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
'''
data: ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first']

with num_skips = 2 and skip_window = 1:
    batch: ['originated', 'originated', 'as', 'as', 'a', 'a', 'term', 'term']
    labels: ['as', 'anarchism', 'a', 'originated', 'term', 'as', 'a', 'of']

with num_skips = 4 and skip_window = 2:
    batch: ['as', 'as', 'as', 'as', 'a', 'a', 'a', 'a']
    labels: ['anarchism', 'originated', 'term', 'a', 'as', 'of', 'originated', 'term']
'''

batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2
valid_size = 16
valid_window = 1100
valid_examples = np.array(random.sample(range(valid_window),valid_size))
num_sampled = 64

graph = tf.Graph()

with graph.as_default():
    train_dataset = tf.placeholder(tf.int32 , shape = [batch_size])
    train_labels = tf.placeholder(tf.int32 , shape = [batch_size,1])
    valid_dataset = tf.constant(valid_examples , dtype = tf.int32)
    
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size , embedding_size],-1.0,1.0))
    softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size , embedding_size],stdddev = 1.0/math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
    embed = tf.nn.embedding_lookup(embeddings , train_dataset)
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights , softmax_biases , embed,train_labels , num_sampled, vocabulary_size))
    
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
    
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims= True))
    normalized_embeddings = embeddings / norm
    valid_embeddings  = tf.nn.embedding_lookup(normalized_embeddings , valid_dataset)
    similarity = tf.matmul(valid_embeddings , tf.transpose(normalized_embeddings))