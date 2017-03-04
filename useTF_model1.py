#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 13:18:01 2017

@author: hp
"""

#from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import datetime
import PIL
import tensorflow as tf
import tensorflow

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

image_size = 28
pixel_depth = 255.0

def load_letter(folder , min_num_images):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files) , image_size ,image_size) ,dtype = np.float32)
    Log("Data folder is " , folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder , image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) - pixel_depth/2) / pixel_depth
            print image_data
            if image_data.shape != (image_size , image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images += 1
        except IOError as e:
            Log("Could not read" , image_file,":",e, '- it\'s ok, skipping.')
    dataset = dataset[0:num_images,:,:]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    Log("Full dataset tensor",dataset.shape)
    Log("Mean : " , np.mean(dataset))
    Log("Standard deviation" , np.std(dataset))
    return dataset

def maybe_pickle(data_folders , min_num_images_per_class , force = False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            Log('%s already present - Skipping pickling.' % set_filename)
        else:
            Log('Pickling %s.' % set_filename)
            dataset = load_letter(folder , min_num_images_per_class)
            try:
                with open(set_filename , 'wb') as f:
                    pickle.dump(dataset , f , pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                Log('Unable to save data to', set_filename, ':', e)
                
    return dataset_names

train_folders =['notMNIST_large/A', 'notMNIST_large/B', 'notMNIST_large/C', 'notMNIST_large/D', 'notMNIST_large/E', 'notMNIST_large/F', 'notMNIST_large/G', 'notMNIST_large/H', 'notMNIST_large/I', 'notMNIST_large/J']
test_folders = ['notMNIST_small/A', 'notMNIST_small/B', 'notMNIST_small/C', 'notMNIST_small/D', 'notMNIST_small/E', 'notMNIST_small/F', 'notMNIST_small/G', 'notMNIST_small/H', 'notMNIST_small/I', 'notMNIST_small/J']
train_datasets = maybe_pickle(train_folders , 45000)
test_datasets = maybe_pickle(test_folders , 1800)

def make_arrays(nb_rows , img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows , img_size , img_size) , dtype = np.float32)
        labels = np.ndarray(nb_rows , dtype = np.int32)
    else:
        dataset , labels = None , None
    return dataset , labels

train_size = 200000
valid_size = 10000
test_size = 10000


def merge_datasets(pickle_files , train_size , valid_size = 0):
    num_classes = len(pickle_files)
    valid_dataset , valid_labels = make_arrays(valid_size , image_size)
    train_dataset , train_labels = make_arrays(train_size , image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes
    
    start_v,start_t = 0 , 0
    end_v , end_t = vsize_per_class ,tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    
    for label , pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file , 'rb') as f:
                letter_set = pickle.load(f)
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class , : , :]
                    valid_dataset[start_v:end_v,:,:] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class
                    
                train_letter = letter_set[vsize_per_class:end_l,:,:]
                train_dataset[start_t:end_t,:,:] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            Log('Unable to process data from', pickle_file, ':', e)
            raise
            
    return valid_dataset, valid_labels, train_dataset, train_labels
    


valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)

_,_,test_dataset, test_labels = merge_datasets(test_datasets, test_size)

Log('Training:', train_dataset.shape, train_labels.shape)
Log('Validation:', valid_dataset.shape, valid_labels.shape)
Log('Testing:', test_dataset.shape, test_labels.shape)

def randomize(dataset,labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset , shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

def showSampleImage(folders):
    folder_num = np.random.randint(1,len(folders))
    files = os.listdir(train_folders[folder_num])
    file_name = files[np.random.randint(1,len(files))]
    file_path = './'+train_folders[folder_num]+'/'+file_name
    Log("Sample Image is " , file_path)
    im = PIL.Image.open(file_path)
    im.show()
    return True
showSampleImage(test_folders)

def showSampleFromData(pickleFile):
    randomPickle = train_datasets[np.random.randint(0,len(train_datasets))]
    data = pickle.load(open(randomPickle,'rb'))
    randomSampleInt = np.random.randint(0,data.shape[0])
    sample = data[randomSampleInt,:,:]
    plt.imshow(sample, cmap="gray")
    plt.show()
    
showSampleFromData('./notMNIST_small/A.pickle')

import pygal
def examBalance(dataset , title):
    dataLen = []
    for f in dataset:
        with open(f ,'rb') as openedF:
            data = pickle.load(openedF)
            dataLen.append(data.shape[0])
            
    chart = pygal.Line(height=350)
    chart.title = title
    chart.x_labels = train_datasets
    chart.add('data Length' , dataLen)
    chart.render_in_browser()
    
#examBalance(train_dataset , 'train_dataset')
#examBalance(test_dataset , 'test_dataset')
#examBalance(valid_dataset , 'valid_dataset')
useLogisticRegression = False
if useLogisticRegression:
    #use LogisticRegression
    train_dataset_R = train_dataset.reshape(200000,784)
    test_dataset_R = test_dataset.reshape(10000,784)
    reg = LogisticRegression()
    reg.fit(train_dataset_R, train_labels)
    #Returns the mean accuracy on the given test data and labels.
    testScore = reg.score(test_dataset_R,test_labels) #shoud around 0.9
    print(testScore)
    Log("LogisticRegression Model test score is" , testScore)

#use TensorFlow
num_labels = 10
def reformat(dataset , labels):
    dataset = dataset.reshape((-1 , image_size*image_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset,labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
Log('Training set', train_dataset.shape, train_labels.shape)
Log('Validation set', valid_dataset.shape, valid_labels.shape)
Log('Test set', test_dataset.shape, test_labels.shape)

train_subset = 10000
Log("a normal and simple deep learning model")
graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.constant(train_dataset[:train_subset,:])
    tf_train_labels = tf.constant(train_labels[:train_subset])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    weights = tf.Variable(tf.truncated_normal([image_size*image_size,num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))
    
    logits = tf.matmul(tf_train_dataset,weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = tf_train_labels))
    
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset,weights)+biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset,weights)+biases)
    
num_steps = 801
def accuracy(predictions , labels):
    return (100*np.sum(np.argmax(predictions,1)==np.argmax(labels,1))/predictions.shape[0])
with tf.Session(graph = graph) as session:
    tf.initialize_all_variables().run()
    Log("Initiallized")
    for step in range(num_steps):
        _,l,predictions = session.run([optimizer , loss , train_prediction])
        if step%100 == 0:
            Log('Loss at step %d: %f' % (step, l))
            Log('Training accuracy: %.1f%%' % accuracy(predictions , train_labels[:train_subset,:]))
            Log('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(),valid_labels))
    Log('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
           
batch_size = 128


Log("Use deep learning model with batch")
graph2 = tf.Graph()
with graph2.as_default():
    tf2_train_dataset = tf.placeholder(tf.float32 , shape=(batch_size , image_size*image_size))
    tf2_train_labels = tf.placeholder(tf.float32 , shape = (batch_size , num_labels))
    tf2_valid_dataset = tf.constant(valid_dataset)
    tf2_test_dataset = tf.constant(test_dataset)
    
    weights = tf.Variable(tf.truncated_normal([image_size*image_size,num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))
    
    logits = tf.matmul(tf2_train_dataset , weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits , labels = tf2_train_labels))
    
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf2_valid_dataset,weights)+biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf2_test_dataset,weights)+biases)
    
num_steps = 3001
with tf.Session(graph = graph2) as session:
    tf.initialize_all_variables().run()
    Log('Initiallzed')
    for step in range(num_steps):
        offset = (step*batch_size)%(train_labels.shape[0]-batch_size)
        batch_data = train_dataset[offset:(offset+batch_size),:]
        batch_labels = train_labels[offset:(offset+batch_size),:]
        
        feed_dict = {tf2_train_dataset:batch_data,
                     tf2_train_labels:batch_labels}
        _,l,predictions = session.run([optimizer,loss , train_prediction],feed_dict = feed_dict)
        if step%500 == 0:
            Log("Minibatch loss at step %d: %f" % (step, l))
            Log("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            Log("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    Log("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
    

Log("Use deep learning model with RELU")
graph3 = tf.Graph()
with graph3.as_default():
    tf3_train_dataset = tf.placeholder(tf.float32,shape=(batch_size , image_size*image_size))
    tf3_train_labels = tf.placeholder(tf.float32 , shape=(batch_size , num_labels))
    tf3_valid_dataset = tf.constant(valid_dataset)
    tf3_test_dataset = tf.constant(test_dataset)
    
    weights = tf.Variable(tf.truncated_normal([image_size*image_size,num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))
    
    logits = tf.nn.relu(tf.matmul(tf3_train_dataset , weights) + biases)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits , labels = tf3_train_labels))
    
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.nn.relu(tf.matmul(tf3_valid_dataset,weights)+biases))
    test_prediction = tf.nn.softmax(tf.nn.relu(tf.matmul(tf3_test_dataset,weights)+biases))
    
num_steps = 3001
#merged_summary_op = tf.merge_all_summaries()

with tf.Session(graph = graph3) as session:
    tf.initialize_all_variables().run()
    #summary_writer = tf.train.SummaryWriter('./mnist_logs', session.graph)
    Log('Initiallzed')
    for step in range(num_steps):
        offset = (step*batch_size)%(train_labels.shape[0]-batch_size)
        batch_data = train_dataset[offset:(offset+batch_size),:]
        batch_labels = train_labels[offset:(offset+batch_size),:]
        
        feed_dict = {tf3_train_dataset:batch_data,
                     tf3_train_labels:batch_labels}
        _,l,predictions = session.run([optimizer,loss , train_prediction],feed_dict = feed_dict)
        if step%500 == 0:
            Log("Minibatch loss at step %d: %f" % (step, l))
            Log("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            Log("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
            #summary_str = session.run(merged_summary_op)
            #summary_writer.add_summary(summary_str, step)
    Log("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))            

    
Log("Deep learning Model with regularization")

    
    
    

           