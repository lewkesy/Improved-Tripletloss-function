import sys
sys.path.append("../utils")
from alexnet import alexnet_model
# from alexnet_multi_loss import alexnet_model
from batch_generator import ImprovedTripletIterator, ImprovedTripletIteratorTest, BasicTriplessIterator, BasicTriplessIteratorTest
from keras.applications.resnet50 import preprocess_input
import os
import numpy as np
from keras.utils.np_utils import to_categorical
from IPython import embed

train_dir = '../data/train'
classes = os.listdir(train_dir)
num_classes = len(classes)
batch_size = 30
num_train_samples = 3700
num_test_samples = 1600

'''
original compile code for alexnet

'''

def get_gen(gen):
    while True:
        images, labels = gen.next()
        images = preprocess_input(images)
        labels = [classes.index(x) for x in labels]
        labels = to_categorical(labels, num_classes=num_classes)
        yield images, labels

train_gen = get_gen(BasicTriplessIterator(batch_size))
test_gen = get_gen(BasicTriplessIteratorTest(batch_size))
