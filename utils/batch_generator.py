import tensorflow as tf
import keras
from keras import backend as Keras
import numpy as np
import os
from skimage.io import imread
from os.path import normpath as fn

def main():
	visited,d_dict=build_dict()
	basic_i=basicIterator(60,6,visited,d_dict)
    print(basic_i)

def build_dict():
    v_dict={}
    d_dict={}
    for i, dirname in enumerate(os.listdir("../data/train")):
        rel_name="../data/train/"+dirname
        d_dict[i]=dirname
        c_dict={}
        for j, filename in enumerate(os.listdir(rel_name)):
            c_dict[j]=filename
        v_dict[dirname]=c_dict
    return v_dict, d_dict

class basicIterator:
    def __init__(self, batch_size, num_cate, visited, d_dict):
        self.num_cate=num_cate
        self.cate_size=batch_size/self.num_cate
        self.batch_idx=np.random.randint(102,size=self.num_cate)
        self.cate_idx={}
        for k in self.batch_idx:
            self.cate_idx[k]=np.random.randint(len(visited[d_dict[k]]),size=int(self.cate_size))
        self.d_dict=d_dict

    def next(self):
        res_x=[]
        res_y=[]
        self.batch_idx=np.random.randint(102,size=self.num_cate)
        self.cate_idx={}
        for k in self.batch_idx:
            self.cate_idx[k]=np.random.randint(len(visited[d_dict[k]]),size=int(self.cate_size))
        for b_idx in self.batch_idx:
            current_x=[]
            current_y=[]
            for c_idx in self.cate_idx[b_idx]:
                img = np.float32(imread(fn('../data/train/'+ d_dict[b_idx] + '/' + visited[d_dict[b_idx]][c_idx])))/255.
                img = resize(img,(224,224,3),anti_aliasing=True)
                res_x.append(img)
                res_y.append(d_dict[b_idx])
        return np.array(res_x),np.array(res_y)

class triplessIterator:

	def __init__(self, batch_size,visited):
        self.nun_cate=2
        self.cate_size=batch_size/self.num_cate
        self.batch_idx=np.random.randint(102, size=2)
        # self.batch_idx=np.array([idx[0],idx[0],idx[1]])
        self.cate_idx={}
        for k in self.batch_idx:
            self.cate_idx[k]=np.random.randint(len(visited[d_dict[k]]),size=int(self.cate_size))
        self.d_dict=d_dict

	def next(self):
		res_x=[]
        res_y=[]
        for k in self.batch_idx:
            pass
		return np.array(res_x),np.array(res_y)
	
if __name__ == "__main__": main()