import numpy as np
import os
from skimage.transform import resize
from skimage.io import imread
from os.path import normpath as fn
import os

_cur_dir = os.path.dirname(os.path.realpath(__file__))
TRAIN_DIR = os.path.join(_cur_dir, '../caltech/caltech_train/')

def build_dict():
    v_dict={}
    d_dict={}
    for i, dirname in enumerate(os.listdir(TRAIN_DIR)):
        rel_name=TRAIN_DIR+dirname
        d_dict[i]=dirname
        c_dict={}
        for j, filename in enumerate(os.listdir(rel_name)):
            c_dict[j]=filename
        v_dict[dirname]=c_dict
    return v_dict, d_dict

class basicIterator:

    def __init__(self, batch_size, num_cate):
        self.visited,self.d_dict=build_dict()
        
        self.num_cate=num_cate
        self.cate_size=batch_size/self.num_cate
        self.batch_idx=np.random.randint(102,size=self.num_cate)
        self.cate_idx={}
        for k in self.batch_idx:
            self.cate_idx[k]=np.random.randint(len(self.visited[self.d_dict[k]]),size=int(self.cate_size))

    def next(self):
        res_x=[]
        res_y=[]
        self.batch_idx=np.random.randint(102,size=self.num_cate)
        self.cate_idx={}
        for k in self.batch_idx:
            self.cate_idx[k]=np.random.randint(len(self.visited[self.d_dict[k]]),size=int(self.cate_size))
        for b_idx in self.batch_idx:
            for c_idx in self.cate_idx[b_idx]:
                img = np.float32(imread(fn(TRAIN_DIR+ d_dict[b_idx] + '/' + visited[d_dict[b_idx]][c_idx])))/255.
                img = resize(img,(224,224,3),anti_aliasing=True)
                res_x.append(img)
                res_y.append(self.d_dict[b_idx])
        return np.array(res_x),np.array(res_y)

class triplessIterator:

    def __init__(self, batch_size):
        self.visited,self.d_dict=build_dict()
        self.num_cate=3
        self.cate_size=batch_size/self.num_cate
        print(self.cate_size)
        self.batch_idx=np.random.randint(102, size=2)
        self.cate_idx={}
        for k in self.batch_idx:
            self.cate_idx[k]=np.random.randint(len(self.visited[self.d_dict[k]]),size=int(self.cate_size))


    def next(self):
        res_x=[]
        res_y=[]
        for k in self.batch_idx:
            self.cate_idx[k]=np.random.randint(len(self.visited[self.d_dict[k]]),size=int(self.cate_size))
        count=0
        for b_idx in self.batch_idx:
            
            for c_idx in self.cate_idx[b_idx]:
                img = np.float32(imread(fn(TRAIN_DIR+ d_dict[b_idx] + '/' + visited[d_dict[b_idx]][c_idx])))/255.
                img = resize(img,(224,224,3),anti_aliasing=True)
                res_x.append(img)
                res_y.append(self.d_dict[b_idx])
                
            if(count==0):
                res_x+=res_x
                res_y+=res_y
                count+=1
        return np.array(res_x),np.array(res_y)
