import numpy as np
import os
from skimage.transform import resize
from skimage.io import imread
from os.path import normpath as fn
import os

_cur_dir = os.path.dirname(os.path.realpath(__file__))
TRAIN_DIR = os.path.join(_cur_dir, '../data/train/')
TEST_DIR = os.path.join(_cur_dir, '../data/test/')

def build_dict():
    v_dict={}
    d_dict={}
    count=0
    for i, dirname in enumerate(os.listdir(TRAIN_DIR)):
        rel_name=TRAIN_DIR+dirname
        d_dict[i]=dirname
        c_dict={}
        for j, filename in enumerate(os.listdir(rel_name)):
            c_dict[j]=filename
        v_dict[dirname]=c_dict
    return v_dict, d_dict

def build_dict_test():
    td_dict={}
    count=0
    for filename in os.listdir(TEST_DIR):
        name_cate=filename.rsplit("_",1)[0]
        if name_cate in td_dict:
            td_dict[name_cate].append(filename)
        else:
            td_dict[name_cate]=[filename]
    return td_dict


class BasicTripletIterator:

    def __init__(self, batch_size):
        self.batch_size=batch_size
        self.visited,self.d_dict=build_dict()
        self.num_cate=3
        self.cate_size=self.batch_size/self.num_cate

    def next(self):
        res_x=[]
        res_y=[]
        for i in range(int(self.cate_size)):
            idx=np.random.randint(102, size=2)
            src_idx1=np.random.randint(len(self.d_dict[idx[0]]))
            src_idx2=np.random.randint(len(self.d_dict[idx[0]]))
            tar_idx=np.random.randint(len(self.d_dict[idx[1]]))
            img_src1 = resize(np.float32(imread(fn(TRAIN_DIR+ self.d_dict[idx[0]] + '/' + self.visited[self.d_dict[idx[0]]][src_idx1])))/255.,(224,224,3),anti_aliasing=True)
            img_src2 = resize(np.float32(imread(fn(TRAIN_DIR+ self.d_dict[idx[0]] + '/' + self.visited[self.d_dict[idx[0]]][src_idx2])))/255.,(224,224,3),anti_aliasing=True)
            img_tar=resize(np.float32(imread(fn(TRAIN_DIR+ self.d_dict[idx[1]] + '/' + self.visited[self.d_dict[idx[1]]][tar_idx])))/255.,(224,224,3),anti_aliasing=True)
            res_x.append([img_src1,img_src2,img_tar])
            label1=np.zeros(102)
            label1[idx[0]]=1
            label2=np.zeros(102)
            label2[idx[1]]=1
            res_y.append([label1,label1,label2])
        return np.array(res_x).reshape((self.batch_size,224,224,3)),np.array(res_y).reshape((self.batch_size,102))

class BasicTripletIteratorTest:

    def __init__(self, batch_size):
        self.batch_size=batch_size
        self.td_dict=build_dict_test()
        _,self.d_dict=build_dict()
        self.num_cate=3
        self.cate_size=self.batch_size/self.num_cate

    def next(self):
        res_x=[]
        res_y=[]
        for i in range(int(self.cate_size)):
            idx=np.random.randint(102, size=2)
            src_name=self.d_dict[idx[0]]
            tar_name=self.d_dict[idx[1]]
            self.td_dict[src_name]
            src_idx1=np.random.randint(len(self.td_dict[src_name]))
            src_idx2=np.random.randint(len(self.td_dict[src_name]))
            tar_idx=np.random.randint(len(self.td_dict[tar_name]))
            img_src1 = resize(np.float32(imread(fn(TEST_DIR + '/' + self.td_dict[src_name][src_idx1])))/255.,(224,224,3),anti_aliasing=True)
            img_src2 = resize(np.float32(imread(fn(TEST_DIR + '/' + self.td_dict[src_name][src_idx2])))/255.,(224,224,3),anti_aliasing=True)
            img_tar = resize(np.float32(imread(fn(TEST_DIR + '/' + self.td_dict[tar_name][tar_idx])))/255.,(224,224,3),anti_aliasing=True)
            label1=np.zeros(102)
            label1[idx[0]]=1
            label2=np.zeros(102)
            label2[idx[1]]=1
            res_x.append([img_src1,img_src2,img_tar])
            res_y.append([label1,label1,label2])
        return np.array(res_x).reshape((self.batch_size,224,224,3)),np.array(res_y).reshape((self.batch_size,102))


class ImprovedTripletIterator:

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
                img = np.float32(imread(fn(TRAIN_DIR+ self.d_dict[b_idx] + '/' + self.visited[self.d_dict[b_idx]][c_idx])))/255.
                img = resize(img,(224,224,3),anti_aliasing=True)
                res_x.append(img)
                label=np.zeros(102)
                label[b_idx]=1
                res_y.append(label)
        return np.array(res_x),np.array(res_y)

class ImprovedTripletIteratorTest:

    def __init__(self, batch_size, num_cate):
        _,self.d_dict=build_dict()
        self.td_dict=build_dict_test()
        self.num_cate=num_cate
        self.cate_size=batch_size/self.num_cate
        self.batch_idx=np.random.randint(102,size=self.num_cate)
        self.cate_idx={}
        for k in self.batch_idx:
            self.cate_idx[k]=np.random.randint(len(self.td_dict[self.d_dict[k]]),size=int(self.cate_size))

    def next(self):
        res_x=[]
        res_y=[]
        self.batch_idx=np.random.randint(102,size=self.num_cate)
        self.cate_idx={}
        for k in self.batch_idx:
            self.cate_idx[k]=np.random.randint(len(self.td_dict[self.d_dict[k]]),size=int(self.cate_size))
        for b_idx in self.batch_idx:
            for c_idx in self.cate_idx[b_idx]:
                name=self.td_dict[self.d_dict[b_idx]][c_idx]
                # print(name)
                img = np.float32(imread(fn(TEST_DIR+ '/' + name)))/255.
                img = resize(img,(224,224,3),anti_aliasing=True)
                res_x.append(img)
                # res_y.append(self.d_dict[b_idx])
                label=np.zeros(102)
                label[b_idx]=1
                res_y.append(label)
        return np.array(res_x),np.array(res_y)

# class ImprovedTriplessIterator:

#     def __init__(self, batch_size, num_cate):
#         self.visited,self.d_dict=build_dict()
#         self.num_cate=3
#         self.cate_size=batch_size/self.num_cate
#         self.batch_idx=np.random.randint(102, size=2)
#         self.cate_idx={}
#         for k in self.batch_idx:
#             self.cate_idx[k]=np.random.randint(len(self.visited[self.d_dict[k]]),size=int(self.cate_size))

#     def next(self):
#         res_x=[]
#         res_y=[]
#         for k in self.batch_idx:
#             self.cate_idx[k]=np.random.randint(len(self.visited[self.d_dict[k]]),size=int(self.cate_size))
#         count=0
#         for b_idx in self.batch_idx:
#             for c_idx in self.cate_idx[b_idx]:
#                 img = np.float32(imread(fn(TRAIN_DIR+ self.d_dict[b_idx] + '/' + self.visited[self.d_dict[b_idx]][c_idx])))/255.
#                 img = resize(img,(224,224,3),anti_aliasing=True)
#                 res_x.append(img)
#                 res_y.append(self.d_dict[b_idx])  
#             if(count==0):
#                 res_x+=res_x
#                 res_y+=res_y
#                 count+=1
#         return np.array(res_x),np.array(res_y)

# class ImprovedTriplessIteratorTest:

#     def __init__(self, batch_size):
#         self.visited,self.d_dict=build_dict()
#         self.num_cate=3
#         self.cate_size=batch_size/self.num_cate
#         self.batch_idx=np.random.randint(102, size=2)
#         self.cate_idx={}
#         for k in self.batch_idx:
#             self.cate_idx[k]=np.random.randint(len(self.visited[self.d_dict[k]]),size=int(self.cate_size))

#     def next(self):
#         res_x=[]
#         res_y=[]
#         for k in self.batch_idx:
#             self.cate_idx[k]=np.random.randint(len(self.visited[self.d_dict[k]]),size=int(self.cate_size))
#         count=0
#         for b_idx in self.batch_idx:
#             for c_idx in self.cate_idx[b_idx]:
#                 img = np.float32(imread(fn(TRAIN_DIR+ self.d_dict[b_idx] + '/' + self.visited[self.d_dict[b_idx]][c_idx])))/255.
#                 img = resize(img,(224,224,3),anti_aliasing=True)
#                 res_x.append(img)
#                 res_y.append(self.d_dict[b_idx])  
#             if(count==0):
#                 res_x+=res_x
#                 res_y+=res_y
#                 count+=1
#         return np.array(res_x),np.array(res_y)