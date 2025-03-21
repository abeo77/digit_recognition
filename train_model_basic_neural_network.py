import numpy as np
from numpy import random
import cv2
import csv
import random as rd
from keras.datasets import mnist
import tkinter as tk
from PIL import Image, ImageDraw
soft = lambda x:(1/(1+np.exp(-x)))
dsoft= lambda x:soft(x)*(1-soft(x))
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    out= exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return out
def setup_b(a=[784,150,100,10]):
    b=[]
    for i in range(len(a)-1):
        b.append(random.random((a[i+1],1))/a[i+1])
    return b
def setup_weight(a=[784,150,100,10]):
    b=[]
    for i in range(len(a)-1):
        b.append(random.random((a[i+1],a[i]))/a[i])
    return b
def coputing(a,w,b):
    c=[a]
    for index,i in enumerate(w):
        k=b[index]
        sd=i@a
        a = soft(sd+k)
        c.append(a)
    return c
def z(a,w,b):
    c = []
    for index, i in enumerate(w):
        c.append(i @ a + b[index])
        a = soft(i @ a + b[index])
    return c
def jacobian_w(a0,y):
    a = coputing(a0, w, b)
    j_w=[]
    j_b = []
    z1 = z(a0, w, b)
    j_a = 2 * (a[-1]-y)
    for i in range(1, len(w) + 1):
        dz=j_a*dsoft(z1[len(b)-i])
        j_b.append(dz)
        dw = dz@a[len(a)-i-1].T
        j_w.append(dw)
        k = w[len(w) - i]
        j_a = (dz.T @ k).T
    return j_w,j_b
def jacobian_b(a0,y):
    a=coputing(a0,w,b)
    j_b=[]
    z1=z(a0,w,b)
    j_a=2*(a[-1]-y)
    for i in range(1,len(b)+1):
        dz=j_a*dsoft(z1[len(b)-i])
        j_b.append(dz)
        k=w[len(w)-i]
        j_a=(dz.T@k).T
    return j_b
def te():
        pas = 0
        for index,i in enumerate(test_features):
            a = coputing(i.reshape(-1,1)/255,w,b)[-1]
            ke = [0,1,2,3,4,5,6,7,8,9]
            ke.sort(key=lambda x: a[x],reverse=True)
            if ke[0] == test_targets[index]:
                pas+=1
        return pas/len(test_features)
def re(a0,y):
    jw,jb=jacobian_w(a0,y)
    global w
    global b
    for i in range(len(b)):
        b[i]=b[i] - 0.05 * jb[len(b)-i-1]
    for i in range(len(w)):
        w[i] = w[i] - 0.05 * jw[len(b) - i - 1]
def data():
        result = []
        for img,y in zip(train_features, train_targets):
            y1 = np.zeros((10, 1))
            y1[y] = 1
            b = img.reshape(-1,1)/255
            result.append((y1,b))
        return result
if __name__ == '__main__':
    (train_features, train_targets), (test_features, test_targets) = mnist.load_data()
    res=data()
    w=setup_weight()
    b=setup_b()
    test0 = te()
    print(test0)
    count = 0
    while True:
        random.shuffle(res)
        for index,i in enumerate(res):
            re(i[1], i[0])
        test1 = te()
        print('pack: ',count)
        print('the accuracy: ',test1)
        if count == 10:
            np.savez('arrays_of_w.npz', *w)
            np.savez('arrays_of_b.npz', *b)
            break
        count +=1
