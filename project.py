# -*- coding: utf-8 -*-
"""
Created on Mon Jul 02 22:17:46 2018

@author: cspabla
"""
import os, struct
import matplotlib as plt
from array import array as pyarray
import pandas as pd
from openpyxl import load_workbook
from pylab import *
import numpy as np

def load_NMNIST(dataset, digits):
    path=r"."
    
    if dataset == "training":
        fname_img = os.path.join(path, "NMNIST_Train_Features.dat")
        #fname_lbl = os.path.join(path, "train-labels-idx1-ubyte\train-labels.idx1-ubyte")
        fname_lbl = os.path.join(path, "NMNIST_Train_Labels.dat")
    elif dataset == "testing":
        fname_img = os.path.join(path, "NMNIST_Test_Features.dat")
        fname_lbl = os.path.join(path, "NMNIST_Test_Labels.dat")
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
    print(str(fname_lbl))
    
    flbl = open(fname_lbl, 'rb')
    lbl = np.fromfile(flbl, dtype=np.uint8)
    flbl.close()

    #print(len(lbl))
    fimg = open(fname_img, 'rb')
    img = np.fromfile(fimg, dtype=np.uint8)
    fimg.close()

    size=len(lbl)
    #print(size)

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)
    #print(N)
    rows=28;cols=28;

    images = zeros((N, rows*cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    count=0
    set_printoptions(linewidth=115)
    for i in range(len(ind)):
        count = count+1
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])
        labels[i] = lbl[ind[i]]
    return images, labels


def vectortoimg(v,show=True):
    plt.imshow(v.reshape(28, 28),interpolation='None', cmap='gray')
    plt.axis('off')
    if show:
        plt.show()

def display_NMNIST(images,labels):
    # converting from NX28X28 array into NX784 array
    flatimages = list()
    for i in images:
        flatimages.append(i.ravel())
    X = asarray(flatimages)
    T=labels
    vectortoimg(X[1])
    
    print("Checking multiple training vectors by plotting images.\nBe patient:")
    plt.close('all')
    fig = plt.figure()
    nrows=8
    ncols=8
    for row in range(nrows):
        for col in range(ncols):
            plt.subplot(nrows, ncols, row*ncols+col + 1)
            vectortoimg(X[np.random.randint(len(T))],show=False)
    plt.show()
    
def load_MNIST(dataset, digits):
    path=r".\MNIST_Data"
    
    if dataset == "training":
        fname_img = os.path.join(path, "train-images.idx3-ubyte")
        fname_lbl = os.path.join(path, "train-labels.idx1-ubyte")
    elif dataset == "testing":
        fname_img = os.path.join(path, "t10k-images.idx3-ubyte")
        fname_lbl = os.path.join(path, "t10k-labels.idx1-ubyte")
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
    #print(str(fname_lbl))
    
    flbl = open(fname_lbl, 'rb')
    lbl = np.fromfile(flbl, dtype=np.uint8)
    flbl.close()

    #print(len(lbl))
    fimg = open(fname_img, 'rb')
    img = np.fromfile(fimg, dtype=np.uint8)
    fimg.close()

    #first 8 bytes of lbl & 16bytes of img database are for formatting, igonre them from data array
    lbl = lbl[8:]
    img = img[16:]
    
    size=len(lbl)
    
    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)
    #print(N)
    rows=28;cols=28;

    images = zeros((N, rows*cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    count=0
    set_printoptions(linewidth=115)
    for i in range(len(ind)):
    #for i in range(1):
        #temp = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])
        count = count+1
        #print(count, temp.shape)
        #images[i] = np.reshape(temp,(rows, cols))
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])
        #print("linear \n" + str(matrix(temp)))
        #print("reshaped \n" +str(images[i]))
        labels[i] = lbl[ind[i]]
    return images, labels

def perform_linear_classification(X,T,TX,TT):

    print("Calculating LC...")
    #number of classes 0-9
    num_classes = 10
    #print(T, TT)
    #print(X.shape)
    
    #create Xa (add 1 at start)
    Xa =  np.insert(X,0,1,axis=1)
    #print(Xa.shape)
    
    #Xa psudeo inverse
    Xai = np.linalg.pinv(Xa)
    #print(Xai.shape)
    
    # 10 class classifier labels
    #create 10 column array & initalize it to -1
    num_rows = Xa.shape[0]
    num_col = num_classes
    L = np.full((num_rows, num_col), -1, dtype=int)
    
    # covert "digit" to Kesler’s construction
    for i in range(num_rows):
        for j in range(num_col):
            if T[i] == j:
                L[i,j] = 1
    #print(L)
    
    # Generate W - 10 class Classifier
    W =  np.matmul(Xai, L)
    #print(W.shape)
    #writeExcelData(W,excelfile,'Classifiers',5,5)
    
    ##############    Application of classifiers ##############
    
    #create Xa (add 1 at start)
    XTDa =  np.insert(TX,0,1,axis=1)
    #print(XTDa.shape)
    
    # apply W to get 10 class label - Tmc -MultiClass estimated
    Tmc = np.matmul(XTDa,W)
    #Convert above float values to type 0-9 (max value of each row represnts the type)
    # therefore, find the indices which has highest value and assign it as type
    Tmc = np.argmax(Tmc,axis=1)
    #print(Tmc)
    #writeExcelData(Tmc,excelfile,'To be classified',5,17)
    
    ##############   classifier performance ##############
    # confusion Matrix
    cm = np.full((num_classes, num_classes), 0, dtype=int)
    
    print("Confusion Matrix")
    #build confusion materix
    for i in range(Tmc.shape[0]):
        cm[TT[i],Tmc[i]] += 1
    
    print(cm)
    #writeExcelData(cm,excelfile,'Performance',19,3)
    
    ppv_max = 0
    ppv_min = 1
    ppv_max_digit = ppv_min_digit = -1 
    #calcuate PPV - iterate over columns
    for j in range(cm.shape[1]):
        ppv = float(cm[j,j])/np.sum(cm[:,j])
        if ppv > ppv_max:
            ppv_max = ppv
            ppv_max_digit = j
        elif ppv < ppv_min:
            ppv_min = ppv
            ppv_min_digit = j
                
    print("ppv_max " + str(ppv_max) + " ppv_max_digit " + str(ppv_max_digit))
    print("ppv_min " +str(ppv_min)+" ppv_min_digit " + str(ppv_min_digit))

# main
digits=[0,1,2,3,4,5,6,7,8,9]

# load NMNIST data - Build X
X, T = load_NMNIST('training', digits)
#print("Checking shape of matrix:", X.shape)
#print("Checking min/max values:",(np.amin(X),np.amax(X)))
#print("Checking unique labels in T:",list(np.unique(T)))

#verify data loaded correctly
display_NMNIST(X,T)

# load Test NMNIST data - Build TX & TT
TX, TT = load_NMNIST('testing', digits)
#print("Checking shape of matrix:", TX.shape)

#verify data loaded correctly
#display_NMNIST(TX,TT)

print(" \n ##### Linear Classifier - Noisy NMNIST Data ##### ")
perform_linear_classification(X,T,TX,TT)

# =============================================================================
# print(" \n ##### Linear Classifier - MNIST Data ##### ")
# # load MNIST data - Build X
# X, T = load_MNIST('training', digits)
# #print("Checking shape of matrix:", X.shape)
# 
# #verify data loaded correctly
# #display_NMNIST(X,T)
# 
# # load Test NMNIST data - Build TX & TT
# TX, TT = load_MNIST('testing', digits)
# #print("Checking shape of matrix:", TX.shape)
# 
# #verify data loaded correctly
# #display_NMNIST(TX,TT)
# 
# perform_linear_classification(X,T,TX,TT)
# =============================================================================
