import pandas as pd
import numpy as np
import os,sys,math

def readlines(path):
    ff=open(path,"r")
    ff=[i.strip() for i in ff.readlines()]
    results=[]
    for i in ff:
        if "test epoch:" in i:
            results.append(i)
    return results

def get_acc_list(results):
    results=[i.split('\t')[-1] for i in results]
    results=[float(i.split(':')[-1]) for i in results]
    return results

def get_last_five_avg(path1,path2,path3=None):
    A=get_acc_list(readlines(path1))
    B=get_acc_list(readlines(path2))
    if path3!=None:
        C=get_acc_list(readlines(path3))[-5:]
    else:
        C=[]
    A=A[-5:]
    B=B[-5:]
    D=A+B+C
    D=np.array(D)*100
    return round(D.mean(),3),round(D.std(),3)
mean,std=get_last_five_avg("/home/sst/product/plotlydict/compare/ResNet32x4_ResNet8x4_C100_DACRD/tarch_resnet32x4_crd_arch_resnet8x4_crd_dataset_cifar100__spkd_3.txt",
                  "/home/sst/product/plotlydict/compare/ResNet32x4_ResNet8x4_C100_DACRD/tarch_resnet32x4_crd_arch_resnet8x4_crd_dataset_cifar100__spkd_4.txt")
print(mean,std)