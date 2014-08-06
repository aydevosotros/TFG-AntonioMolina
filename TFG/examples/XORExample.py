'''
Created on Jun 14, 2014

@author: antonio
'''
from basicRBFNN.RBFNN import RBFNN
from scipy import *
from matplotlib import pyplot as plt

if __name__ == '__main__':
    rbfnn = RBFNN(1, 10, 1)
    # number of samples of each cluster
    K = 100;
    # offset of clusters
    q = 0.6;
    # define 2 groups of input data
    A = rand(100)
#     print(A)
#     A.append([rand(1,K)+q,rand(1,K)-q])
    B = rand(100)
#     B.append([rand(1,K)+q,rand(1,K)-q])
    
    plt.plotDataSet(A,B)
    plt.show()
    #plt.plotDataSet((A(1,:),A(2,:),'k+',B(1,:),B(2,:),'b*'))
#     grid on
#     hold on