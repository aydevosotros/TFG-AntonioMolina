'''
Created on Jun 14, 2014
Based of a RBFNN for python written by Thomas Rueckstiess. 
http://www.rueckstiess.net/research/snippets/show/72d2363e
Added by Antonio Molina:
    - K-means clustering

@author: antonio
'''

from scipy import *
import numpy as np
from scipy.linalg import norm, pinv
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.vq import kmeans
from matplotlib import pyplot as plt
 
class RBFNN:
     
    def __init__(self, indim, numCenters, outdim, trainingCentroidsMethod="random", trainingWeightsMethod="pseudoinverse"):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in xrange(numCenters)]
        self.beta = 8
        self.training=trainingCentroidsMethod
        self.W = random.random((self.numCenters, self.outdim))
         
    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c-d)**2)
     
    def _calcAct(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G
     
    def train(self, X, Y):
        """ X: matrix of dimensions n x indim
            y: column vector of dimension n x 1 """
        if self.training == "random":
            print 'Training with randomly chosen centroids'
            rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
            self.centers = [X[i,:] for i in rnd_idx]
        elif self.training == "knn":
            print 'Training with centroids from k-means algorithm'
            self.centers = kmeans(X, self.numCenters)[0]
        elif self.training == "meta":
            #TODO: Implementar metaplasticidad
            pass
        else:
            print "You must set the training method"
            return
#         print "center", self.centers
        # calculate activations of RBFs
        G = self._calcAct(X)
#         print G
         
        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)
         
    def test(self, X):
        """ X: matrix of dimensions n x indim """
         
        G = self._calcAct(X)
        print "G=", G
        Y = dot(G, self.W)
        return Y
    

# Some code to debug
if __name__ == '__main__':
    print "Ejecutando codigo de debug"
    for i in xrange(3):
        mean = np.random.rand(2)*30 + (np.random.rand(2)*10)
        print "Creating cluster with center: " + str(mean)
        samples = np.random.normal(size=[1000, 2], loc=mean)
        n=0
        for sample in samples:
            sample = np.append(sample, (i))
            if n==0 and i==0:
                data = sample
            else:
                data = np.vstack((data, sample))
            n+=1
    np.random.shuffle(data) # Shuffle data to avoid dataset to be ordered by centroid    cdt.plotDataSet()
    
    rbfnn = RBFNN(2, 200, 1, 'knn', 'estaSinImplementar')
    rbfnn.train(data[:int(0.75*3000),:-1], data[:int(0.75*3000),-1])
    Y = rbfnn.test(data[int(0.75*3000)+1:,:-1])
    vY = data[int(0.75*3000+1):,-1]
    
    hits = 0.0
    fails = 0.0
    for cy, y in enumerate(Y):
        if y == vY[cy]:
            hits +=1
        else:
            fails +=1
    print "Hits: ", hits
    print "Fais: ", fails        
    print "Performance", hits/len(vY)
     
        