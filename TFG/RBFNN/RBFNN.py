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
from scipy.cluster.vq import kmeans
from matplotlib import pyplot as plt
from benchmarks import DataGenerator
from scipy.optimize.optimize import fmin_cg

 
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
        # TODO: Aqui falta el bias que flipas
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G
    
    def _costFunction(self, W, *args):
        X, Y = args
        ej = 0.0
        # Error computation
        for i, xi in enumerate(X):
            G = zeros(self.numCenters, float)
            for ci, c in enumerate(self.centers):
                G[ci] = self._basisfunc(c, xi)
            fx = dot(np.transpose(G), W)
            ej += (Y[i]-fx)**2
        ej /= 2
        return ej
    
    def _gradientWeights(self, W, *args):
        X, Y = args
        grad = zeros(shape(W), float)
        ej = self._costFunction(W, X, Y)
        for i, xi in enumerate(X):
            for j, cj in enumerate(self.centers):
                grad[j] += ej*self._basisfunc(cj, xi)
        return grad
    
    def _cgMinimization(self, X, Y):
        w = self.W
        print fmin_cg(self._costFunction, w, fprime=None, args=(X,Y))
#         res = minimize(loglikelihood, (0.01, 0.1,0.1), method = 'Nelder-Mead',args = (atimes,))
    
    def _gradientDescent(self, X, Y, iterations):
        error = np.zeros(iterations, float)
        G = self._calcAct(X)
        print G
        for it in xrange(iterations):
            ej = self._costFunction(self.W, X, Y)
            gj = self._gradientWeights(self.W, X, Y)
            error[it] = ej;
            print "El error para la iteracion %d es %.15f y el gradiente:\n"%(it, ej), gj
            self.W = self.W - 0.1*gj
        plt.clf()
        plt.plot(xrange(iterations), error)
        plt.show()
 
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
        else:
            print "You must set the training method"
            return
#         print "center", self.centers
        # calculate activations of RBFs
        self.G = self._calcAct(X)
#         print G
         
        self._gradientDescent(X, Y, 7000)
#         self._cgMinimization(X, Y)
        # calculate output weights (pseudoinverse)
#         self.W = dot(pinv(self.G), Y)
        
         
    def test(self, X):
        """ X: matrix of dimensions n x indim """
         
        G = self._calcAct(X)
#         print "G=", G
        Y = dot(G, self.W)
        return Y
    

# Some code to debug
if __name__ == '__main__':
    print "Ejecutando codigo de debug"
    dim=2
    nSamples=500
    dataGenerator = DataGenerator()
#     dataGenerator.generateClusteredRandomData(nSamples, 2, dim)
    dataGenerator.generateRealData('cancer', False)

    print "InDim: ", len(dataGenerator.getTrainingX()[0])
    rbfnn = RBFNN(len(dataGenerator.getTrainingX()[0]), 5, 1, 'knn')
    rbfnn.train(dataGenerator.getTrainingX(), dataGenerator.getTrainingY())
    dataGenerator.verifyResult(rbfnn.test(dataGenerator.getValidationX()))
    
    #Plotting data
    colors = ["red", "blue"]
    for i,x in enumerate(dataGenerator.getTrainingX()):
        plt.plot(x[0], x[1], "o", color= colors[0] if dataGenerator.getTrainingY()[i]>0 else colors[1])
    plt.xlabel("First dimension")
    plt.ylabel("Second dimension")
    plt.show()
        