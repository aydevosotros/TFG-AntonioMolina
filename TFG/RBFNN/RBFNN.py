'''
Created on Jun 14, 2014
Based of a RBFNN for python written by Thomas Rueckstiess. 
http://www.rueckstiess.net/research/snippets/show/72d2363e
Added by Antonio Molina:
    - K-means clustering

@author: antonio
'''

from enum import Enum
from scipy import *
import numpy as np
from scipy.linalg import norm, pinv
from scipy.cluster.vq import kmeans
from matplotlib import pyplot as plt
from benchmarks.DataGenerator import DataGenerator
from scipy.optimize.optimize import fmin_cg
from scipy.optimize import minimize

# class MinimizationMethods(Enum):
#     metaplasticity = 'metaplasticity'
#     NelderMead = 'Nelder-Mead'
#     Powell = 'Powell'
#     CG = 'CG'
#     BFGS = 'BFGS'
#     NewtonCG = 'Newton-CG'
#     LBFGSB = 'L-BFGS-B'
#     TNC = 'TNC'

# MinimizationMethods = Enum(METAPLASTICITY = 'metaplasticity', NELDERMEAD = 'Nelder-Mead', POWELL = 'Powell', CG = 'CG', BFGS = 'BFGS', NEWTONGC = 'Newton-CG', LBFGSB = 'L-BFGS-B', TNC = 'TNC', COBYLA = 'COBYLA')
 
class RBFNN(object):
     
    def __init__(self, indim, numCenters, outdim, trainingCentroidsMethod="knn", trainingWeightsMethod="BFGS", beta=8):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in xrange(numCenters)]
        self.beta = beta
        self.trainingCentroidsMethod=trainingCentroidsMethod
        self.trainingWeightsMethod = trainingWeightsMethod
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
        gError = zeros(0)
        # Error computation
        for i, xi in enumerate(X):
            fx = dot(transpose(self.G[i]), W)
            ej += (Y[i]-fx)**2
        ej /= len(X)
        append(gError, ej)
        return ej
    
    def _partialCost(self, W, *args):
        X, Y = args
        grad = zeros(shape(W), float)
        ej = self._costFunction(W, X, Y)
        for i in xrange(len(W)):
            for j in xrange(len(X)):
                ej = (Y[i]-dot(transpose(self.G[i]), W))**2
                grad[i] += ej*self.G[j,i]
        return grad
    
    def _stimateProbability(self, x):
        pass
    
    def _minimization(self, X, Y, method=None):
        w = np.copy(self.W)
        self.it = 0
        if method == 'metaplasticity':
            res = minimize(self._costFunction, w, method = 'Nelder-Mead', args = (X,Y))
        else:
            res = minimize(self._costFunction, w, method = method, args = (X,Y))
        print res
        self.W = res.x
        
    def _cgmin(self, X, Y):
        w = np.copy(self.W)
        res = fmin_cg(self._costFunction, w, fprime=self._partialCost, args=(X,Y), retall=True)
        plt.clf()
        plt.plot(range(len(res[1])), [(self._costFunction(wx, X, Y)) for wx in res[1]])
        plt.show()
        
    def _gcb(self, xk):
        self.it+=1
        print 'The cost for the it: ', self.it, ' is: ', xk
    
    def _gradientDescent(self, X, Y, iterations):
        error = np.zeros(iterations, float)
        G = self._calcAct(X)
        for it in xrange(iterations):
            ej = self._costFunction(self.W, X, Y)
            gj = self._partialCost(self.W, X, Y)
            error[it] = ej;
            print "El error para la iteracion %d es %.15f y el gradiente:\n"%(it, ej), gj
            self.W = self.W - 0.1*gj
        plt.clf()
        plt.plot(range(iterations), error)
#         plt.show()
 
    def train(self, X, Y):
        """ X: matrix of dimensions n x indim
            y: column vector of dimension n x 1 """
        if self.trainingCentroidsMethod == "random":
            print 'Training with randomly chosen centroids'
            rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
            self.centers = [X[i,:] for i in rnd_idx]
        elif self.trainingCentroidsMethod == "knn":
            print 'Training with centroids from k-means algorithm'
            self.centers = kmeans(X, self.numCenters)[0]
        else:
            print "You must set the training method"
            return

        # calculate activations of RBFs
        self.G = self._calcAct(X)
#         print G
         
#         self._gradientDescent(X, Y, 7000)
        if self.trainingWeightsMethod == "pseudoinverse":
            self.W = dot(pinv(self.G), Y)
        elif self.trainingWeightsMethod == "cgmin":
            self._cgmin(X, Y)
        else:
#             self._gradientDescent(X, Y, 100)
            self._minimization(X, Y, self.trainingWeightsMethod)
        # calculate output weights (pseudoinverse)
        
         
    def test(self, X):
        """ X: matrix of dimensions n x indim """
        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y
    

# Some code to debug
if __name__ == '__main__':
    print "Ejecutando codigo de debug"
    dim=2
    nSamples=500
    dataGenerator = DataGenerator()
    dataGenerator.generateClusteredRandomData(nSamples, 2, dim)
#     dataGenerator.generateRealData('cancer', False)

    print "InDim: ", len(dataGenerator.getTrainingX()[0])
    rbfnn = RBFNN(len(dataGenerator.getTrainingX()[0]), 5, 1, 'knn', 'BFGS')
    rbfnn.train(dataGenerator.getTrainingX(), dataGenerator.getTrainingY())
    dataGenerator.verifyResult(rbfnn.test(dataGenerator.getValidationX()))
    
    #Plotting data
#     colors = ["red", "blue"]
#     for i,x in enumerate(dataGenerator.getTrainingX()):
#         plt.plot(x[0], x[1], "o", color= colors[0] if dataGenerator.getTrainingY()[i]>0 else colors[1])
#     plt.xlabel("First dimension")
#     plt.ylabel("Second dimension")
#     plt.show()
        