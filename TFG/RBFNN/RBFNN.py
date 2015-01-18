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
from scipy.optimize.optimize import fmin_cg, fmin_bfgs
from scipy.optimize import minimize
from scipy.spatial import distance

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
     
    def __init__(self, indim, numCenters, outdim, trainingCentroidsMethod="knn", trainingWeightsMethod="BFGS", beta=1.0/8.0, metaplasticity=False):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in xrange(numCenters)]
        self.beta = beta
        self.trainingCentroidsMethod=trainingCentroidsMethod
        self.trainingWeightsMethod = trainingWeightsMethod
        self.W = random.random((self.numCenters, self.outdim))
        self.metaplasticity = metaplasticity
        self.radialFunc = "iGaussian"
        self.gError = zeros(0)
         
    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c-d)**2)
    
    def _gaussianFunc(self, c, d):
        return exp(-((self.beta * norm(c-d))**2))
    
    def _isotropicGaussian(self, c, d):
        return exp((-(self.numCenters/(self.dm)**2))*(distance.euclidean(c, d)**2))    
     
    def _calcAct(self, X):
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                if self.radialFunc == "gaussian":
                    G[xi,ci] = self._gaussianFunc(c, x)
                elif self.radialFunc == "iGaussian":
                    G[xi,ci] = self._isotropicGaussian(c, x)
                else:
                    G[xi,ci] = self._basisfunc(c, x)
        return G
    
    def _costBasic(self, W, *args):
        X, Y = args
        ej = 0.0
        # Error computation
        for i, xi in enumerate(X):
            fx = dot(transpose(self.G[i]), W)
            ej += ((Y[i]-fx)**2)
        ej = ej/len(X)
        self.gError = append(self.gError, ej)
        return ej

    
    def _costFunction(self, W, *args):
        X, Y = args
        ej = 0.0
        # Error computation
        for i, xi in enumerate(X):
            fx = dot(transpose(self.G[i]), W)
            p = (1.0) if not self.metaplasticity else 1-self.P[i]
            ej += ((Y[i]-fx)**2)*(1/p)
        ej = ej/len(X)
        self._costBasic(W, X, Y)
        return ej

    
    def _pEstimator(self, x):
        p=0.0
        for k in xrange(self.numCenters):
            p+=self._isotropicGaussian(self.centers[k], x)
        return p/self.numCenters
        
    
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
        res = fmin_bfgs(self._costFunction, w, args=(X,Y), full_output=1, retall=1)
        self.allvec = res[-1]
        self.W = res[0]
        self.gradEval = res[-2]
        
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
        # Para la gaussiana isotropica calculo la distncia maxima
        dis = [distance.euclidean(self.centers[i], self.centers[j]) for i in range(self.numCenters) for j in range(self.numCenters) if i != j]
        self.dm = np.amax(dis)
        self.P = [self._pEstimator(x[1]) for x in enumerate(X)]
        self.G = self._calcAct(X)
#         print self.G
         
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
#     dataGenerator.generateClusteredRandomData(nSamples, 2, dim)
    dataGenerator.generateRealData('cancer', True)
    
    perf = 0.0
    print "InDim: ", len(dataGenerator.getTrainingX()[0])
    for i in range(10):
        rbfnn = RBFNN(len(dataGenerator.getTrainingX()[0]), 2, 1, 'knn', 'cgmin', metaplasticity=True)
        rbfnn.train(dataGenerator.getTrainingX(), dataGenerator.getTrainingY())
        perf += dataGenerator.verifyResult(rbfnn.test(dataGenerator.getValidationX()))
    print "El rendimiento medio es: ", perf/10
    #Plotting data
#     colors = ["red", "blue"]
#     for i,x in enumerate(dataGenerator.getTrainingX()):
#         plt.plot(x[0], x[1], "o", color= colors[0] if dataGenerator.getTrainingY()[i]>0 else colors[1])
#     plt.xlabel("First dimension")
#     plt.ylabel("Second dimension")
#     plt.show()
        