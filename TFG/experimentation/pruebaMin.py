'''
Created on Nov 11, 2014

@author: antonio
'''
from scipy import *
from scipy.cluster.vq import kmeans
from scipy.linalg import norm
from scipy.optimize.optimize import fmin_cg
from benchmarks import DataGenerator
from matplotlib import pyplot as plt


dmax = 700.0
gError = zeros(0)

def basisfunc(c, d):
    return exp(2/dmax * (norm(c-d)**2))

def costFunction(W, *args):
#     print "-----", W
    X, Y, G = args
    ej = 0.0
    # Error computation
    for i, xi in enumerate(X):
        fx = dot(transpose(G[i]), W)
        ej += (Y[i]-fx)**2
    ej /= len(X)
    append(gError, ej)
    return ej

def fcPrime(W, *args):
    X, Y, G = args
    grad = zeros(shape(W), float)
    ej = costFunction(W, X, Y, G)
    for i in xrange(len(W)):
        for j in xrange(len(X)):
            print "El producto es: ", dot(transpose(G[i]), W)
            ej = (Y[i]-dot(transpose(G[i]), W))
            grad[i] += ej*G[j,i]
    return grad

def gradientDescent(W, X, Y, G, iterations):
    error = zeros(iterations, float)
    for it in xrange(iterations):
        ej = costFunction(W, X, Y, G)
        gj = fcPrime(W, X, Y, G)
        error[it] = ej;
        print "El error para la iteracion %d es %.15f"%(it, ej)
        W = W - 000.1*gj
    plt.clf()
    plt.plot(xrange(iterations), error)
    plt.show()

def cbMin(xk):
    print "For this iteration, xk: ", xk

if __name__ == '__main__':
    numCenters = 2
    outdim = 1
    dg = DataGenerator()
    
    # Inicializo
    dg.generateClusteredRandomData(500, 2, 2)
#     dg.generateRealData('column', True)
    X = dg.getTrainingX()
    Y = dg.getTrainingY()
    centers = kmeans(X, numCenters)[0]
    w = random.random((numCenters, outdim))
    G = zeros((X.shape[0], numCenters), float)
    
    for ci, c in enumerate(centers):
        for xi, x in enumerate(X):
            G[xi,ci] = basisfunc(c, x)
            
#     for i in range(10):
    fmin_cg(costFunction, w, fprime=fcPrime, args=(X, Y, G))
    gradientDescent(w, X, Y, G, 5)
    print gError