'''
Created on Jul 28, 2014
    By executing this file you'll be executing and evaluating the test written in this package
@author: antonio
'''
from EvaluationTests.ClusteredDataGenerator import ClusteredDataGenerator
from basicRBFNN.RBFNN import RBFNN
import numpy as np

if __name__ == '__main__':
    print 'Performing test over the RBFNNs'
    # Performing test based on clustered data
    cdt = ClusteredDataGenerator(1000, 3, 2)
#     cdt.plotDataSet()
    rbfnn = RBFNN(2, 500, 1)
    rbfnn.train(cdt.getX(), cdt.getY())
    cdt.verifyResult(rbfnn.test(cdt.getX()))