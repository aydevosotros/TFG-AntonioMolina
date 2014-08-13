'''
Created on Jul 28, 2014
    By executing this file you'll be executing and evaluating the test written in this package
@author: antonio
'''
from EvaluationTests.ClusteredDataTest import ClusteredDataGenerator

if __name__ == '__main__':
    print 'Performing test over the RBFNNs'
    # Performing test based on clustered data
    cdt = ClusteredDataGenerator(centroids=3, samples=10)
    cdt.plotDataSet()