'''
Created on Jul 28, 2014

@author: antonio
'''

from matplotlib import pyplot as plt
import numpy as np
from basicRBFNN import RBFNN

class ClusteredDataGenerator(object):
    '''
    Esta clase preparara el entorno, ejecutara pruebas de clasisficacion sobre M elementos
    que siguen una distribucion gausiana alrededor de un numero N de centroides aleatorios.
    '''


    def __init__(self, nSamples, nCentroids, indim):
        '''
        Constructor
        '''
        # Params setting
        self.nSamples = nSamples
        self.nCentroids = nCentroids
        self.indim = indim
        self._generateData()
        
    def _generateData(self):
        for i in range(self.nCentroids):
            mean = np.random.rand(2)*10 + (np.random.rand(2)*10)
            print "Creating cluster with center: " + str(mean)
            samples = np.random.normal(size=[self.nSamples, 2], loc=mean)
            n=0
            for sample in samples:
                sample = np.append(sample, (i))
                if n==0 and i==0:
                    self.data = sample
                else:
                    self.data = np.vstack((self.data, sample))
                n+=1
        np.random.shuffle(self.data) # Shuffle data to avoid dataset to be ordered by centroid
    
    def plotDataSet(self):
        colors = ["red", "blue" , "green", "orange", "purple"]
        for i in range(len(self.getX())):
            plt.plot(self.getX()[i,0], self.getX()[i,1], "o", color=colors[int(self.getY()[i])])
        plt.title("Plot of two first features of each sample in the dataset")
        plt.xlabel("First dimension")
        plt.ylabel("Second dimension")
        plt.show()
        
    def getX(self):
        return self.data[:,:-1]
        
    def getY(self):
        return self.data[:,-1]
        
    def getDataSet(self):
        return self.getX(), self.getY()
            
    def verifyResult(self, Y):
        print Y
        hits = 0.0
        fails = 0.0
        truePositive = 0
        falsePositive = 0
        trueNegative = 0
        falseNegative = 0
        for cy, y in enumerate(Y):
            if y == self.data[cy,-1]:
                hits +=1
            else:
                fails +=1
        
        #Printing results
        print "Hits: ", hits
        print "Fais: ", fails
        
        print "Performance", hits/(self.nSamples*self.nCentroids)
        
# Some code to debug
if __name__ == '__main__':
    cdt = ClusteredDataGenerator(100, 3, 2)
    cdt.plotDataSet()
    print cdt.getDataSet()