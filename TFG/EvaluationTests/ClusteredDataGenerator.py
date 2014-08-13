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


    def __init__(self, **params):
        '''
        Constructor
        '''
        # Params setting
        self.samples = params["samples"]
        self.centroids = params["centroids"]
        self.indim = params["indim"]
        self._generateData()
        
    def _generateData(self):
        for i in range(self.centroids):
            mean = np.random.rand(2)*10 + (np.random.rand(2)*10)
            print "Creating cluster with center: " + str(mean)
            samples = np.random.normal(size=[self.samples, 2], loc=mean)
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
        print "Imprimiendo las X", self.getX()[3,0]
        colors = ["red", "blue" , "green", "orange", "purple"]
        for i in range(len(self.getX())):
            print "Ploteando muestra de la clase: ", colors[int(self.getY()[i])]
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
            

        
# Some code to debug
if __name__ == '__main__':
    cdt = ClusteredDataGenerator(centroids=3, samples=100, indim=2)
    cdt.plotDataSet()
    print cdt.getDataSet()