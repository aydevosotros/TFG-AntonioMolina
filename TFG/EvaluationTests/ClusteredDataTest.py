'''
Created on Jul 28, 2014

@author: antonio
'''

from matplotlib import pyplot as plt
import numpy as np

class ClusteredDataTest(object):
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
        
        self.data = []
        self.generateData()
        
    def generateData(self):
        for i in range(self.centroids):
            mean = np.random.rand(2)*10 + (np.random.rand(2)*10)
            print "Creating cluster with center: " + str(mean)
            sample = [i, np.random.normal(size=[self.samples, 2], loc=mean)]
            self.data.append(sample)
    
    def plotDataSet(self):
        print "Muestro los datos antes de imprimir"
        print self.data[0][1]
        colors = ["red", "blue" , "green", "orange", "purple"]
        for i in range(self.centroids):
            plt.plot(self.data[i][1][:,0], self.data[i][1][:,1], "o", color=colors[i])
        plt.show()
        
        
        
# Some code to debug
if __name__ == '__main__':
    cdt = ClusteredDataTest(centroids=3, samples=100)
    cdt.plotDataSet()