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
        self.totalSamples = nSamples
        self.nSamples = nSamples / nCentroids
        self.nCentroids = nCentroids
        self.indim = indim
        self._generateData()
        #TODO: Tengo que hacer también para probar para diferentes dimensiones de entrada
        
    def _generateData(self):
        for i in range(self.nCentroids):
            mean = np.random.rand(2)*30 + (np.random.rand(2)*10)
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
        ts = self.getTrainingX()
        for i in range(len(self.getTrainingX())):
            plt.plot(ts[i,0], ts[i,1], "o", color=colors[int(self.getTrainingY()[i])])
        plt.title("Plot of two first features of each sample in the dataset")
        plt.xlabel("First dimension")
        plt.ylabel("Second dimension")
        plt.show()
        
    def savePlot(self):
        plt.clf()
        colors = ["red", "blue" , "green", "orange", "purple"]
        ts = self.getTrainingX()
        for i in range(len(self.getTrainingX())):
            plt.plot(ts[i,0], ts[i,1], "o", color=colors[int(self.getTrainingY()[i])])
        plt.title("Plot of two first features of each sample in the dataset")
        plt.xlabel("First dimension")
        plt.ylabel("Second dimension")
        plt.savefig("plots/"+str(self.nCentroids)+"c"+str(self.totalSamples)+"s"+".png")
        
    def getTrainingX(self):
        '''It returns the first 75% of the X in dataSet for training'''
        print "El trainingSet tiene: ", len(self.data[:int(0.75*(self.nCentroids*self.nSamples)),:-1])
        return self.data[:int(0.75*(self.nCentroids*self.nSamples)),:-1]
        
    def getTrainingY(self):
        '''It returns the first 75% of the Y in dataSet for training'''
        return self.data[:int(0.75*(self.nCentroids*self.nSamples)),-1]
    
    def getValidationX(self):
        print "El validationTest tiene: ", len(self.data[int(0.75*(self.nCentroids*self.nSamples))+1:,:-1])
        '''It returns the last 25% of the X in dataSet for verifying'''
        return self.data[int(0.75*(self.nCentroids*self.nSamples))+1:,:-1]
    
    def getValidationY(self):
        '''It returns the last 25% of the X in dataSet for verifying'''
        return self.data[int(0.75*(self.nCentroids*self.nSamples)+1):,-1]
        
    def getDataSet(self):
        return self.getTrainingX(), self.getTrainingY()
            
    def verifyResult(self, Y):
        vY = self.getValidationY()
        hits = 0.0
        fails = 0.0
        truePositive = 0
        falsePositive = 0
        trueNegative = 0
        falseNegative = 0
        for cy, y in enumerate(Y):
            if y == vY[cy]:
                hits +=1
            else:
                fails +=1
        
        #Printing results
        print "Hits: ", hits
        print "Fais: ", fails        
        print "Performance", hits/len(vY)
        
        return hits/len(vY)*100
        
# Some code to debug
if __name__ == '__main__':
    cdt = ClusteredDataGenerator(1000, 3, 2)
    cdt.plotDataSet()
    print cdt.getDataSet()
    print len(cdt.getTrainingX())+len(cdt.getValidationX())