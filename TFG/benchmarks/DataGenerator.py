'''
Created on Jul 28, 2014

@author: antonio
'''

from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
import csv

class DataGenerator(object):
    '''
    Esta clase preparara el entorno, ejecutara pruebas de clasisficacion sobre M elementos
    que siguen una distribucion gausiana alrededor de un numero N de centroides aleatorios.
    '''


    def __init__(self):
        '''
        Constructor
        '''
        pass
        
    def generateClusteredRandomData(self, nSamples=500, nCentroids=2, dim=2):
        #TODO: Para mas de dos clases tengo que jugar con el outdim y calcular las neuronas de la ultima capa y tal
        for i in xrange(nCentroids):
            #TODO: Parametrizar la media
            mean = np.random.rand(dim)*30 + (np.random.rand(dim)*10)
#             print "Creating cluster with center: " + str(mean)
            samples = np.random.normal(size=[nSamples, dim], loc=mean)
            for n, sample in enumerate(samples):
                sample = np.append(sample, (1 if i==0 else -1))
                if n==0 and i==0:
                    self.data = sample
                else:
                    self.data = np.vstack((self.data, sample))
        np.random.shuffle(self.data) # Shuffle data to avoid dataset to be ordered by centroid
    
    def generateRealData(self, dataSet='column', scale=True):
        if dataSet == "cancer": #Wisconsi breast cancer dataset
            with open('DataSets/wdbc.data', 'rb') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for ns, sample in enumerate(spamreader):
                    s = str.split(sample[0], ',')
                    if s[1]=='B': 
                        t=1.0
                    else:
                        t=-1.0
                    sample = np.append(np.array(s[2:], float), t)
    #                 print sample , ns
                    if ns == 0:
                        self.data = sample
                    else:
                        self.data = np.vstack((self.data, sample))
        elif dataSet == "column":
            with open('DataSets/column_2C.data', 'rb') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for ns, sample in enumerate(spamreader):
                    s = str.split(sample[0], ',')
                    if s[-1]=='Abnormal': 
                        t=-1.0
                    else:
                        t=1.0
                    sample = np.append(np.array(s[:-1], float), t)
    #                 print sample , ns
                    if ns == 0:
                        self.data = sample
                    else:
                        self.data = np.vstack((self.data, sample))
        np.random.shuffle(self.data)
        
        if scale:
            self.data[:,:-1] = preprocessing.scale(self.data[:,:-1])
    
    def plotDataSet(self):
        colors = ["red", "blue" , "green", "orange", "purple"]
        ts = self.getTrainingX()
        for i in xrange(len(self.getTrainingX())):
            plt.plot(ts[i,0], ts[i,1], "o", color=colors[int(self.getTrainingY()[i])])
        plt.title("Plot of two first features of each sample in the dataset")
        plt.xlabel("First dimension")
        plt.ylabel("Second dimension")
        plt.show()
        
    def savePlot(self):
        plt.clf()
        colors = ["red", "blue" , "green", "orange", "purple"]
        ts = self.getTrainingX()
        for i in xrange(len(self.getTrainingX())):
            plt.plot(ts[i,0], ts[i,1], "o", color=colors[int(self.getTrainingY()[i])])
        plt.title("Plot of two first features of each sample in the dataset")
        plt.xlabel("First dimension")
        plt.ylabel("Second dimension")
        plt.savefig("plots/"+str(len(self.data))+"s"+".png")
        
    def getTrainingX(self):
        '''It returns the first 75% of the X in dataSet for training'''
        return self.data[:int(0.75*len(self.data)),:-1]
        
    def getTrainingY(self):
        '''It returns the first 75% of the Y in dataSet for training'''
        return self.data[:int(0.75*len(self.data)),-1]
    
    def getValidationX(self):
        '''It returns the last 25% of the X in dataSet for verifying'''
        return self.data[int(0.75*len(self.data))+1:,:-1]
    
    def getValidationY(self):
        '''It returns the last 25% of the X in dataSet for verifying'''
        return self.data[int(0.75*len(self.data)+1):,-1]
        
    def getDataSet(self):
        #TODO: Esto ta mu mal
        return self.getTrainingX(), self.getTrainingY()
    
            
    def verifyResult(self, Y):
        vY = self.getValidationY()
        hits = 0.0
        fails = 0.0
        #TODO: Tengo que hacer una verificacion mas exhaustiva
        truePositive = 0
        falsePositive = 0
        trueNegative = 0
        falseNegative = 0
        for cy, y in enumerate(Y):
            if y > 0 and vY[cy] > 0:
                hits +=1
            elif y < 0 and vY[cy] < 0:
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
    cdt = DataGenerator()
    print cdt.getDataSet()
    print len(cdt.getTrainingX())+len(cdt.getValidationX())