'''
Created on Nov 13, 2014
En este experimento se tratara de encontrar el numero de controides optimos 
para el dataset de columna
@author: antonio
'''

import os
from benchmarks.ReportGenerator import LatexGenerator, LatexReport
from benchmarks.DataGenerator import DataGenerator
from RBFNN.RBFNN import RBFNN
import numpy as np
from matplotlib import pyplot as plt
from benchmarks import ReportGenerator

if __name__ == '__main__':
    #Inicializo los componentes principales
    latexReport = LatexReport("Pruebas de experimentacion", "Antonio Molina")
    dataGenerator = DataGenerator()
#     dataGenerator.generateClusteredRandomData(300, 10, 1)
    dataGenerator.generateRealData('cancer', True)
    
    #Obtengo training y validation sets
    trainingX = dataGenerator.getTrainingX()
    trainingY = dataGenerator.getTrainingY()
    validatingX = dataGenerator.getValidationX()    
    
    #Inicializo los parametros del experimento
    minNc = 2
    maxNc = 11
    stepNc = 1
    meanSamples = 5
    steps = (maxNc-minNc)/stepNc
    results = np.zeros((steps,5), float)
    beta = np.zeros(steps)
    kperf = np.zeros(steps)
    j=0
    
    # Realizo la experimentacion
    for nc in xrange(minNc, maxNc, stepNc):
        print "Testing RBFNN looking for %d centroids"%(nc)     
        for k in xrange(meanSamples):   
            krbfnn = RBFNN(len(trainingX[0]), nc, 1, "knn", "cgmin", metaplasticity=False)
            rbfnn2 = RBFNN(len(trainingX[0]), nc, 1, "knn", "cgmin", metaplasticity=True)
            #Training and verifying results by k-means clustering
            krbfnn.train(trainingX, trainingY)
            rbfnn2.train(trainingX, trainingY)
            results[j][1] += dataGenerator.verifyResult(krbfnn.test(validatingX))
            results[j][2] += krbfnn.gradEval
            results[j][3] += dataGenerator.verifyResult(rbfnn2.test(validatingX))
            results[j][4] += rbfnn2.gradEval
            # Ploteo
            plt.clf()
#             plt.plot(range(len(krbfnn.allvec)), [(krbfnn._costFunction(wx, trainingX, trainingY)) for wx in krbfnn.allvec])
#             plt.plot(range(len(rbfnn2.allvec)), [(rbfnn2._costFunction(wx, trainingX, trainingY)) for wx in rbfnn2.allvec])
#             plt.plot(range(len(krbfnn.gError)), krbfnn.gError)
#             plt.plot(range(len(rbfnn2.gError)), rbfnn2.gError)
            plt.show()
        results[j][1] /= meanSamples
        results[j][2] /= meanSamples
        results[j][3] /= meanSamples
        results[j][4] /= meanSamples
        results[j][0] = nc
        j+=1

    print "Generando informe"
    latexReport.addSection("Una primera prueba")
    latexReport.addContent("En esta experimentacion se trata de obtener un error razonable al hacer un descenso por gradiente")
    latexReport.addContent(ReportGenerator.LatexGenerator.generateTable(results, ["Centroides", "Metodo1", "Metodo2"]))
    latexReport.createPDF("r1")
    