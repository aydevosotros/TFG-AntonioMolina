'''
Created on Nov 13, 2014
En este experimento se tratara de encontrar el numero de controides optimos 
para el dataset de columna
@author: antonio
'''

import os
import time
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
    dataGenerator.generateClusteredRandomData(1000, 5, 0.01)
#     dataGenerator.generateRealData('column', True)
    
    #Obtengo training y validation sets
    trainingX = dataGenerator.getTrainingX()
    trainingY = dataGenerator.getTrainingY()
    validatingX = dataGenerator.getValidationX()    
    
    #Inicializo los parametros del experimento
    minNc = 2
    maxNc = 10
    stepNc = 1
    meanSamples = 10
    steps = (maxNc-minNc)/stepNc
    results = np.zeros((steps,5), float)
    beta = np.zeros(steps)
    kperf = np.zeros(steps)
    j=0
    
    # Realizo la experimentacion
    for nc in xrange(minNc, maxNc, stepNc):
        print "Testing RBFNN looking for %d centroids"%(nc)     
        for k in xrange(meanSamples):   
            krbfnn = RBFNN(len(trainingX[0]), nc, 1, "knn", "cgmin", 1/9)
#             rbfnn2 = RBFNN(len(trainingX[0]), nc, 1, "knn", "BFGS", 9)
            rbfnn3 = RBFNN(len(trainingX[0]), nc, 1, "knn", "cgmin", 1/9, metaplasticity=True)
            #Training and verifying results by k-means clustering
            inicio = time.time()
            krbfnn.train(trainingX, trainingY)
            results[j][2] += time.time() - inicio
            inicio = time.time()
            rbfnn3.train(trainingX, trainingY)
            results[j][4] += time.time() - inicio
#             inicio = time.time()
#             rbfnn3.train(trainingX, trainingY)
#             results[j][6] += time.time() - inicio
            
            results[j][1] += dataGenerator.verifyResult(krbfnn.test(validatingX))
            results[j][3] += dataGenerator.verifyResult(rbfnn3.test(validatingX))
#             results[j][5] += dataGenerator.verifyResult(rbfnn3.test(validatingX))
        results[j][1] /= meanSamples
        results[j][2] /= meanSamples
        results[j][3] /= meanSamples
        results[j][4] /= meanSamples
#         results[j][5] /= meanSamples
#         results[j][6] /= meanSamples
        results[j][0] = nc
        j+=1

    print "Generando informe"
    latexReport.addSection("Una primera prueba")
    latexReport.addContent("En esta experimentacion se trata de obtener un error razonable al hacer un descenso por gradiente")
    latexReport.addContent(ReportGenerator.LatexGenerator.generateTable(results, ["Centroides", "cgmin", "time", "cgmin/metaplasticity", "time"]))
    latexReport.createPDF("realCancerBig")
    