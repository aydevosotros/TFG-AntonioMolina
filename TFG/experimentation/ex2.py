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
#     dataGenerator.generateClusteredRandomData(50, 2, 2)
    dataGenerator.generateRealData('column', True)
    
    #Obtengo training y validation sets
    trainingX = dataGenerator.getTrainingX()
    trainingY = dataGenerator.getTrainingY()
    validatingX = dataGenerator.getValidationX()    
    
    #Inicializo los parametros del experimento
    minBeta = 2
    maxBeta = 10
    stepCBeta = 1
    meanSamples = 5
    steps = (maxBeta-minBeta)/stepCBeta
    results = np.zeros((steps,2), float)
    beta = np.zeros(steps)
    kperf = np.zeros(steps)
    j=0
    
    # Realizo la experimentacion
    for b in xrange(minBeta, maxBeta, stepCBeta):
        print "Testing RBFNN looking for beta with value %d"%(b)        
        krbfnn = RBFNN(len(trainingX[0]), 10, 1, "knn", "gd", b)
        partResult = 0.0
        for k in xrange(meanSamples):
            #Training and verifying results by k-means clustering
            krbfnn.train(trainingX, trainingY)
            results[j][1] += dataGenerator.verifyResult(krbfnn.test(validatingX))
        results[j][1] /= meanSamples
        results[j][0] = b
        j+=1

    print "Generando informe"
    latexReport.addSection("Una primera prueba")
    latexReport.addContent("En esta experimentacion se trata de obtener un error razonable al hacer un descenso por gradiente")
    latexReport.addContent(ReportGenerator.LatexGenerator.generateTable(results, ["Beta", "Resultado"]))
    latexReport.createPDF("r1")
    