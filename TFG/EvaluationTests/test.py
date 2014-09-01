'''
Created on Jul 28, 2014
    By executing this file you'll be executing and evaluating the test written in this package
@author: antonio
'''
from EvaluationTests.DataGenerator import DataGenerator
from basicRBFNN.RBFNN import RBFNN
from sklearn import preprocessing
import numpy as np
import time
import shlex, subprocess
from matplotlib import pyplot as plt
import Strings
import os

def RealDataTest(dataSet="cancer", minCentroids=10, maxCentroids=200, stepCentroids=20):
    latexReport = ""
    try:
        os.mkdir("plots/") #Creo, si no esta, la carpeta en la que guardar las figuras
    except:
        pass
    dataGenerator = DataGenerator()
    dataGenerator.generateRealData("cancer")
    j = 0            
    nC = np.zeros(maxCentroids/stepCentroids)
    rperf = np.zeros(maxCentroids/stepCentroids)
    rtT = np.zeros(maxCentroids/stepCentroids)
    kperf = np.zeros(maxCentroids/stepCentroids)
    ktT = np.zeros(maxCentroids/stepCentroids)
    for nc in xrange(minCentroids, maxCentroids, stepCentroids):
        print "Testing RBFNN looking for %d centroids"%(nc)
        rrbfnn = RBFNN(len(dataGenerator.getTrainingX()[0]), nc, 1, "random")
        krbfnn = RBFNN(len(dataGenerator.getTrainingX()[0]), nc, 1, "knn")
        #Training and verifying results by randomly chosen centroids
        #Pruebo a escalar
        t = time.clock()
        rrbfnn.train(preprocessing.scale(dataGenerator.getTrainingX()), dataGenerator.getTrainingY())
        rtT[j] = time.clock()-t
#         print "la salida de la red es:",
        rperf[j] = dataGenerator.verifyResult(np.around(rrbfnn.test(preprocessing.scale(dataGenerator.getValidationX()))))
        #Training and verifying results by k-means clustering
        t = time.clock()
        krbfnn.train(dataGenerator.getTrainingX(), dataGenerator.getTrainingY())
        ktT[j] = time.clock()-t
        kperf[j] = dataGenerator.verifyResult(np.around(krbfnn.test(dataGenerator.getValidationX())))
        #Training and verifying results with metaplasticity modified weights
        nC[j] = nc
        j+=1
    
    #Writing table of results
    latexReport += '''\\begin{tabular}{|l | c | r|} \\hline Centroids & Performance & Training time \\\\ \\hline'''
    for j in range(len(nC)):
        latexReport += '''%d & %s & %s\\\\ \\hline''' % (nC[j], str(kperf[j]), str(ktT[j])) #TODO: Tengo que reconstruir la tabla para que entre todo
    latexReport += '''\\end{tabular}'''
    return latexReport
        
        
        
    
def ClusteredDataTest(minCentroids=2, maxCentroids=200, stepCentroids=10, dim=2):
    latexReport = ""
    try:
        os.mkdir("plots/") #Creo, si no esta, la carpeta en la que guardar las figuras
    except:
        pass
    for n in xrange(500, 1000, 100): # Tomo diferentes tamanyos de training sets
        latexReport += '''\\section{For a data set of %d random samples.}''' % (n)
        print "Performing test with clustered data for %d random samples"% (n)
        for i in xrange(3): #TODO: Parametrizar los centroides en el dataset
            print "For %d clusters in the data set"%(i+2)
            latexReport += '''\\subsubsection{For %d clusters in the data set.}''' % (i+2)
            dataGenerator = DataGenerator()
            dataGenerator.generateClusteredRandomData(n, i+2, dim)
            dataGenerator.savePlot()
#             latexReport += '''\\begin{figure}[!h]{}
#                                 \\centering
#                                 \\includegraphics[width=0.4\\textwidth]{plots/%ds.png}
#                                 \\label{fig:clusteredData1}
#                             \\end{figure}''' % (n)
            j = 0
            steps = (maxCentroids-minCentroids)/stepCentroids
            nC = np.zeros(steps)
            rperf = np.zeros(steps)
            rtT = np.zeros(steps)
            kperf = np.zeros(steps)
            ktT = np.zeros(steps)
            for nc in xrange(minCentroids, maxCentroids, stepCentroids): # Test the network for an incremental number of centroids
                print "Testing RBFNN looking for %d centroids"%(nc)
                rrbfnn = RBFNN(dim, nc, 1, "random")
                krbfnn = RBFNN(dim, nc, 1, "knn")
                #Training and verifying results by randomly chosen centroids
                t = time.clock()
                rrbfnn.train(dataGenerator.getTrainingX(), dataGenerator.getTrainingY())
                rtT[j] = time.clock()-t
                rperf[j] = dataGenerator.verifyResult(np.around(rrbfnn.test(dataGenerator.getValidationX())))
                #Training and verifying results by k-means clustering
                t = time.clock()
                krbfnn.train(dataGenerator.getTrainingX(), dataGenerator.getTrainingY())
                ktT[j] = time.clock()-t
                kperf[j] = dataGenerator.verifyResult(np.around(krbfnn.test(dataGenerator.getValidationX())))
                #Training and verifying results with metaplasticity modified weights
                nC[j] = nc
                j+=1
            
            #Writing table of results
            latexReport += '''\\onecolumn\\begin{center}\\begin{tabular}{|c|c|c|c|c|c|c|}
                        \\hline
                        \\multicolumn{7}{|c|}{RBFNNs performance over n samples} \\\\
                        \\hline
                        \\multirow{2}{*}{nC} & \multicolumn{2}{|c|}{Random centroids} & \multicolumn{2}{|c|}{Knn} & \multicolumn{2}{|c|}{Metaplasticity} \\\\
                        %\\hline
                        & Time & Accuracy & Time & Accuracy & Time & Accuracy \\\\
                        \\hline'''
            for j in range(len(nC)):
                latexReport += '''%d & %s & %s & %s & %s & %s & %s \\\\ \\hline''' % (nC[j], str(rtT[j]), str(rperf[j]), str(ktT[j]), str(kperf[j]), "na", "na")
            latexReport += '''\\end{tabular}\\end{center}\\twocolumn\n'''
                
            #Plotting accuracy over nCentroids
            #TODO: Tengo que hacer un multiplot. quedara mejor
            plt.clf()
            plt.plot(nC, rperf, "-b")
            plt.plot(nC, kperf, "-r")
            plt.title("Accuracy of prediction over the number of centroids to be looked for")
            plt.xlabel("Number of centroids")
            plt.ylabel("Accuracy (%)")
            plt.ylim(0,100)
            plt.legend(["Randomed", "K-means"])
            plt.savefig("plots/Accuracy%dc%ds.png" % (nc, n))
            latexReport += '''\n\\begin{figure}[]{}
                            \\centering
                            \\includegraphics[width=0.4\\textwidth]{plots/Accuracy%dc%ds.png}
                            \\label{fig:clusteredData1}
                            \\end{figure}\n''' % (nc, n)
            #Plotting time over nCentroids
            plt.clf()
            plt.plot(nC, rtT, "-b")
            plt.plot(nC, ktT, "-r")
            plt.title("Time consumed in training over the number of centroids to be looked for")
            plt.xlabel("Number of centroids")
            plt.ylabel("Time (s)")
            plt.legend(["Randomed", "K-means"])
            plt.savefig("plots/Time%dc%ds.png" % (nc, n))
            latexReport += '''\n\\begin{figure}[]{}
                            \\centering
                            \\includegraphics[width=0.4\\textwidth]{plots/Time%dc%ds.png}
                            \\label{fig:clusteredData1}
                            \\end{figure}\n''' % (nc, n)
    return latexReport

if __name__ == '__main__':
    print 'Performing test over the RBFNNs'
    latexReport = Strings.headerReportClusterTemplate #Incluyo la cabecera tex
    
    latexReport += ClusteredDataTest(minCentroids = 200,
                                maxCentroids = 400,
                                stepCentroids = 50,
                                dim = 10)
    
    latexReport += RealDataTest(dataSet='cancer',       #Data set to test 
                                minCentroids = 10,     #Min number of centroids (neurons in hiddent layer
                                maxCentroids = 50,    #Max centroids
                                stepCentroids = 10)     

#     latexReport += RealDataTest(dataSet='column',       #Data set to test 
#                                 minCentroids = 10,     #Min number of centroids (neurons in hiddent layer
#                                 maxCentroids = 50,    #Max centroids
#                                 stepCentroids = 10)     
    
    latexReport += '''\\end{document}'''
    with open('report.tex','w') as f:
        f.write(latexReport.encode('utf8')) #Escribo el archivo .tex
    fname='report'
    proc=subprocess.Popen(shlex.split('pdflatex %s.tex'%(fname))) #Genero el pdf del informe
    proc.communicate()
    proc=subprocess.Popen(shlex.split('rm %s.aux %s.idx %s.log %s.tex %s.toc'%(fname,fname,fname,fname,fname)))
    proc.communicate()
    