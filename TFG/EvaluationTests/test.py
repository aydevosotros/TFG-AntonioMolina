'''
Created on Jul 28, 2014
    By executing this file you'll be executing and evaluating the test written in this package
@author: antonio
'''
from EvaluationTests.ClusteredDataGenerator import ClusteredDataGenerator
from basicRBFNN.RBFNN import RBFNN
import numpy as np
import time
import shlex, subprocess
from matplotlib import pyplot as plt
import Strings
import os
import csv

def RealDataTest(dataSet="cancer", minCentroids=2, maxCentroids=200, stepCentroids=10):
    latexReport = ""
    try:
        os.mkdir("plots/") #Creo, si no esta, la carpeta en la que guardar las figuras
    except:
        pass
    #load dataset
    if dataSet == "cancer": #Wisconsi breast cancer dataset
        with open('DataSets/wdbc.data', 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for ns, sample in enumerate(spamreader):
                print "la linea %d tiene: %s"%(ns, sample)
    
def ClusteredDataTest(minCentroids=2, maxCentroids=200, stepCentroids=10):
    latexReport = ""
    try:
        os.mkdir("plots/") #Creo, si no esta, la carpeta en la que guardar las figuras
    except:
        pass
    for n in xrange(500, 10000, 500): # Tomo diferentes tamanyos de training sets
        latexReport += '''\\section{For a data set of %d random samples.}''' % (n)
        print "Performing test with clustered data for %d random samples"% (n)
        for i in range(3): #TODO: Parametrizar los centroides en el dataset
            print "For %d clusters in the data set"%(i+2)
            latexReport += '''\\subsubsection{For %d clusters in the data set.}''' % (i+2)
            cdt = ClusteredDataGenerator(n, i+2, 2)
            cdt.savePlot()
            latexReport += '''\\begin{figure}[!h]{}
                                \\centering
                                \\includegraphics[width=0.4\\textwidth]{plots/%dc%ds.png}
                                \\label{fig:clusteredData1}
                            \\end{figure}''' % (i+2, n)
            j = 0            
            nC = np.zeros(maxCentroids/stepCentroids)
            rperf = np.zeros(maxCentroids/stepCentroids)
            rtT = np.zeros(maxCentroids/stepCentroids)
            kperf = np.zeros(maxCentroids/stepCentroids)
            ktT = np.zeros(maxCentroids/stepCentroids)
            for nc in xrange(minCentroids, maxCentroids, stepCentroids): # Test the network for an incremental number of centroids
                print "Testing RBFNN looking for %d centroids"%(nc)
                rrbfnn = RBFNN(2, nc, 1, "random")
                krbfnn = RBFNN(2, nc, 1, "knn")
                #Training and verifying results by randomly chosen centroids
                t = time.clock()
                rrbfnn.train(cdt.getTrainingX(), cdt.getTrainingY())
                rtT[j] = time.clock()-t
                rperf[j] = cdt.verifyResult(rrbfnn.test(cdt.getValidationX()))
                #Training and verifying results by k-means clustering
                t = time.clock()
                krbfnn.train(cdt.getTrainingX(), cdt.getTrainingY())
                ktT[j] = time.clock()-t
                kperf[j] = cdt.verifyResult(krbfnn.test(cdt.getValidationX()))
                #Training and verifying results with metaplasticity modified weights
                nC[j] = nc
                j+=1
            
            #Writing table of results
            latexReport += '''\\begin{tabular}{|l | c | r|} \\hline Centroids & Performance & Training time \\\\ \\hline'''
            for j in range(len(nC)):
                latexReport += '''%d & %s & %s\\\\ \\hline''' % (nC[j], str(kperf[j]), str(ktT[j])) #TODO: Tengo que reconstruir la tabla para que entre todo
            latexReport += '''\\end{tabular}'''
                
            #Plotting accuracy over nCentroids
            plt.clf()
            plt.plot(nC, rperf, "-b")
            plt.plot(nC, kperf, "-r")
            plt.title("Accuracy of prediction over the number of centroids to be looked for")
            plt.xlabel("Number of centroids")
            plt.ylabel("Accuracy (%)")
            plt.ylim(0,100)
            plt.legend(["Randomed", "K-means"])
            plt.savefig("plots/Accuracy%dc%ds.png" % (nc, n))
            latexReport += '''\\begin{figure}[!h]{}
                            \\centering
                            \\includegraphics[width=0.4\\textwidth]{plots/Accuracy%dc%ds.png}
                            \\label{fig:clusteredData1}
                            \\end{figure}''' % (nc, n)
            #Plotting time over nCentroids
            plt.clf()
            plt.plot(nC, rtT, "-b")
            plt.plot(nC, ktT, "-r")
            plt.title("Time consumed in training over the number of centroids to be looked for")
            plt.xlabel("Number of centroids")
            plt.ylabel("Time (s)")
            plt.legend(["Randomed", "K-means"])
            plt.savefig("plots/Time%dc%ds.png" % (nc, n))
            latexReport += '''\\begin{figure}[!h]{}
                            \\centering
                            \\includegraphics[width=0.4\\textwidth]{plots/Time%dc%ds.png}
                            \\label{fig:clusteredData1}
                            \\end{figure}''' % (nc, n)
    return latexReport

if __name__ == '__main__':
    print 'Performing test over the RBFNNs'
    latexReport = Strings.headerReportClusterTemplate #Incluyo la cabecera tex
    
    latexReport += ClusteredDataTest()
    
    latexReport += '''\\end{document}'''
    with open('report.tex','w') as f:
        f.write(latexReport.encode('utf8')) #Escribo el archivo .tex
    proc=subprocess.Popen(shlex.split('pdflatex report.tex')) #Genero el pdf del informe
    proc.communicate()
    