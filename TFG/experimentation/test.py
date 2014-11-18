'''
Created on Jul 28, 2014
    By executing this file you'll be executing and evaluating the test written in this package
@author: antonio
'''
from benchmarks.DataGenerator import DataGenerator
from RBFNN.RBFNN import RBFNN
from sklearn import preprocessing
import numpy as np
import time
import shlex, subprocess
from matplotlib import pyplot as plt
from benchmarks import Strings
import os

def RealDataTest(dataSet="cancer", minBeta=10, maxBeta=200, stepCBeta=20, meanSamples=10):
    latexReport = ""
    try:
        os.mkdir("plots/") #Creo, si no esta, la carpeta en la que guardar las figuras
    except:
        pass
    dataGenerator = DataGenerator()
    dataGenerator.generateRealData(dataSet)
    trainingX = preprocessing.scale(dataGenerator.getTrainingX())
    trainingY = dataGenerator.getTrainingY()
    validatingX = preprocessing.scale(dataGenerator.getValidationX())
    j = 0
    steps = (maxBeta-minBeta)/stepCBeta        
    beta = np.zeros(steps)
    rperf = np.zeros(steps)
    rtT = np.zeros(steps)
    kperf = np.zeros(steps)
    ktT = np.zeros(steps)
    for b in xrange(minBeta, maxBeta, stepCBeta):
        print "Testing RBFNN looking for %d centroids"%(b)        
        rrbfnn = RBFNN(len(trainingX[0]), b, 1, "random")
        krbfnn = RBFNN(len(trainingX[0]), b, 1, "knn")
        #Training and verifying results by randomly chosen centroids
        #TODO: Enviar el libro
        for k in xrange(meanSamples):
            t = time.clock()
            print trainingX
            rrbfnn.train(trainingX, trainingY)
            rtT[j] += time.clock()-t
    #         print "la salida de la red es:",
            rperf[j] += dataGenerator.verifyResult(rrbfnn.test(validatingX))
            #Training and verifying results by k-means clustering
            t = time.clock()
            krbfnn.train(trainingX, trainingY)
            ktT[j] += time.clock()-t
            kperf[j] += dataGenerator.verifyResult(krbfnn.test(validatingX))
            #Training and verifying results with metaplasticity modified weights
            beta[j] = b
        rtT[j] /= meanSamples
        rperf[j] /= meanSamples
        ktT[j] /= meanSamples
        kperf[j] /= meanSamples
        beta[j] = b
        j+=1
    
    #Writing table of results
    latexReport += '''\\onecolumn\\begin{center}\\begin{tabular}{|c|c|c|c|c|c|c|}
                \\hline
                \\multicolumn{7}{|c|}{RBFNNs performance over %s dataset} \\\\
                \\hline
                \\multirow{2}{*}{beta} & \multicolumn{2}{|c|}{Random centroids} & \multicolumn{2}{|c|}{Knn} & \multicolumn{2}{|c|}{Metaplasticity} \\\\
                & Time & Accuracy & Time & Accuracy & Time & Accuracy \\\\
                \\hline''' % (dataSet)
    for j in range(len(beta)):
        latexReport += '''%d & %s & %s & %s & %s & %s & %s \\\\ \\hline''' % (beta[j], str(rtT[j]), str(rperf[j]), str(ktT[j]), str(kperf[j]), "na", "na")
    latexReport += '''\\end{tabular}\\end{center}\\twocolumn\n'''
    return latexReport
        
        
        
    
def ClusteredDataTest(minBeta=2, maxBeta=200, stepCBeta=10, dim=2, meanSamples=10):
    latexReport = ""
    try:
        os.mkdir("plots/") #Creo, si no esta, la carpeta en la que guardar las figuras
    except:
        pass
    for n in xrange(100, 1000, 100): # Tomo diferentes tamanyos de training sets
        latexReport += '''\\section{For a data set of %d random samples.}''' % (n)
        print "Performing test with clustered data for %d random samples"% (n)
        for i in xrange(2): #TODO: Parametrizar los centroides en el dataset
            print "For %d clusters in the data set"%(i+2)
            latexReport += '''\\subsubsection{For %d clusters in the data set.}''' % (i+2)
            dataGenerator = DataGenerator()
            j = 0
            steps = (maxBeta-minBeta)/stepCBeta
            beta = np.zeros(steps)
            rperf = np.zeros(steps)
            rtT = np.zeros(steps)
            kperf = np.zeros(steps)
            ktT = np.zeros(steps)
            for b in xrange(steps): # Test the network for an incremental number of centroids
                print "Testing RBFNN looking for %d centroids"%(b)
                rrbfnn = RBFNN(dim, b, 1, "random")
                krbfnn = RBFNN(dim, b, 1, "knn")
                #Training and verifying results by randomly chosen centroids
                for k in xrange(meanSamples):
                    dataGenerator.generateClusteredRandomData(n, i+2, dim)
                    t = time.clock()
                    rrbfnn.train(dataGenerator.getTrainingX(), dataGenerator.getTrainingY())
                    rtT[j] += time.clock()-t
                    rperf[j] += dataGenerator.verifyResult(rrbfnn.test(dataGenerator.getValidationX()))
                    #Training and verifying results by k-means clustering
                    t = time.clock()
                    krbfnn.train(dataGenerator.getTrainingX(), dataGenerator.getTrainingY())
                    ktT[j] += time.clock()-t
                    kperf[j] += dataGenerator.verifyResult(krbfnn.test(dataGenerator.getValidationX()))
                    #Training and verifying results with metaplasticity modified weights
                rtT[j] /= meanSamples
                rperf[j] /= meanSamples
                ktT[j] /= meanSamples
                kperf[j] /= meanSamples
                beta[j] = b
                j+=1
            
            #Writing table of results
            latexReport += '''\\onecolumn\\begin{center}\\begin{tabular}{|c|c|c|c|c|c|c|}
                        \\hline
                        \\multicolumn{7}{|c|}{RBFNNs performance over %d samples} \\\\
                        \\hline
                        \\multirow{2}{*}{beta} & \multicolumn{2}{|c|}{Random centroids} & \multicolumn{2}{|c|}{Knn} & \multicolumn{2}{|c|}{Metaplasticity} \\\\
                        & Time & Accuracy & Time & Accuracy & Time & Accuracy \\\\
                        \\hline''' % (n)
            for j in range(len(beta)):
                latexReport += '''%d & %s & %s & %s & %s & %s & %s \\\\ \\hline''' % (beta[j], str(rtT[j]), str(rperf[j]), str(ktT[j]), str(kperf[j]), "na", "na")
            latexReport += '''\\end{tabular}\\end{center}\\twocolumn\n'''
                
            #Plotting accuracy over nCentroids
            #TODO: Tengo que hacer un multiplot. quedara mejor
            plt.clf()
            plt.plot(beta, rperf, "-b")
            plt.plot(beta, kperf, "-r")
            plt.title("Accuracy of prediction over the number of centroids to be looked for")
            plt.xlabel("Number of centroids")
            plt.ylabel("Accuracy (%)")
            plt.ylim(0,100)
            plt.legend(["Randomed", "K-means"])
            plt.savefig("plots/Accuracy%dc%ds.png" % (b, n))
            latexReport += '''\n\\begin{figure}[]{}
                            \\centering
                            \\includegraphics[width=0.4\\textwidth]{plots/Accuracy%dc%ds.png}
                            \\label{fig:clusteredData1}
                            \\end{figure}\n''' % (b, n)
            #Plotting time over nCentroids
            plt.clf()
            plt.plot(beta, rtT, "-b")
            plt.plot(beta, ktT, "-r")
            plt.title("Time consumed in training over the number of centroids to be looked for")
            plt.xlabel("Number of centroids")
            plt.ylabel("Time (s)")
            plt.legend(["Randomed", "K-means"])
            plt.savefig("plots/Time%dc%ds.png" % (b, n))
            latexReport += '''\n\\begin{figure}[]{}
                            \\centering
                            \\includegraphics[width=0.4\\textwidth]{plots/Time%dc%ds.png}
                            \\label{fig:clusteredData1}
                            \\end{figure}\n''' % (b, n)
    return latexReport

if __name__ == '__main__':
    print 'Performing test over the RBFNNs'
    latexReport = Strings.headerReportClusterTemplate #Incluyo la cabecera tex
    
    latexReport += ClusteredDataTest(minBeta = 2,  # Minimun number of neurons in hidden layer
                                maxBeta = 8,       # Maximun number of neurons in hidden layer
                                stepCBeta = 2,      # Step in hidden layer growing
                                dim = 20,               # Data set dimension and NNs indim
                                meanSamples=1)          # Number of samples for meaning each result
    
#     latexReport += RealDataTest(dataSet='cancer',       #Data set to test 
#                                 minBeta = 2,     #Min number of centroids (neurons in hiddent layer
#                                 maxBeta = 200,    #Max centroids
#                                 stepCBeta = 5,
#                                 meanSamples=5)     
# 
#     latexReport += RealDataTest(dataSet='column',       #Data set to test 
#                                 minBeta = 2,     #Min number of centroids (neurons in hiddent layer
#                                 maxBeta = 200,    #Max centroids
#                                 stepCBeta = 5,
#                                 meanSamples=5)     
    
    latexReport += '''\\end{document}'''
    with open('report.tex','w') as f:
        f.write(latexReport.encode('utf8')) #Escribo el archivo .tex
    fname='report'
    proc=subprocess.Popen(shlex.split('pdflatex %s.tex'%(fname))) #Genero el pdf del informe
    proc.communicate()
    proc=subprocess.Popen(shlex.split('rm %s.aux %s.idx %s.log %s.tex %s.toc'%(fname,fname,fname,fname,fname)))
    proc.communicate()
    