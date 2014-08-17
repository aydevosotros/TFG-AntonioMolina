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

def ClusteredDataTest():
    try:
        os.mkdir("plots/")
    except:
        pass
    latexReport = Strings.headerReportClusterTemplate
    for n in range(500, 1000, 100):
        latexReport += '''\\section{For a data set of %d samples.}''' % (n)
        for i in range(3):
            latexReport += '''\\subsubsection{For %d clusters in the data set.}''' % (i+2)
            cdt = ClusteredDataGenerator(n, i+2, 2)
            cdt.savePlot()
            latexReport += '''\\begin{figure}[!h]{}
                                \\centering
                                \\includegraphics[width=0.4\\textwidth]{plots/%dc%ds.png}
                                \\label{fig:clusteredData1}
                            \\end{figure}''' % (i+2, n)
            nc = 50
            nC = np.zeros(10)
            perf = np.zeros(10)
            tT = np.zeros(10)
            for j in range(10): # Test the network for an incremental number of centroids
                rbfnn = RBFNN(2, nc, 1)
                t = time.clock()
                rbfnn.train(cdt.getTrainingX(), cdt.getTrainingY())
                tT[j] = time.clock()-t
                perf[j] = cdt.verifyResult(rbfnn.test(cdt.getValidationX()))
                nC[j] = nc
                nc += 50
            
            #Writing table of results
            latexReport += '''\\begin{tabular}{|l | c | r|} \\hline Centroids & Performance & Training time \\\\ \\hline'''
            for j in range(len(nC)):
                latexReport += '''%d & %s & %s\\\\ \\hline''' % (nC[j], str(perf[j]), str(tT[j]))
            latexReport += '''\\end{tabular}'''
                
            #Plotting accuracy over nCentroids
            plt.clf()
            plt.plot(nC, perf, "-b")
            #Plotting time over nCentroids
            plt.plot(nC, tT, "-r")
            plt.savefig("plots/Time%dc%ds.png" % (nc, n))
            latexReport += '''\\begin{figure}[!h]{}
                            \\centering
                            \\includegraphics[width=0.4\\textwidth]{plots/Time%dc%ds.png}
                            \\label{fig:clusteredData1}
                            \\end{figure}''' % (nc, n)
                    
    latexReport += '''\\end{document}'''
    with open('report.tex','w') as f:
        f.write(latexReport.encode('utf8'))
    proc=subprocess.Popen(shlex.split('pdflatex report.tex'))
    proc.communicate()


if __name__ == '__main__':
    print 'Performing test over the RBFNNs'
    ClusteredDataTest()
    