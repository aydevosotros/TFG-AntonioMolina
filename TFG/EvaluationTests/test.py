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
import os

def generatePdfReport(nCentroids, perf, tTime):
    #TODO: Create the script for creating report
    pass

def ClusteredDataTest():
    os.mkdir("plots/")
    latexReport = '''
    \\documentclass[10pt,a4paper, twocolumn]{report}
    \\usepackage[T1]{fontenc} 
    \\usepackage[utf8]{inputenc} 
    \\usepackage[spanish]{babel}
    \\usepackage{amsmath}
    \\usepackage{amsfonts}
    \\usepackage{amssymb}
    \\usepackage{graphicx}
    \\author{Antonio Molina Garca-Retamero}
    \\title{Results report}
    \\makeindex
    \\begin{document}
    \\maketitle
    \\pagebreak
    \\tableofcontents
    \\pagebreak
    \\section{Results for classification in clustered data set}
    In this section we'll perform a test battery in order to determine how well works the neural network by classifying data in a clustered data set.
    For each test, we test the accuracy of the prediction for a determine data set clustered in k centroids and the time elapsed in training phase. We will increase the number of neurons in first layer for each test
    '''
    for i in range(3):
        latexReport += '''\\subsection{For %d clusters and %d samples}''' % (i+2, 1000)
        cdt = ClusteredDataGenerator(1000, i+2, 2)
        cdt.savePlot()
        latexReport += '''\\subsubsection{Two first dimensions one over other} \\begin{figure}[!h]{}
                            \\centering
                            \\includegraphics[width=0.4\\textwidth]{plots/%dc%ds.png}
                            \\label{fig:clusteredData1}
                        \\end{figure}''' % (i+2, 1000)
        nc = 50
        nC = np.zeros(10)
        perf = np.zeros(10)
        tT = np.zeros(10)
        for j in range(10):
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
        plt.plot(nC, perf, "-r")
        plt.savefig("plots/Accuracy%dc%ds.png" % (nc, 1000))
        latexReport += '''\\subsubsection{Accuracy over number of centroids} \\begin{figure}[!h]{}
                        \\centering
                        \\includegraphics[width=0.4\\textwidth]{plots/Accuracy%dc%ds.png}
                        \\label{fig:clusteredData1}
                    \\end{figure}''' % (nc, 1000)
        #Plotting time over nCentroids
        plt.clf()
        plt.plot(nC, tT, "-r")
        plt.savefig("plots/Time%dc%ds.png" % (nc, 1000))
        latexReport += '''\\subsubsection{Time required for training over number of centroids} \\begin{figure}[!h]{}
                        \\centering
                        \\includegraphics[width=0.4\\textwidth]{plots/Time%dc%ds.png}
                        \\label{fig:clusteredData1}
                    \\end{figure}''' % (nc, 1000)
    latexReport += '''\\end{document}'''
    with open('report.tex','w') as f:
        f.write(latexReport.encode('utf8'))
    proc=subprocess.Popen(shlex.split('pdflatex report.tex'))
    proc.communicate()


if __name__ == '__main__':
    print 'Performing test over the RBFNNs'
    ClusteredDataTest()
    