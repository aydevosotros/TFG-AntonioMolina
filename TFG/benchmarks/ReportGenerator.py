# -*- encoding: utf-8 -*-
'''
Created on Nov 12, 2014

@author: antonio
'''

import os, shlex, subprocess

class LatexReport(object):
    '''
    Clase representa un informe
    '''
    def __init__(self, title, author):
        try:
            os.mkdir("plots/") #Creo, si no esta, la carpeta en la que guardar las figuras
        except:
            pass
        self.latexReport = '''
        \\documentclass[10pt,a4paper]{report}
        \\usepackage[utf8]{inputenc} 
        \\usepackage[spanish]{babel}
        \\usepackage{amsmath}
        \\usepackage{amsfonts}
        \\usepackage{amssymb}
        \\usepackage{graphicx}
        \\usepackage{multirow}
        \\author{%s}
        \\title{%s}
        \\makeindex
        \\begin{document}
        \\maketitle
        \\pagebreak
        \\tableofcontents
        \\pagebreak
        '''%(author, title)
        
        
        
    def addSection(self, sectionTitle):
        self.latexReport += '''\\section{%s}'''%(sectionTitle)
    
    def addContent(self, content):
        self.latexReport+=content;
    
    def createPDF(self, fname="report"):
        self.latexReport += '''\\end{document}'''
        with open('%s.tex'%(fname),'w') as f:
            f.write(self.latexReport.encode('utf8')) #Escribo el archivo .tex
        proc=subprocess.Popen(shlex.split('pdflatex %s.tex'%(fname))) #Genero el pdf del informe
        proc.communicate()
        proc=subprocess.Popen(shlex.split('rm %s.aux %s.idx %s.log %s.tex %s.toc'%(fname,fname,fname,fname,fname)))
        proc=subprocess.Popen(shlex.split('evince %s.pdf'%(fname)))
        proc.communicate()
        
        
class LatexGenerator(object):  
      
    @staticmethod
    def generatePlot(self):
        return "hola k ase"
    
    @staticmethod
    def generateTable(data, titles):
        lTable = "\\begin{center}\\begin{tabular}{"
        for i in xrange(len(data[0])):
            lTable += "|c"
        lTable += "|}\\hline"
        nRowTitle = len(data[0])/len(titles)
        for i in xrange(len(titles)):
            if i == 0:
                lTable += "\\multicolumn{%d}{|c|}{%s}"%(nRowTitle, titles[i])
            else:
                lTable += "& \\multicolumn{%d}{|c|}{%s}"%(nRowTitle, titles[i])
        lTable += "\\\\ \\hline "
        for i, row in enumerate(data):
            for j, cell in enumerate(row):
                if j == 0:
                    lTable += "%s "%(cell)
                else:
                    lTable += "& %s"%(cell)
            lTable += "\\\\ \\hline "
        lTable += "\\end{tabular}\\end{center}"
        return lTable

# Some code to debug
if __name__ == '__main__':
    print LatexGenerator.generateTable([[1,2,3,4], [1,2,3,4]], ["titulo1", "titulo2"])