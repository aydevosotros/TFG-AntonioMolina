# -*- encoding: utf-8 -*-
'''
Created on Nov 12, 2014

@author: antonio
'''


class LatexReport(object):
    '''
    Clase representa un informe
    '''
    def __init__(self):
        pass
    
    def setTitle(self):
        pass
    
    def addSection(self):
        pass    
    
    def createPDF(self):
        pass
        
        
class LatexGenerator(object):  
      
    @staticmethod
    def generatePlot(self):
        return "hola k ase"
    
    @staticmethod
    def generateTable(self):
        pass
    

# Some code to debug
if __name__ == '__main__':
    print LatexGenerator.generatePlot()