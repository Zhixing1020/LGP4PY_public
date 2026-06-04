from src.ec.util import *
from src.ec import *
# from copy import deepcopy

class ClassErrorFitness(Fitness):

    def isIdealFitness(self):
        return self.value == 0