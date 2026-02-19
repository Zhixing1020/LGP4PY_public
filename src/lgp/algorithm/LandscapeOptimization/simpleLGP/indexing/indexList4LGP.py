from abc import abstractmethod
import math
import random
import sys
import numpy as np

from src.ec import *
from src.lgp.individual.lgp_individual import LGPIndividual
from src.lgp.algorithm.LandscapeOptimization.indexing.indexList import IndexList
from src.lgp.algorithm.LandscapeOptimization.indexing.board import Board
from src.lgp.algorithm.LandscapeOptimization.indexing.genoVector import GenoVector

from src.lgp.individual.gp_tree_struct import GPTreeStruct

class IndexList4LGP(IndexList[GPTreeStruct]):
    
    @abstractmethod
    def initialize(self, state:EvolutionState, thread:int):
        pass

    @abstractmethod
    def optimizeIndex(self, state, thread, subpop, board):
        # Update the indexes based on the new leading board
        pass

    def getGenoVector(self, ind:GPIndividual):
        if not isinstance(ind, LGPIndividual):
            print("Error: Non-LGP individual in LGP Index List.", file=sys.stderr)
            sys.exit(1)
            
        geno:GenoVector = GenoVector(ind.getMaxNumTrees(), ind.getMinNumTrees())
        
        for i in range(len(geno.G)):
            if i < ind.getTreesLength():
                idx = self.getIndexBySymbol(ind.getTree(i))
                if idx == GenoVector.NoneVal: # Assuming None is a constant in GenoVector
                    print(f"Unknown LGP instruction at index {i}, Please check whether the builders in genetic operators and the index list are the same.", file=sys.stderr)
                    sys.exit(1)
                geno.G[i] = idx
            else:
                geno.G[i] = GenoVector.NoneVal
        return geno

    def cloneIndexList(self):
        new_list = IndexList4LGP()
        # Ensure deep copy of index items
        for item in self:
            new_list.append(item.clone())
        return new_list

    def evaluateObjectives(self, list_obj:IndexList, board:Board):
        fit = 0.0
        for ob in range(self.numobjectives):
            fit += self.coefficiency[ob] * self.objectives[ob].evaluate(list_obj, board)
        return fit