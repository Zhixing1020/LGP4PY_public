
from typing import override

from src.ec import *
from src.lgp.algorithm.LandscapeOptimization.indexing.direction import Direction
from src.lgp.algorithm.LandscapeOptimization.indexing.genoVector import GenoVector
from src.lgp.algorithm.typed_lgp.FLReductionLGP.indexing.indexList4TypedLGP_FLR import IndexList4TypedLGP_FLR
from src.lgp.algorithm.typed_lgp.individual.typed_gp_tree_struct import TypedGPTreeStruct
from src.lgp.individual.gp_tree_struct import GPTreeStruct
from src.lgp.individual.lgp_individual import LGPIndividual

from src.lgp.algorithm.LandscapeOptimization.FLReductionLGP.indexing.indexList4LGP_FLR import IndexList4LGP_FLR
from src.lgp.algorithm.LandscapeOptimization.FLReductionLGP.individual.reproduce.neighborhoodSearchFast import NeighborhoodSearchFast
from src.lgp.algorithm.LandscapeOptimization.subpopulationFLO import SubpopulationFLO


class NeighborhoodSearchFastTyped(NeighborhoodSearchFast):
    
    @override
    def _apply_micro_mutation(self, newind:LGPIndividual, direction:Direction, indexlist:IndexList4TypedLGP_FLR, state:EvolutionState, thread:int):
        checkList = []

        for i in range(newind.getTreesLength()):
            if direction[i] != 0:
                instr:TypedGPTreeStruct = newind.getTree(i)
                targetType = state.random[thread].choice(list(instr.effTypes)) if len(instr.effTypes) > 0 else None
                sourceType = instr.effInputTypes
                symIndex = indexlist.getIndexBySymbol(instr)

                newsym = GenoVector.moveSymbol(state, thread, symIndex, direction[i] * symIndex, self.maxStep)
                if newsym < 0 or newsym >= len(indexlist):
                    newsym = state.random[thread].randint(0, len(indexlist)-1)

                newInstr:GPTreeStruct = indexlist.getSymbolByIndex(newsym, state, thread, targetType, sourceType)
                newind.setTree(i, newInstr, update_status=False)
                checkList.append(newInstr)

        newind.updateStatus()
        return checkList

    @override
    def _apply_macro_mutation(self, newind:LGPIndividual, direction:Direction, indexlist:IndexList4LGP_FLR, state:EvolutionState, thread:int, checkList:list):
        while direction.sizeDirection > 0 and newind.getTreesLength() < newind.getMaxNumTrees():
            newindex = state.random[thread].randint(0, newind.getTreesLength()-1)
            targetType = state.random[thread].choice(list(newind.getTree(newindex).effTypes)) if len(newind.getTree(newindex).effTypes) > 0 else None
            sourceType = newind.getTree(newindex).effInputTypes

            newsym = state.random[thread].randint(0, len(indexlist)-1)
            newInstr = indexlist.getSymbolByIndex(newsym, state, thread, targetType, sourceType)
            
            newind.addTree(newindex, newInstr)
            checkList.append(newInstr)
            direction.sizeDirection -= 1

        while direction.sizeDirection < 0 and newind.getTreesLength() > newind.getMinNumTrees():
            newindex = state.random[thread].randint(0, newind.getTreesLength()-1)
            newind.removeTree(newindex)
            direction.sizeDirection += 1