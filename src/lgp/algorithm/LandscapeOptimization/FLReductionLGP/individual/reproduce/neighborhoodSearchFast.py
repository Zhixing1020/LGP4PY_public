import sys
import math

from src.ec import *
from src.lgp.individual.lgp_individual import LGPIndividual
from src.lgp.individual.gp_tree_struct import GPTreeStruct
from src.lgp.individual.primitive import *

from src.lgp.algorithm.LandscapeOptimization.reproduce.neighborhoodSearch import NeighborhoodSearch
from src.lgp.algorithm.LandscapeOptimization.subpopulationFLO import SubpopulationFLO
from src.lgp.algorithm.LandscapeOptimization.simpleLGP.indexing.indexList4LGP import IndexList4LGP
from src.lgp.algorithm.LandscapeOptimization.indexing.direction import Direction
from src.lgp.algorithm.LandscapeOptimization.indexing.genoVector import GenoVector

from src.lgp.algorithm.LandscapeOptimization.FLReductionLGP.indexing.indexList4LGP_FLR import IndexList4LGP_FLR

class NeighborhoodSearchFast(NeighborhoodSearch):

    def produce_individual(self, min_val:int, max_val:int, start:int, subpopulation:int, 
                inds:list[LGPIndividual], state:EvolutionState, thread:int, parents:list[LGPIndividual])->int:
        # The loop for numTries is commented out in original Java, so we proceed directly
        inds[start] = self.all_in_one_produce(subpopulation, inds, state, thread, parents)
        return self.INDS_PRODUCED

    def all_in_one_produce(self, subpopulation:int, inds:list[LGPIndividual], state:EvolutionState, thread:int, parents:list[LGPIndividual])->LGPIndividual:
        # 1. Setup Data Structures
        subpop_obj:SubpopulationFLO = state.population.subpops[subpopulation]
        board = subpop_obj.fullBoard
        indexlist:IndexList4LGP_FLR = subpop_obj.IndList  # cast to IndexList4LGP implied

        # 2. Update LeadBoard (Lazy Initialization / Generation Check)
        if self.leadBoard is None or self.cur_generation != state.generation:
            self.leadBoard = board.lightTrimCloneByBest()
            # Commented out calls from original preserved as comments
            # self.leadBoard.trim2MaxsizeByBest()
            # self.leadBoard.resetGenoAngle(indexlist, state.generation)
            # self.leadBoard.resetGenoDifference(indexlist, state.generation)
            
            self.cur_generation = state.generation

        self.master_i = 0
        parent = parents[self.master_i] # Assumed to be LGPIndividual
        progLength = parent.getTreesLength()

        # 3. Calculate Mutation Direction
        # Using a helper class to mimic Java's Direction object behavior
        direction = Direction()
        
        # Calculate mask size (how many instructions to mutate)
        # min(maxMaskSize, progLength) prevents out of bounds
        limit_mask = min(self.maxMaskSize, progLength)
        if limit_mask > 0:
            mask_size = 1 + state.random[thread].randint(0, limit_mask-1)
        else:
            mask_size = 1

        # Initialize direction vector
        for d in range(parent.getMaxNumTrees()):
            if d < progLength:
                direction.append(-1.0) # Placeholder for reduction/modification
            else:
                direction.append(0.0)

        # Filter direction vector to match mask_size
        cntNonZero = 0
        for i in range(len(direction)):
            if direction[i] != 0:
                cntNonZero += 1
        
        index = state.random[thread].randint(0, len( direction )-1)
        cn = cntNonZero
        
        # Randomly zero out entries until we match mask_size
        cnt = 0
        while cn > mask_size and cnt < len( direction ):
            if direction[index] != 0:
                direction[index] = 0.0
                cn -= 1
            index = (index + 1) % len( direction )
            cnt += 1

        # 4. Determine Macro Mutation (Size Direction)
        rate = state.random[thread].uniform(0,1)
        
        # Add Rate check
        if rate <= self.addRate or progLength == parent.getMinNumTrees():
            limit_macro = min(self.maxMacroSize, cntNonZero) + 1
            direction.sizeDirection = state.random[thread].randint(0, limit_macro-1)
            
        # Remove Rate check
        elif rate <= (self.addRate + self.removeRate) or progLength == parent.getMaxNumTrees():
            limit_macro = min(self.maxMacroSize, cntNonZero) + 1
            direction.sizeDirection = -1 * state.random[thread].randint(0, limit_macro-1)

        # 5. Balance Micro vs Macro mutation
        # If we are adding/removing too many, reduce the number of instruction modifications (mask_size)
        if mask_size > abs(direction.sizeDirection):
            mask_size -= abs(direction.sizeDirection)
        elif abs(direction.sizeDirection) > mask_size:
            if direction.sizeDirection > 0:
                direction.sizeDirection -= mask_size
            # else: direction.sizeDirection < 0, we don't care
        else: 
            # If sizes are equal, randomly zero out one of them
            if state.random[thread].uniform(0,1) < 0.5:
                mask_size = 0
            else:
                direction.sizeDirection = 0

        # 6. Apply Mutations
        checkList = []
        newind = parent.clone()

        # 6a. Micro Mutation (Modify existing instructions)
        for i in range(newind.getTreesLength()):
            if direction[i] != 0:
                instr:GPTreeStruct = newind.getTree(i)
                symIndex = indexlist.getIndexBySymbol(instr)
                
                # Move symbol in genotype space
                # Assuming GenoVector class has static method moveSymbol
                newsym = GenoVector.moveSymbol(state, thread, symIndex, direction[i] * symIndex, self.maxStep)
                
                if newsym < 0 or newsym >= len(indexlist):
                    newsym = state.random[thread].randint(0, len(indexlist)-1)
                
                # Retrieve new instruction (IndexList4LGP_FLR specific method)
                newInstr:GPTreeStruct = indexlist.getRandSymbolByIndex(newsym, state, thread, subpopulation)
                
                newind.setTree(i, newInstr)
                checkList.append(newInstr)

        newind.updateStatus()

        # 6b. Macro Mutation (Add/Remove instructions)
        # Adding
        while direction.sizeDirection > 0 and newind.getTreesLength() < newind.getMaxNumTrees():
            newsym = state.random[thread].randint(0, len(indexlist)-1)
            newInstr = indexlist.getSymbolByIndex(newsym, state, thread)
            
            newindex = state.random[thread].randint(0, newind.getTreesLength()-1)
            newind.addTree(newindex, newInstr)
            checkList.append(newInstr)
            direction.sizeDirection -= 1
            
        # Removing
        while direction.sizeDirection < 0 and newind.getTreesLength() > newind.getMinNumTrees():
            newindex = state.random[thread].randint(0, newind.getTreesLength()-1)
            newind.removeTree(newindex)
            direction.sizeDirection += 1

        # 7. Validation and Repair
        # Check constraints (min/max length, effective length)
        if (newind.getTreesLength() < newind.getMinNumTrees() or 
            newind.getTreesLength() > newind.getMaxNumTrees() or 
            newind.getEffTreesLength() < 1):
            newind.rebuildIndividual(state, thread)

        # 8. Effectiveness Optimization
        # Try to fix a random mutated instruction if it is ineffective (intron)
        if len(checkList) > 0:
            instr = checkList[state.random[thread].randint(0, len(checkList)-1)]
            if not instr.status:
                # Try to redirect output to an effective register
                eff_regs_list = list(instr.effRegisters)
                if len(eff_regs_list) > 0:
                    reg_idx = state.random[thread].choice(eff_regs_list)
                    # Assuming child is WriteRegisterGPNode
                    if isinstance(instr.child, WriteRegisterGPNode):
                        instr.child.setIndex(reg_idx)
                
                newind.updateStatus()

        return newind

    def maintainPhenotype(self, state:EvolutionState, thread:int, oldind:LGPIndividual, newind:LGPIndividual, newgv:GenoVector):
        print("NeighborhoodSearchFast does not implement maintainPhenotype() method", file=sys.stderr)
        sys.exit(1)
        return False
