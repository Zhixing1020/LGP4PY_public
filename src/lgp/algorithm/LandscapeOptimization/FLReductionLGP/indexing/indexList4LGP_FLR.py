import math
import sys
# import copy
# import random
# from typing import List, Optional, Any, Set

import numpy as np
from src.ec import *
from src.ec.util import *
from src.lgp.individual.lgp_individual import LGPIndividual
from src.lgp.individual.gp_tree_struct import GPTreeStruct

from src.lgp.algorithm.LandscapeOptimization.simpleLGP.indexing.indexList4LGP import IndexList4LGP
from src.lgp.algorithm.LandscapeOptimization.indexing.board import Board
from src.lgp.algorithm.LandscapeOptimization.indexing.index import Index
from src.lgp.algorithm.LandscapeOptimization.indexing.indexList import IndexList
from src.lgp.algorithm.LandscapeOptimization.indexing.genoVector import GenoVector
from src.lgp.algorithm.LandscapeOptimization.objectives.objective4FLO import Objective4FLO
from src.lgp.algorithm.LandscapeOptimization.subpopulationFLO import SubpopulationFLO
from src.lgp.individual.primitive import *

class IndexList4LGP_FLR(IndexList4LGP):
    """
    The index list for fitness landscape reduction.
    This list has a maximum size which is much smaller than the total number of symbols.
    Different from the fix-size list, FLR list adds symbols from the population.
    FLR list removes less effective symbols when its size exceeds the limitation.
    """

    # Static Constants
    INDEXLIST_FLR = "IndexList_FLR"
    MAXLISTSIZE = "maxListSize"
    P_NODESELECTOR = "nodeSelector"
    P_UPDATERATE = "random_update_rate"
    
    # Defaults
    MAXNUMTRIES = 10
    rand_update_rate = 0.2

    def __init__(self):
        super().__init__()
        self.maxlistsize = 1000
        
        # State/Thread cache for consistency with interfaces that might lack them
        self.self_state = None
        self.self_thread = None
        
        self.nodeselect:GPNodeSelector = None # GPNodeSelector
        self.builder:GPBuilder = None    # GPNodeBuilder
        
        # FLR specific fields
        self.numobjectives = 0
        self.objectives = []
        self.coefficiency = []
        self.boardsize = []
        self.numiterations = 1
        self.step_rate = 0.0
        self.min_step = 1.0
        self.batchsize = 1

    def setup(self, state:EvolutionState, base:Parameter):
        # super().setup(state, base) # Java commented this out
        
        def_param = Parameter(self.INDEXLIST_FLR)
        
        self.maxlistsize = state.parameters.getIntWithDefault(
            base.push(self.MAXLISTSIZE), def_param.push(self.MAXLISTSIZE), 1000
        )
        if self.maxlistsize <= 0:
            state.output.fatal("the maximum size of index list in fitness landscape reduction must be at least 1",
                               base.push(self.MAXLISTSIZE), def_param.push(self.MAXLISTSIZE))
            
        self.nodeselect = state.parameters.getInstanceForParameter(
            base.push(self.P_NODESELECTOR), def_param.push(self.P_NODESELECTOR), GPNodeSelector
        )
        self.nodeselect.setup(state, base.push(self.P_NODESELECTOR))

        # Assuming BUILDER is a constant in the parent class or defined globally
        self.builder = state.parameters.getInstanceForParameter(
            base.push(self.BUILDER), def_param.push(self.BUILDER), GPBuilder
        )
        self.builder.setup(state, base.push(self.BUILDER))
        
        self.rand_update_rate = state.parameters.getDoubleWithDefault(
            base.push(self.P_UPDATERATE), def_param.push(self.P_UPDATERATE), 0.2
        )
        if not (0 <= self.rand_update_rate <= 1):
            state.output.fatal("the random updating rate in fitness landscape reduction must be [0,1]",
                               base.push(self.P_UPDATERATE), def_param.push(self.P_UPDATERATE))
            
        self.parent_setup(state, base)

    def parent_setup(self, state:EvolutionState, base:Parameter):
        # Replicates the logic used in Java to load common IndexList parameters
        def_param = Parameter(self.INDEXLIST)
        
        self.prototype = state.parameters.getInstanceForParameter(
            base.push(self.ITEMPROTOTYPE), def_param.push(self.ITEMPROTOTYPE), Index[GPTreeStruct]
        )
        self.prototype.setup(state, base.push(self.ITEMPROTOTYPE))

        
        self.numobjectives = state.parameters.getInt(
            base.push(self.NUMOBJECTIVES), def_param.push(self.NUMOBJECTIVES)
        )
        if self.numobjectives <= 0:
            state.output.fatal("the number of objectives must be at least 1")
            
        self.objectives = [None] * self.numobjectives
        self.coefficiency = [0.0] * self.numobjectives
        self.boardsize = [0] * self.numobjectives
        
        for obj in range(self.numobjectives):
            obj_base = base.push(self.OBJECTIVES).push(str(obj))
            def_obj_base = def_param.push(self.OBJECTIVES).push(str(obj))
            
            self.objectives[obj] = state.parameters.getInstanceForParameter(
                obj_base, def_obj_base, Objective4FLO
            )
            
            self.coefficiency[obj] = state.parameters.getDoubleWithDefault(
                obj_base.push(self.P_COEFFICIENCY), def_obj_base.push(self.P_COEFFICIENCY), 1.0
            )
            if self.coefficiency[obj] <= 0:
                state.output.fatal(f"coefficiency for objective {obj} must be > 0")
                
            self.boardsize[obj] = state.parameters.getIntWithDefault(
                obj_base.push(self.P_BOARDSIZE), def_obj_base.push(self.P_BOARDSIZE), 10
            )
            if self.boardsize[obj] <= 0:
                state.output.fatal(f"boardsize for objective {obj} must be > 0")
                
        self.numiterations = state.parameters.getInt(
            base.push(self.NUMITERATIONS), def_param.push(self.NUMITERATIONS)
        )
        if self.numiterations <= 0:
            state.output.fatal("numiterations must be at least 1")
            
        self.step_rate = state.parameters.getDoubleWithDefault(base.push(self.P_STEP), def_param.push(self.P_STEP), 0.1)
        if not (0 < self.step_rate <= 1):
             state.output.fatal("step must be in (0, 1]")
             
        self.min_step = state.parameters.getDoubleWithDefault(
            base.push(self.P_MINSTEP), def_param.push(self.P_MINSTEP), 1.0
        )
        if self.min_step <= 0:
            state.output.fatal("min_step must be > 0")
            
        self.batchsize = state.parameters.getIntWithDefault(
            base.push(self.P_BATCHSIZE), def_param.push(self.P_BATCHSIZE), 1
        )
        if self.batchsize <= 0:
            state.output.fatal("batchsize must be > 0")
            
        self.initialize(state, 0)

    def initialize(self, state:EvolutionState, thread:int):
        # initializer = state.initializer
        # type_ = self.prototype.sym_prototype.constraints(initializer).treetype
        
        for i in range(self.maxlistsize):
            # randomly generate a tree
            ins:GPTreeStruct = self.prototype.sym_prototype.lightClone()
            ins.owner = None
            ins.status = False
            ins.effRegisters = set()
            
            ins.buildTree(state, thread)
            
            # check duplication
            dup = False
            name = str(ins)
            for nd in self:
                # if nd.isduplicated(ins):
                #     dup = True
                #     # exist = False
                #     # for sym in nd.symbols:
                #     #     if str(sym) == str(ins):
                #     #         exist = True
                #     #         break
                #     exist = str(ins) in nd.symbol_names
                #     if not exist:
                #         nd.addSymbol(ins)
                #     break
                if name in nd.symbol_names:
                    dup = True
                    break
            
            if not dup:
                nd:Index[GPTreeStruct] = self.prototype.clone()
                nd.index = len(self)
                nd.addSymbol(ins)
                self.append(nd) # In Python list 'add' is 'append'

        # Shuffle Indices logic
        to_be_shuffle_index = [item.index for item in self]
        IndexList.shuffleIndex(state, thread, to_be_shuffle_index)
        
        for i, item in enumerate(self):
            item.index = to_be_shuffle_index[i]
            
        self.self_state = state
        self.self_thread = thread
        
        self.basic_initialize(state, thread)

    def optimizeIndex(self, state:EvolutionState, thread:int, subpop:SubpopulationFLO, board:Board):
        # 1. Initialize search
        # add the symbols from the current population into the list
        self.updateSymbolsFromRandPop(state, thread, subpop)
        self.updateSymbolsFromObjs(state, thread, subpop, board)
        
        # Shuffle indices for diversity
        limit = int(self.rand_update_rate * len(self))
        for _ in range(limit):
            samplei = state.random[thread].randint(0, len(self)-1)
            samplej = state.random[thread].randint(0, len(self)-1)
            
            itemi:Index[GPTreeStruct] = self[samplei]
            itemj:Index[GPTreeStruct] = self[samplej]
            
            # Swap indices
            itemi.index, itemj.index = itemj.index, itemi.index

        # 2. Iteratively update indexes
        oldfit = 0.0
        newfit = 0.0
        newlist = self.cloneIndexList()
        
        priorityItem = [False] * len(self)
        
        # Determine priority items based on objectives
        for ob_idx in range(self.numobjectives):
            obj = self.objectives[ob_idx]
            tmp = obj.getUsedItem()

            arr_p = np.array(priorityItem)
            arr_t = np.array(tmp)
            priorityItem = (arr_p | arr_t).tolist()

        priorityItemList:list[int] = []
        unpriorityItemList:list[int] = []
        for i in range(len(newlist)):
            if priorityItem[i]:
                priorityItemList.append(i)
            else:
                unpriorityItemList.append(i)
        
        IndexList.shuffleIndex(state, thread, priorityItemList)
        IndexList.shuffleIndex(state, thread, unpriorityItemList)
        
        oldfit = self.evaluateObjectives(self, board)
        
        skip = False
        grad = [0.0] * len(self)
        toBreakCnt = 0
        
        for _ in range(self.numiterations):
            
            # 2.a Get negative gradient
            if not skip:
                grad = [0.0] * len(self)
                
                for ob_idx in range(self.numobjectives):
                    tmp = self.objectives[ob_idx].gradient(newlist, board)
                    for i in range(len(self)):
                        grad[i] += -1 * tmp[i] * self.coefficiency[ob_idx]
                
                # Normalize
                norm = sum(g*g for g in grad)
                norm = math.sqrt(norm) + 1e-9
                
                for i in range(len(self)):
                    grad[i] /= norm
                    val = abs(grad[i]) + 1e-9
                    sign = math.copysign(1, grad[i]) if grad[i] != 0 else 0
                    grad[i] = sign / (-math.log(val))
                            
            
            # 2.b Update indices
            if self.step_rate <= 0:
                sys.stderr.write("the step of index list is not initialized.\n")
                sys.exit(1)
                
            stepg = state.random[thread].uniform(0,1) * self.step_rate
            used = [False] * len(self)
            
            # Helper function for logic reuse
            def update_items(item_list):
                for i in item_list:
                    g_val = grad[i]
                    sign_g = math.copysign(1, g_val) if g_val != 0 else 0
                    
                    current_idx_val = newlist[i].index
                    
                    newind = int(math.floor(current_idx_val + sign_g * self.min_step + stepg * g_val))
                    
                    # Ranges
                    if newind < 0: newind = 0
                    if newind >= len(self): newind = len(self) - 1
                    
                    # Uniques
                    if used[newind]:
                        trialind = newind
                        # Searching spiral/alternating
                        for j in range(1, 2 * len(self)):
                            offset = math.pow(-1, j) * math.ceil(j / 2.0)
                            trialind = int(newind + offset)
                            
                            if trialind < 0 or trialind >= len(self) or used[trialind]:
                                continue
                            
                            newind = trialind
                            break
                    
                    if used[newind]:
                        sys.stderr.write("Indexlist cannot find unique unused index. What happens!!?\n")
                        sys.exit(1)
                        
                    newlist[i].index = newind
                    used[newind] = True

            # Update priority then unpriority
            update_items(priorityItemList)
            update_items(unpriorityItemList)
            
            newfit = self.evaluateObjectives(newlist, board)
            
            if newfit <= oldfit:
                # Accept change
                for i in range(len(self)):
                    self[i].index = newlist[i].index
                oldfit = newfit
                skip = False
                toBreakCnt = 0
            else:
                # Revert change
                for i in range(len(self)):
                    newlist[i].index = self[i].index
                skip = True
                toBreakCnt += 1
                if toBreakCnt == 5:
                    break
                
                IndexList.shuffleIndex(state, thread, priorityItemList)
                IndexList.shuffleIndex(state, thread, unpriorityItemList)

        self.trimList(state, thread)

    def getGenoVector(self, ind:LGPIndividual) -> GenoVector: # Returns GenoVector
        # Assuming LGPIndividual is imported or available
        if not isinstance(ind, LGPIndividual):
            sys.stderr.write("non-LGP individual in index list for LGP, quit\n")
            sys.exit(1)
        
        geno = GenoVector(ind.getMaxNumTrees(), ind.getMinNumTrees())
        
        for i in range(len(geno.G)):
            if i < ind.getTreesLength():
                # Note: Java calls addSymbol which modifies the list!
                geno.G[i] = self.addSymbol(ind.getTree(i), self.self_state, self.self_thread)
                if geno.G[i] < 0:
                    sys.stderr.write("uncompatible LGP instruction. Please check whether the builders in genetic operators and the index list are the same\n")
                    sys.exit(i)
            else:
                geno.G[i] = -1 # GenoVector.None
                
        return geno

    def getSymbolByIndex(self, index:int, state:EvolutionState, thread:int) -> GPTreeStruct:
        i = 0
        for _ in range(self.MAXNUMTRIES):
            offset = math.pow(-1, i) * math.ceil(i / 2.0)
            next_index = int(index + offset)
            
            if next_index < 0 or next_index >= len(self):
                i += 1
                continue
                
            item = self.getItemByIndex(next_index)
            
            if state.random[thread].uniform(0,1) > item.get_tabu_frequency():
                # Random choice from item.symbols
                sym:GPTreeStruct = state.random[thread].choice(item.symbols)
                # Clone it
                return sym.clone() # Assuming GPTreeStruct has clone/deepcopy
            
            i += 1
        
        # If exhausted tries
        if i >= self.MAXNUMTRIES:
            ins:GPTreeStruct = self.prototype.sym_prototype.lightClone()
            ins.owner = None
            ins.status = False
            ins.effRegisters = set()
            ins.buildTree(state, thread)
            return ins
        
        sys.stderr.write(f"The index list cannot find the index {index}")
        sys.exit(1)

        return None

    def getRandSymbolByIndex(self, index:int, state:EvolutionState, thread:int, subpopulation:int)->GPTreeStruct:
        item = self.getItemByIndex(index)
        
        if state.random[thread].uniform(0,1) > item.get_tabu_frequency():
            sym = state.random[thread].choice(item.symbols)
            return sym.clone()
        else:
            # MicroMutation
            sym = state.random[thread].choice(item.symbols)
            # Assuming microMut is defined in parent class or globally
            ins = self.microMut(sym, state, thread, subpopulation)
            return ins

    def addSymbol(self, symbol:GPTreeStruct, state:EvolutionState, thread:int):
        # return the index of the newly added (or existing) symbol
        # # check the compatibility
        # initializer = state.initializer
        
        # # Accessing constraints to check treetype compatibility
        # symbol_constraints = symbol.constraints(initializer)
        # prototype_constraints = self.prototype.sym_prototype.constraints(initializer)
        
        # if symbol_constraints.treetype != prototype_constraints.treetype:
        #     return -1
        
        # check the duplication
        name = str(symbol)
        for listind in range(len(self)):
            item = self[listind]
            # if item.isduplicated(symbol):
                
            #     # exist = False
            #     # for sym in item.symbols:
            #     #     # Compare genotype via string representation
            #     #     if str(sym) == str(symbol):
            #     #         exist = True
            #     #         break
            #     exist = str(symbol) in item.symbol_names
                
            #     if not exist:
            #         item.addSymbol(symbol)
                
            #     # move the visited item forward (Self-organizing list logic)
            #     next_pos = int(listind * 0.8)
            #     # if next_pos < listind:
            #     # swap the listind and next_pos
            #     tmp = self[next_pos]
            #     self[next_pos] = item
            #     self[listind] = tmp
                
            #     return item.index
            
            if name in item.symbol_names:
                # move the visited item forward (Self-organizing list logic)
                next_pos = int(listind * 0.8)
                # if next_pos < listind:
                # swap the listind and next_pos
                tmp = self[next_pos]
                self[next_pos] = item
                self[listind] = tmp
                
                return item.index
        
        # if not exist, then add to the list.
        n = len(self)
        index = n
        
        # add the new symbol into the list
        nd:Index[GPTreeStruct] = self.prototype.clone()
        nd.index = index
        nd.addSymbol(symbol)
        
        # Insert at the beginning of the list (position 0)
        self.insert(0, nd)
        
        return index

    def setSymbol(self, symbol:GPTreeStruct, index:int, state:EvolutionState, thread:int):
        name = str(symbol)
        for item in self:
            # if item.isduplicated(symbol):
            #     # exist = any(str(sym) == str(symbol) for sym in item.symbols)
            #     exist = str(symbol) in item.symbol_names
            #     if not exist:
            #         item.addSymbol(symbol)
            #     return item.index
            if name in item.symbol_names:
                return item.index
        
        item = self.getItemByIndex(index)
        item.symbols.clear()
        item.symbol_names.clear()

        item.set_tabu_frequency(0)

        item.symbols.append(symbol)
        item.symbol_names.add(name)
        return index

    def trimList(self, state:EvolutionState, thread:int):
        tarsize = self.maxlistsize
        
        while len(self) > tarsize:
            thre = state.random[thread].uniform(0,1)
            targeti = state.random[thread].randint(0, len(self)-1)
            
            for i in range(len(self)):
                idx_in_list = (targeti + i) % len(self)
                item:Index[GPTreeStruct] = self[idx_in_list]
                
                # Logic: Higher index items are more likely to be removed
                if 0.1 + 0.9 * item.index / len(self) >= thre:
                    removed_index_val = item.index
                    
                    self.pop(idx_in_list)
                    
                    # Renumber
                    for it in self:
                        if it.index > removed_index_val:
                            it.index -= 1
                            
                    break
                
        self.sortList()

    def updateSymbolsFromRandPop(self, state:EvolutionState, thread:int, subpop:SubpopulationFLO):
        self.clear_tabu_frequency()
        
        num_inds = 0.5 * len(subpop.individuals)
        
        for _ in range(int(num_inds)):
            # Tournament selection
            lgpind = subpop.individuals[state.random[thread].randint(0, int(num_inds)-1)]
            for _ in range(6): # tr=1 to 7
                tmpind:LGPIndividual = subpop.individuals[state.random[thread].randint(0, int(num_inds)-1)]
                if tmpind.fitness.betterThan(lgpind.fitness):
                    lgpind = tmpind
            
            for tree in lgpind.getTreelist():
                # clone tree and add
                sym_clone = tree.clone()
                symind = self.addSymbol(sym_clone, state, thread)
                
                item = self.getItemByIndex(symind) # Helper from parent class recommended
                if item:
                    freq = item.get_tabu_frequency()
                    item.set_tabu_frequency(min(Index.TABU_THRESHOLD, freq + 1.0 * 2 / num_inds))

    def updateSymbolsFromObjs(self, state:EvolutionState, thread:int, subpop:SubpopulationFLO, board:Board):
        for ob in range(self.numobjectives):
            self.objectives[ob].preprocessing(state, thread, self, board, self.batchsize, self.boardsize[ob])
            self.objectives[ob].setPrivateCoefficiency(self, board)
            
        for ob in range(self.numobjectives):
            self.objectives[ob].updateNewIndexList(state, thread, self, board)

    def cloneIndexList(self)->IndexList:
        new_list = IndexList4LGP_FLR()
        new_list.self_state = self.self_state
        new_list.self_thread = self.self_thread
        new_list.nodeselect = self.nodeselect.clone() # Assuming clone exists
        new_list.maxlistsize = self.maxlistsize
        new_list.builder = self.builder
        new_list.prototype = self.prototype
        
        # Deep copy of items
        for ind in self:
            new_list.append(ind.clone())
            
        return new_list

    def evaluateObjectives(self, lst:IndexList, board:Board) -> float:
        fit = 0.0
        for ob in range(self.numobjectives):
            fit += self.coefficiency[ob] * self.objectives[ob].evaluate(lst, board)
        return fit


    def checkPoints(self, p1:GPNode, p2:GPNode, state:EvolutionState, thread:int, ind:LGPIndividual, treeStr:GPTreeStruct) -> bool:
        """
        Validates if the two nodes are compatible for replacement and different enough.
        """
        res = False
        
        # p1 and p2 must have the same arity
        if p1.expectedChildren() == p2.expectedChildren():
            # They must be logically different (comparing string representation)
            if str(p1) != str(p2):
                res = True
                
        return res

    def microMut(self, orig_ins:GPTreeStruct, state:EvolutionState, thread:int, subpopi:int) -> GPTreeStruct:
        """
        Performs a micro-mutation by picking a node and replacing it with a 
        compatible point-mutated version.
        """
        
        # Clone the original instruction to mutate it
        ins = orig_ins.clone()
        p1 = None
        p2 = None
        func_set:GPPrimitiveSet = orig_ins.species.primitiveset  # all trees have the same function set
        
        res = False
        # Try up to 10 times to find a valid mutation point
        for t in range(10):
            self.nodeselect.reset()
            # Pick a node within the tree structure
            p1 = self.nodeselect.pickNode(state, subpopi, thread, None, ins)
            
            # 1. Case: Destination Register (WriteRegisterGPNode)
            if isinstance(p1, WriteRegisterGPNode):
                reg_node = func_set.registers[state.random[thread].randint(0, len(func_set.registers)-1)]
                p2 = reg_node.lightClone()
                p2.resetNode(state, thread)
                
            # 2. Case: Terminals (No children)
            elif len(p1.children) == 0:
                # Check if we should mutate to a constant
                if (state.random[thread].uniform(0,1) < self.builder.probCons and 
                    self.builder.canAddConstant(ins.child)):
                    
                    # if isinstance(p1, Entity) and state.random[thread].uniform() < 0.5:
                    #     p2 = p1.lightClone()
                    #     p2.getArguments().varyNode(state, thread, p1)
                    # else:
                    #     const_node = func_set.constants_v[state.random[thread].nextInt(len(func_set.constants_v))]
                    #     p2 = const_node.lightClone()
                    #     p2.resetNode(state, thread)
                    const_node = func_set.constants[state.random[thread].randint(0, len(func_set.constants)-1)]
                    p2 = const_node.lightClone()
                    p2.resetNode(state, thread)
                else:
                    # Mutate to a non-constant terminal
                    non_const = func_set.nonconstants[state.random[thread].randint(0, len(func_set.nonconstants)-1)]
                    p2 = non_const.lightClone()
                    p2.resetNode(state, thread)
            
            # 3. Case: Functions (Non-terminals)
            else:
                # if isinstance(p1, Entity) and state.random[thread].nextDouble() < 0.5:
                #     p2 = p1.lightClone()
                #     p2.getArguments().varyNode(state, thread, p1)
                # else:
                #     non_term = func_set.nonterminals_v[state.random[thread].nextInt(len(func_set.nonterminals_v))]
                #     p2 = non_term.lightClone()
                #     p2.resetNode(state, thread)
                non_term = func_set.nonterminals[state.random[thread].randint(0, len(func_set.nonterminals)-1)]
                p2 = non_term.lightClone()
                p2.resetNode(state, thread)

            # Validate the mutation point
            res = self.checkPoints(p1, p2, state, thread, None, ins)
            if res:
                break
        
        if not res:
            # If no valid mutation found, generate a completely new random tree
            ins:GPTreeStruct = self.prototype.sym_prototype.lightClone()
            ins.owner = None 
            ins.status = False
            ins.effRegisters = set()
            ins.buildTree(state, thread)
        else:
            # Perform the replacement
            p1.replaceWith(p2)
            ins.child.parent = ins
            ins.child.argposition = 0
            
        return ins

    def getItemByIndex(self, index: int):
        """
        Finds the Index wrapper object associated with a specific logical index.
        """
        for it in self:
            if it.index == index:
                return it
        return None

    def sortList(self):
        """
        Sorts the list based on the 'index' attribute of the Index objects.
        Replaces the Java IndexComparator class.
        """
        self.sort(key=lambda x: x.index)