

import math
import sys

from src.ec import *
from src.ec.util import *
from src.lgp.algorithm.LandscapeOptimization.FLReductionLGP.indexing.indexList4LGP_FLR import IndexList4LGP_FLR
from src.lgp.individual.gp_tree_struct import GPTreeStruct
from src.lgp.algorithm.typed_lgp.individual.typed_gp_tree_struct import TypedGPTreeStruct

from src.lgp.individual.primitive import *

class IndexList4TypedLGP_FLR(IndexList4LGP_FLR):
        
    def getSymbolByIndex(self, index:int, state:EvolutionState, thread:int, targetType=None, sourceType=None) -> GPTreeStruct:
        i = 0
        for _ in range(self.MAXNUMTRIES):
            offset = math.pow(-1, i) * math.ceil(i / 2.0)
            next_index = int(index + offset)
            
            if next_index < 0 or next_index >= len(self):
                i += 1
                continue
                
            item = self.getItemByIndex(next_index)
            
            if state.random[thread].uniform(0,1) > item.get_tabu_frequency()\
            and (not targetType or item.symbols[0].child.children[0].output_type == targetType) \
            and (not sourceType or item.symbols[0].child.children[0].input_type == sourceType):
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
            ins.buildTree(state, thread, targetType=targetType, sourceType=sourceType)

            return ins
        
        sys.stderr.write(f"The index list cannot find the index {index}")
        sys.exit(1)

        return None