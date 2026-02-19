
from src.ec import *
from src.ec.util import *
from src.lgp.algorithm.LandscapeOptimization.indexing.index import Index
from src.lgp.individual.gp_tree_struct import GPTreeStruct

from src.lgp.individual.primitive import *


class Index4LGP(Index[GPTreeStruct]):
    INDEX4LGP = "index4LGP"
    P_PRIMITIVESET: str = "fset"
    
    def setup(self, state:EvolutionState, base:Parameter):
        super().setup(state, base)
        
        # Creating the parameter path equivalent to base.push(SYMBOL_PROTO)
        def_param = Parameter(self.INDEX4LGP)
        
        # In Python, we assume state.parameters has a similar access method
        sym_proto_base = base.push(self.SYMBOL_PROTO)
        sym_proto_def = def_param.push(self.SYMBOL_PROTO)
        self.sym_prototype:GPTreeStruct = state.parameters.getInstanceForParameter(
            sym_proto_base, 
            sym_proto_def,
            GPTree
        )
        self.sym_prototype.setup(state, sym_proto_base)
        self.sym_prototype.species = state.parameters.getInstanceForParameter(
            sym_proto_base.push(Subpopulation.P_SPECIES), 
            sym_proto_def.push(Subpopulation.P_SPECIES), 
            GPSpecies
        )
        
        prim_base = sym_proto_base.push(Subpopulation.P_SPECIES).push(self.P_PRIMITIVESET)
        prim_def = sym_proto_def.push(Subpopulation.P_SPECIES).push(self.P_PRIMITIVESET)
        primset_name = state.parameters.getString(prim_base, prim_def)
        
        if primset_name is None or primset_name == "":
            state.output.fatal("We did not find the primitive set name", prim_base, prim_def)
        
        # for each primitive set in EvolutionState, we check the primitive set and grab the consistent one
        for ps in state.primitive_sets:
            if ps.name == primset_name:
                self.sym_prototype.species.primitiveset = ps.clone()
                break

    def isduplicated(self, newsym:GPTreeStruct) -> bool:
        res = False
        
        if len(self.symbols) > 0:
            item:GPTreeStruct = self.symbols[0] # Use the first item to check equality property
            
            # Genotype check via recursion
            if self.is_duplicate_instrs(item.child, newsym.child):
                return True
            
            # Phenotype check: specifically for trees of depth 3
            if item.child.depth() == 3 and newsym.child.depth() == 3:
                item_comp = [None] * 4
                newsym_comp = [None] * 4
                
                # Extracting string representations for comparison
                item_comp[0] = str(item.child)
                item_comp[1] = str(item.child.children[0])
                item_comp[2] = str(item.child.children[0].children[0])
                if len(item.child.children[0].children) > 1:
                    item_comp[3] = str(item.child.children[0].children[1])
                
                newsym_comp[0] = str(newsym.child)
                newsym_comp[1] = str(newsym.child.children[0])
                newsym_comp[2] = str(newsym.child.children[0].children[0])
                if len(newsym.child.children[0].children) > 1:
                    newsym_comp[3] = str(newsym.child.children[0].children[1])
                
                # Commutative property check (+, *, max, min)
                if item_comp[0] == newsym_comp[0]:
                    commutative_ops = ["+", "*", "max", "min"]
                    if item_comp[1] == newsym_comp[1] and item_comp[1] in commutative_ops:
                        if (len(item.child.children[0].children) > 1 and 
                            len(newsym.child.children[0].children) > 1):
                            if (item_comp[2] == newsym_comp[3] and 
                                item_comp[3] == newsym_comp[2]):
                                return True
                
                # FlowOperator check: ignore destination registers
                if (item_comp[1] == newsym_comp[1] and 
                    isinstance(item.child.children[0], FlowOperator) and
                    item_comp[2] == newsym_comp[2] and 
                    item_comp[3] == newsym_comp[3]):
                    return True
                    
        return res

    def clone(self):
        n = Index4LGP()
        
        n.num_inputs = self.num_inputs
        n.dim_inputs = self.dim_inputs
        
        n.input_lb = self.input_lb
        n.input_ub = self.input_ub
        
        n.sym_prototype = self.sym_prototype
        n.index = self.index
        
        n.set_tabu_frequency(self.get_tabu_frequency())
        
        # Deep copy the symbols list
        for t in self.symbols:
            n.symbols.append(t) # GPTreeStruct should have its own clone logic if needed
        
        n.symbol_names.update(self.symbol_names)
            
        return n

    def is_duplicate_instrs(self, p1:GPNode, p2:GPNode):
        # Compare current node strings
        if str(p1) != str(p2):
            return False
        
        # Compare children length
        cn = len(p1.children)
        if len(p2.children) != cn:
            return False
        
        # Recursive check on all children
        res = True
        for c in range(cn):
            res = res and self.is_duplicate_instrs(p1.children[c], p2.children[c])
            if not res: break
            
        return res