from src.ec.gp_species import GPSpecies
from src.lgp.individual.gp_tree_struct import GPTreeStruct
from src.ec.evolution_state import EvolutionState
from src.ec.util.parameter import Parameter
from src.ec.gp_builder import GPBuilder

class LGPSpecies(GPSpecies):

    def __init__(self):
        super().__init__()
        self.instr_prototype:GPTreeStruct = None

    def clone(self):
        new_species = super().clone()
        new_species.instr_prototype = self.instr_prototype.clone()
        return new_species
    
    def __deepcopy__(self, memo):
        self.clone()

    def setup(self, state: EvolutionState, base: Parameter):
        super().setup(state, base)
        self.instr_prototype = state.parameters.getInstanceForParameter(
            base.push(self.P_INDIVIDUAL).push(self.i_prototype.P_TREE).push("0"), 
            super().default_base().push(self.P_INDIVIDUAL).push(self.i_prototype.P_TREE).push("0"),
            GPTreeStruct)
        self.instr_prototype.setup(state, base.push(self.P_INDIVIDUAL).push(self.i_prototype.P_TREE).push("0"))
        self.instr_prototype.species = self