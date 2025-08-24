from src.ec.util.parameter import Parameter
from src.ec.util.parameter_database import ParameterDatabase
from src.ec.gp_individual import GPIndividual
from src.ec.fitness import Fitness
from src.ec.evolution_state import EvolutionState
from src.ec.gp_defaults import GPDefaults
from src.ec.breeding_pipeline import BreedingPipeline
from src.ec.gp_primitive_set import GPPrimitiveSet
from abc import ABC
from typing import Type
from copy import deepcopy

class GPSpecies(ABC):
    P_INDIVIDUAL: str = "ind"
    P_PIPE: str = "pipe"
    P_FITNESS: str = "fitness"
    P_GPSPECIES: str = "species"
    P_PRIMITIVESET: str = "fset"

    def __init__(self):
        self.i_prototype: GPIndividual = None
        self.pipe_prototype: BreedingPipeline = None
        self.f_prototype: Fitness = None
        self.primitiveset:GPPrimitiveSet = None

    def clone(self):
        new_species = self.__class__()
        new_species.i_prototype = self.i_prototype.clone()
        new_species.f_prototype = self.f_prototype.clone()
        new_species.pipe_prototype = self.pipe_prototype.clone()
        new_species.primitiveset = self.primitiveset.clone()
        return new_species
    
    def __deepcopy__(self, memo):
        self.clone()

    def newIndividual(self, state: EvolutionState, thread: int) -> GPIndividual:
        newind = self.i_prototype.lightClone()

        newind.fitness = self.f_prototype.clone()
        newind.evaluated = False
        newind.species = self

        # Initialize the trees
        # for tree in newind.treelist:
        #     tree.buildTree(state, thread)
        newind.rebuildIndividual(state, thread)
        return newind

    def setup(self, state: EvolutionState, base: Parameter):
        default = GPSpecies.default_base()

        self.pipe_prototype = state.parameters.getInstanceForParameter(
            base.push(self.P_PIPE), default.push(self.P_PIPE), BreedingPipeline)
        self.pipe_prototype.setup(state, base.push(self.P_PIPE))

        self.i_prototype = state.parameters.getInstanceForParameter(
            base.push(self.P_INDIVIDUAL), default.push(self.P_INDIVIDUAL), GPIndividual)
        self.i_prototype.species = self
        self.i_prototype.setup(state, base.push(self.P_INDIVIDUAL))
        # Ensure individual prototype is a GPIndividual
        if not isinstance(self.i_prototype, GPIndividual):
            state.output.fatal(f"The Individual class for the Species {self.__class__.__name__} must be a subclass of GPIndividual.", base)
        self.i_prototype.species = self

        self.f_prototype = state.parameters.getInstanceForParameter(
            base.push(self.P_FITNESS), default.push(self.P_FITNESS), Fitness)
        self.f_prototype.setup(state, base.push(self.P_FITNESS))

        primset_name = state.parameters.getString(base.push(self.P_PRIMITIVESET), default.push(self.P_PRIMITIVESET))
        if primset_name is None or primset_name == "":
            state.output.fatal("We did not find the primitive set name", base.push(self.P_PRIMITIVESET), default.push(self.P_PRIMITIVESET))
        
        # for each primitive set in EvolutionState, we check the primitive set and grab the consistent one
        for ps in state.primitive_sets:
            if ps.name == primset_name:
                self.primitiveset = ps.clone()
                break

    @classmethod
    def default_base(cls) -> Parameter:
        return GPDefaults.base().push(cls.P_GPSPECIES)