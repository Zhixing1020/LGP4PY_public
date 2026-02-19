from src.ec.evolution_state import EvolutionState
from src.ec.gp_individual import GPIndividual
from src.ec.gp_species import GPSpecies
from src.ec.ec_defaults import ECDefaults
from typing import Set
from src.ec.util import Parameter, ParameterDatabase

class Subpopulation:
    P_SUBPOPULATION = "subpop"
    P_SUBPOPSIZE = "size"
    P_RETRIES = "duplicate-retries"
    P_SPECIES = "species"

    def __init__(self):
        self.individuals:list[GPIndividual] = []
        self.numDuplicateRetries = 0
        self.species = None
        self.duplicateSet:Set[str] = None

    def defaultBase(self):
        return ECDefaults.base().push(Subpopulation.P_SUBPOPULATION)
    
    # def emptyClone(self):
    #     clone = Subpopulation()
    #     clone.individuals = [None] * len(self.individuals)
    #     return clone

    # def resize(self, to_this: int):
    #     self.individuals = self.individuals[:to_this] + [None] * max(0, to_this - len(self.individuals))

    def clear(self):
        self.individuals = [None] * len(self.individuals)

    def setup(self, state:EvolutionState, base:Parameter):
        def_base = self.defaultBase()
        size_param = base.push(Subpopulation.P_SUBPOPSIZE)
        species_param = base.push(Subpopulation.P_SPECIES)
        retries_param = base.push(Subpopulation.P_RETRIES)

        self.species = state.parameters.getInstanceForParameter(
            species_param, def_base.push(Subpopulation.P_SPECIES), GPSpecies
        )
        self.species.setup(state, species_param)

        size = state.parameters.getInt(size_param, def_base.push(Subpopulation.P_SUBPOPSIZE))
        if size <= 0:
            state.output.fatal("Subpopulation size must be >= 1", size_param)

        self.numDuplicateRetries = state.parameters.getInt(retries_param, def_base.push(Subpopulation.P_RETRIES))
        if self.numDuplicateRetries < 0:
            state.output.fatal("Duplicate retries must be >= 0", retries_param)

        self.individuals = [None] * size

        # if self.loadInds:
        #     extra = state.parameters.getStringWithDefault(
        #         base.push(Subpopulation.P_EXTRA_BEHAVIOR), def_base.push(Subpopulation.P_EXTRA_BEHAVIOR), None
        #     )
        #     if extra is None:
        #         state.output.warning("No extra-behavior defined; defaulting to TRUNCATE.")
        #     elif extra.lower() == Subpopulation.V_FILL:
        #         self.extraBehavior = Subpopulation.FILL
        #     elif extra.lower() == Subpopulation.V_WRAP:
        #         self.extraBehavior = Subpopulation.WRAP
        #     elif extra.lower() != Subpopulation.V_TRUNCATE:
        #         state.output.fatal(f"Bad extra-behavior value: {extra}")


    def populate(self, state:EvolutionState, thread: int):
        start = 0

        self.duplicateSet:Set[str] = set() if self.numDuplicateRetries >= 1 else None

        for i in range(start, len(self.individuals)):
            for d in range(self.numDuplicateRetries + 1):
                ind = self.species.newIndividual(state, thread)
                if self.duplicateSet is None or ind.printTrees(state) not in self.duplicateSet or d == self.numDuplicateRetries:
                    self.individuals[i] = ind
                    if self.duplicateSet is not None and ind.printTrees() not in self.duplicateSet:
                        self.duplicateSet.add(ind.printTrees())
                    break
                

    def emptyclone(self)->'Subpopulation':
        subp = self.__class__()
        subp.species = self.species
        subp.numDuplicateRetries = self.numDuplicateRetries
        subp.individuals = [None] * len(self.individuals) # empty clone
        subp.duplicateSet = None
        if self.numDuplicateRetries >= 1:
            subp.duplicateSet = set()
            subp.duplicateSet.update(self.duplicateSet)
        return subp