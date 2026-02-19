

from abc import ABC, abstractmethod
from src.ec import *
from src.ec.util import *
from src.lgp.algorithm.LandscapeOptimization.indexing.board import Board
from src.lgp.algorithm.LandscapeOptimization.indexing.genoVector import GenoVector
from src.lgp.algorithm.LandscapeOptimization.subpopulationFLO import SubpopulationFLO


class NeighborhoodSearch(BreedingPipeline, ABC):
    
    # Static string constants
    P_NEIGHBORHOODSEARCH = "nbhsearch"
    P_MASKLENGTH = "mask_length"
    P_MACROSIZE = "macro_size"
    P_MAXSTEP = "maxstep"
    P_NUM_TRIES = "tries"
    P_INSERT = "prob_insert"
    P_DELETE = "prob_delete"
    
    INDS_PRODUCED = 1
    NUM_SOURCES = 2
    
    default_step = 1.0
    default_cosLimit = 0.0
    
    def __init__(self):
        super().__init__()
        self.maxMaskSize = 0
        self.maxMacroSize = 0
        self.maxStep = 0.0
        self.numTries = 0
        
        self.mask:list[int] = None
        self.masklength = 0
        self.master_i = 0
        
        self.cur_generation = -1
        self.leadBoard:Board = None
        
        self.addRate = 0.0
        self.removeRate = 0.0
        
        # NUM_SOURCES is 2
        self.parents:list[GPIndividual] = [None] * self.NUM_SOURCES

    def defaultBase(self):
        return Parameter(self.P_NEIGHBORHOODSEARCH)

    def numSources(self):
        return self.NUM_SOURCES

    # def clone(self):
    #     # Python equivalent of the Java clone()
    #     c = super().clone()

    #     # c.parents = list(self.parents) # Shallow copy of the list, similar to Java array clone
    #     return c

    def setup(self, state, base):
        super().setup(state, base)
        
        def_param = self.defaultBase()

        self.maxMaskSize = state.parameters.getInt(
            base.push(self.P_MASKLENGTH), def_param.push(self.P_MASKLENGTH))
        if self.maxMaskSize < 1:
            state.output.fatal("NeighborhoodSearch has an invalid max mask size (it must be >= 1).",
                               base.push(self.P_MASKLENGTH), def_param.push(self.P_MASKLENGTH))
            
        self.maxMacroSize = state.parameters.getInt(
            base.push(self.P_MACROSIZE), def_param.push(self.P_MACROSIZE))
        if self.maxMacroSize < 1:
            state.output.fatal("NeighborhoodSearch has an invalid max macro size (it must be >= 1).",
                               base.push(self.P_MACROSIZE), def_param.push(self.P_MACROSIZE))
            
        self.maxStep = state.parameters.getDoubleWithDefault(
            base.push(self.P_MAXSTEP), def_param.push(self.P_MAXSTEP), 0.2)
        if self.maxStep <= 0:
            state.output.fatal("NeighborhoodSearch has an invalid max step (it must be in (0, 1]).",
                               base.push(self.P_MAXSTEP), def_param.push(self.P_MAXSTEP))
            
        self.numTries = state.parameters.getIntWithDefault(
            base.push(self.P_NUM_TRIES), def_param.push(self.P_NUM_TRIES), 20)
        if self.numTries == 0:
            state.output.fatal("NeighborhoodSearch has an invalid number of tries (it must be >= 1).",
                               base.push(self.P_NUM_TRIES), def_param.push(self.P_NUM_TRIES))

        self.addRate = state.parameters.getDoubleWithDefault(base.push(self.P_INSERT), def_param.push(self.P_INSERT), 0.65)
        if self.addRate < 0 or self.addRate > 1:
            state.output.fatal("the probability of adding symbols in NeighborhoodSearch must be [0,1].", 
                               base.push(self.P_INSERT), def_param.push(self.P_INSERT))
            
        self.removeRate = state.parameters.getDoubleWithDefault(base.push(self.P_DELETE), def_param.push(self.P_DELETE), 0.35)
        if self.removeRate < 0 or self.removeRate > 1:
            state.output.fatal("the probability of removing symbols in NeighborhoodSearch must be [0,1].", 
                               base.push(self.P_DELETE), def_param.push(self.P_DELETE))

    def produce(self, min:int, max:int, start:int, subpopulation:int, inds:list[GPIndividual], state:EvolutionState, thread:int):
        
        # Standard produce logic
        if not isinstance(state.population.subpops[subpopulation], SubpopulationFLO):
            state.output.fatal("NeighborhoodSearch does not support other subpopulation types except SubpopulationFLO")
        
        n = self.INDS_PRODUCED
        
        # if not state.random[thread].nextBoolean(self.likelihood):
        #     return self.reproduce(n, start, subpopulation, inds, state, thread, True)

        q = start
        while q < n + start:
            if self.sources[0] == self.sources[1]:
                self.sources[0].produce(2, 2, 0, subpopulation, self.parents, state, thread)
            else:
                self.sources[0].produce(1, 1, 0, subpopulation, self.parents, state, thread)
                self.sources[1].produce(1, 1, 1, subpopulation, self.parents, state, thread)
            
            # Create the local parnts array/list
            parnts:list[GPIndividual] = [None] * self.NUM_SOURCES
            for ind_idx in range(len(parnts)):
                parnts[ind_idx] = self.parents[ind_idx]
            
            # Recursively call produce with the parents argument filled
            q += self.produce_individual(min, max, q, subpopulation, inds, state, thread, parnts)
            
        return n

    @abstractmethod
    def produce_individual(self, min, max, start, subpopulation, inds, state, thread, parents):
        """
        This represents the 'public abstract int produce(... Individual[] parents)' method.
        In subclasses, implement this logic.
        """
        pass

    @abstractmethod
    def maintainPhenotype(self, state:EvolutionState, thread:int, oldind:GPIndividual, newind:GPIndividual, newgv:GenoVector):
        pass