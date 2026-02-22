from abc import ABC, abstractmethod
from typing import List

from src.ec.util.parameter import Parameter
# from src.ec.population import Population
from src.ec.evolution_state import EvolutionState
from src.ec.gp_individual import GPIndividual
from src.ec.breeding_source import BreedingSource
from src.ec.selection_method import SelectionMethod


class BreedingPipeline(BreedingSource):
    '''
    BreedingPipeline is a concrete subclass of BreedingSource that performs genetic operations like crossover, mutation, or reproduction. 
    It takes input from one or more other BreedingSources (often SelectionMethods), applies a genetic operator, and outputs individuals.
    '''
    __slots__ = ("sources", "operatorRate", "mybase",)
    #Indicates that a source is the exact same source as the previous source.
    V_SAME = "same" 

    #Indicates the probability that the Breeding Pipeline will perform its mutative action instead of just doing reproduction.
    P_LIKELIHOOD = "likelihood"

    # Indicates that the number of sources is variable and determined by the user in the parameter file.
    DYNAMIC_SOURCES = -1

    #Standard parameter for number of sources (only used if numSources returns DYNAMIC_SOURCES
    P_NUMSOURCES = "num-sources"

    #Standard parameter for individual-selectors associated with a BreedingPipeline
    P_SOURCE = "source"

    # Standard parameter for node-selectors associated with a GPBreedingPipeline
    P_NODESELECTOR = "ns"

    # Standard parameter for tree fixin
    P_TREE = "tree"

    NO_SIZE_LIMIT = -1
    TREE_UNFIXED = -1

    def __init__(self):
        self.mybase:Parameter = None
        # self.likelihood = 1.0
        self.sources: List[BreedingSource] = []
        self.operatorRate: List[float] = []

    @abstractmethod
    def numSources(self):
        pass

    def minChildProduction(self)->int:
        if len(self.sources) == 0:
            return 0
        return min(s.typicalIndsProduced() for s in self.sources)

    def maxChildProduction(self)->int:
        if len(self.sources) == 0:
            return 0
        return max(s.typicalIndsProduced() for s in self.sources)

    def typicalIndsProduced(self)->int:
        return self.maxChildProduction()

    # def defaultBase(self)->Parameter:
    #     return self.mybase

    def setup(self, state:EvolutionState, base:Parameter):
        super().setup(state, base)
        self.mybase = base
        def_:Parameter = self.defaultBase()

        # self.likelihood = state.parameters.getDoubleWithDefault(
        #     base.push(self.P_LIKELIHOOD), def_.push(self.P_LIKELIHOOD), 1.0
        # )
        # if self.likelihood < 0.0 or self.likelihood > 1.0:
        #     state.output.fatal(f"Breeding Pipeline likelihood must be between 0.0 and 1.0 inclusive for"
        #                        +f"{base.push(self.P_LIKELIHOOD)} or {def_.push(self.P_LIKELIHOOD)}")
        
        numsources = self.numSources()

        if numsources == self.DYNAMIC_SOURCES:
            numsources = state.parameters.getIntWithDefault(base.push(self.P_NUMSOURCES), def_.push(self.P_NUMSOURCES), 0)
            if numsources < 0:
                state.output.fatal("Breeding pipeline num-sources must exist and be >= 0",
                                    base.push(self.P_NUMSOURCES), def_.push(self.P_NUMSOURCES))

        self.sources = [None] * numsources
        self.operatorRate = [None] * numsources
        for x in range(numsources):
            p = base.push(self.P_SOURCE).push(str(x))
            d = def_.push(self.P_SOURCE).push(str(x))
            s = state.parameters.getString(p, d)
            if s is not None and s == self.V_SAME:
                if x == 0:
                    state.output.fatal("Source #0 cannot be declared with \"same\".", p, d)
                self.sources[x] = self.sources[x - 1]
            else:
                self.sources[x] = state.parameters.getInstanceForParameter(p, d, BreedingSource)
                self.sources[x].setup(state, p)
                self.operatorRate[x] = state.parameters.getDoubleWithDefault(p.push(self.P_PROB), d.push(self.P_PROB),  0.0)


    def clone(self):
        # import copy
        # c = copy.copy(self)
        c = super().clone()
        c.sources = [None] * len( self.sources )
        c.operatorRate = [None] * len(self.operatorRate)
        # c.likelihood = self.likelihood
        for x in range(len(self.sources)):
            if x == 0 or self.sources[x] != self.sources[x - 1]:
                c.sources[x] = (self.sources[x].clone())
                c.operatorRate[x] = (self.operatorRate[x])
            else:
                c.sources[x] = (c.sources[x - 1])
                c.operatorRate[x] = (c.operatorRate[x - 1])
        return c

    def reproduce(self, 
                  n:int, 
                  start:int, 
                  subpopulation:int, 
                  inds:List[GPIndividual], 
                  state:EvolutionState, 
                  thread:int, 
                  produceChildrenFromSource:bool)->int:
        if produceChildrenFromSource:
            self.sources[0].produce(n, n, start, subpopulation, inds, state, thread)
        if isinstance(self.sources[0], SelectionMethod):
            for q in range(start, n + start):
                inds[q] = inds[q].clone()
        return n

    def prepareToProduce(self, 
                         state:EvolutionState, 
                         subpopulation:int, 
                         thread:int):
        for x in range(len(self.sources)):
            if x == 0 or self.sources[x] != self.sources[x - 1]:
                self.sources[x].prepareToProduce(state, subpopulation, thread)

    def finishProducing(self, 
                         state:EvolutionState, 
                         subpopulation:int, 
                         thread:int):
        for x in range(len(self.sources)):
            if x == 0 or self.sources[x] != self.sources[x - 1]:
                self.sources[x].finishProducing(state, subpopulation, thread)

    # def preparePipeline(self, hook):
    #     for source in self.sources:
    #         source.preparePipeline(hook)

    # def individualReplaced(self, 
    #                        state:EvolutionState, 
    #                        subpopulation:int, 
    #                        thread:int, 
    #                        individual:int):
    #     for source in self.sources:
    #         # if isinstance(source, SteadyStateBSourceForm):
    #         #     source.individualReplaced(state, subpopulation, thread, individual)
    #         pass

    # def sourcesAreProperForm(self, 
    #                          state:EvolutionState):
    #     for x, source in enumerate(self.sources):
    #         # if not isinstance(source, SteadyStateBSourceForm):
    #         #     state.output.error("Source is not SteadyStateBSourceForm",
    #         #                        self.mybase.push(self.P_SOURCE).push(str(x)),
    #         #                        self.defaultBase().push(self.P_SOURCE).push(str(x)))
    #         # else:
    #         #     source.sourcesAreProperForm(state)
    #         pass

    