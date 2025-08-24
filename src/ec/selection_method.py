from abc import ABC, abstractmethod
from .breeding_source import BreedingSource
from .gp_individual import GPIndividual
from .evolution_state import EvolutionState
# from .population import Population

class SelectionMethod(BreedingSource, ABC):
    '''
    A SelectionMethod is a type of BreedingSource that doesn't modify individuals — instead, it selects individuals from the existing population, usually based on some fitness criteria.
    It is used as the input source for a BreedingPipeline.
    '''

    INDS_PRODUCED = 1

    def typicalIndsProduced(self):
        return self.INDS_PRODUCED

    def prepareToProduce(self, state: EvolutionState, subpopulation: int, thread: int):
        return

    def finishProducing(self, state: EvolutionState, subpopulation: int, thread: int):
        return

    def produce(self, min: int, max: int, start: int, subpopulation: int,
                inds: list, state: EvolutionState, thread: int)->int:
        n = self.INDS_PRODUCED
        if n < min:
            n = min
        if n > max:
            n = max

        for q in range(n):
            pos = self.produce_select(subpopulation, state, thread)
            inds[start + q] = state.population.subpops[subpopulation].individuals[pos]
        
        return n

    @abstractmethod
    def produce_select(self, subpopulation: int, state: EvolutionState, thread: int) -> int:
        pass
