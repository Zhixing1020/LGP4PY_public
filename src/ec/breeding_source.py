from abc import ABC, abstractmethod
from src.ec import *
from src.ec.util import *

class BreedingSource(ABC):
    '''
    BreedingSource is an abstract superclass that defines a common interface for all components that can produce individuals for the next generation.
    BreedingSource is the general blueprint for any object that can "breed" or "supply" individuals — including both selection and variation mechanisms.
    It is the superclass of BreedingPipeline and SelectionMethod
    '''
    P_PROB = "prob"
    NO_PROBABILITY = -1.0

    def __init__(self):
        self.probability = BreedingSource.NO_PROBABILITY

    def setup(self, state:EvolutionState, base:Parameter):
        default = self.defaultBase()
        if not state.parameters.exists(base.push(self.P_PROB), default.push(self.P_PROB)):
            self.probability = self.NO_PROBABILITY
        else:
            self.probability = state.parameters.getDoubleWithDefault(base.push(self.P_PROB), default.push(self.P_PROB), 0.0)
            if self.probability < 0.0:
                state.output.error(
                    "Breeding Source's probability must be a double floating point value >= 0.0, or empty, which represents NO_PROBABILITY."
                )

    def getProbability(self):
        return self.probability

    # def setProbability(self, prob):
    #     self.probability = prob

    @abstractmethod
    def typicalIndsProduced(self):
        pass

    @abstractmethod
    def prepareToProduce(self, state, subpopulation, thread):
        pass

    @abstractmethod
    def finishProducing(self, state, subpopulation, thread):
        pass

    @abstractmethod
    def produce(self, min:int, max:int, start:int, subpopulation:int, inds:list[GPIndividual], 
                state:EvolutionState, thread:int)->int:
        pass

    def clone(self):
        obj = self.__class__()
        obj.probability = self.probability
        return obj

    # def preparePipeline(self, hook):
    #     pass

    def __str__(self):
        return self.__class__.__name__
    
    def defaultBase(self)->Parameter:
        return Parameter("")