from abc import ABC, abstractmethod
from src.ec import *
from src.ec.gp_data import GPData
from src.ec.util import Parameter

class Problem(ABC):
    
    P_PROBLEM = "problem"
    P_DATA = "data"


    def __init__(self):
        self.input:GPData = None

    def defaultBase(self) -> Parameter:
        return GPDefaults.base().push(self.P_PROBLEM)
    
    def setup(self, state: EvolutionState, base: Parameter):
        def_base = self.defaultBase()

        p = base.push(self.P_DATA)
        self.input = state.parameters.getInstanceForParameter(p, def_base.push(self.P_DATA), GPData)
        self.input.setup(state, p)

    def clone(self):
        import copy
        new_prob = self.__class__()
        new_prob.input = copy.deepcopy(self.input)
        return new_prob
    
    @abstractmethod
    def evaluate(self, state:EvolutionState, ind, subpopulation:int, threadnum:int):
        pass