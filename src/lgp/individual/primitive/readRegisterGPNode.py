
from src.ec import *
from src.ec.util import *
from tasks.problem import Problem
# from src.ec.gp_node import GPNode

# from src.lgp.individual.lgp_individual import LGPIndividual


class ReadRegisterGPNode(GPNode):
    P_NUMREGISTERS = "numregisters"

    def __init__(self, index=0, range_=None):
        super().__init__()
        self.index = index
        self.range = range_ if range_ is not None else 1

    def getIndex(self):
        return self.index

    def getRange(self):
        return self.range

    def setIndex(self, index:int):
        self.index = index

    def setRange(self, range:int):
        self.range = range

    def __str__(self):
        return f"R{self.index}"

    def expectedChildren(self)->int:
        return 0

    def setup(self, state:EvolutionState, base:Parameter):
        super().setup(state, base)
        param_key = base.push(self.P_NUMREGISTERS)
        self.range = state.parameters.getInt(param_key, None)
        if self.range < 1:
            state.output.fatal("number of registers must be >=1")

    def eval(self, state:EvolutionState, thread:int, input:GPData, individual, problem:Problem, argval: list[float] = None):            
        # if isinstance(individual, LGPIndividual):
        #     input.value = individual.getRegisters(self.index)
        # else:
        #     state.output.fatal("The individual might not have getRegisters() method\n")
        if not input.to_vectorize:
            input.value = individual.getRegistersIndex(self.index)
        else:
            input.values = individual.getRegistersIndex(self.index)

    def __eq__(self, other):
        res = super().__eq__(other)
        return res and self.index == other.getIndex()

    def resetNode(self, state:EvolutionState, thread:int):
        self.index = state.random[thread].randint(0, self.range - 1)

    def enumerateNode(self, state:EvolutionState, thread:int):
        self.index = (self.index + 1) % self.range

    def lightClone(self):
        clone = super().lightClone()
        clone.setIndex(self.index)
        clone.setRange(self.range)
        return clone