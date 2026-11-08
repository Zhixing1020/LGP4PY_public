from src.ec import *
from src.ec.util import *
from tasks.problem import Problem
from src.lgp.algorithm.typed_lgp.individual.primitives.typed_interface import TypedInterface
from src.lgp.algorithm.typed_lgp.ec import *

class TypedConstant(GPNode, TypedInterface):
    
    def __init__(self, val=0, name:str=None):
        GPNode.__init__(self)
        TypedInterface.__init__(self)
        self.value = val
        self.name = name
        self.all_constants = None

    def setup(self, state: TypedEvolutionState, base: Parameter):
        super().setup(state, base)
        self.agent = state.typeagent
        self.all_constants = self.agent.getConstantSet()

    def getValue(self):
        return self.value
    
    def setValue(self, val=0, name:str=None):
        self.value = val
        if self.agent.tc.check_type(self.value, bool):
            self.output_type = bool
        elif self.agent.tc.check_type(self.value, int):
            self.output_type = int
        elif self.agent.tc.check_type(self.value, (int, int)):
            self.output_type = tuple(int, int)    
        elif self.agent.tc.check_type(self.value, float):
            self.output_type = float
        else:
            raise Exception(f"unknown output type of {self.value}")    
        self.name = name

    
    def __str__(self):
        return self.name

    def expectedChildren(self):
        return 0

    def eval(self, state:TypedEvolutionState, thread:int, input:TypedData, individual, problem, argval: list = None):
        input.type = self.output_type
        if not input.to_vectorize:
            input.value = self.value
        else:
            input.values = self.value

    # def __hash__(self):
    #     return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, TypedConstant) \
                and self.value == other.getValue() \
                and self.name == other.name \
                and self.output_type == other.output_type

    def resetNode(self, state:EvolutionState, thread:int, candidate:dict=None):
        super().resetNode(state, thread)
        
        if candidate == None:
            name = state.random[thread].choice(list(self.all_constants.keys()))
            val = self.all_constants[name]
        else:
            name = state.random[thread].choice(list(candidate.keys()))
            val = candidate[name]

        self.setValue(val, name)

    def lightClone(self) -> 'TypedConstant':
        obj:TypedConstant = GPNode.lightClone(self)
        obj.lightCloneInterface(self)
        obj.setValue(self.value, self.name)
        obj.all_constants = self.all_constants
        return obj

    def clone(self) -> 'TypedConstant':
        obj:TypedConstant = GPNode.clone(self)
        obj.lightCloneInterface(self)
        obj.setValue(self.value, self.name)
        obj.all_constants = self.all_constants
        return obj

    def swapCompatibleWith(self, node:'TypedInterface', state:TypedEvolutionState=None) -> bool:
        res = GPNode.swapCompatibleWith(self, node) and TypedInterface.swapCompatibleWith(self, node, state)
        return res
