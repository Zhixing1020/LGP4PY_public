from src.ec import *
from src.ec.util import *
from tasks.problem import Problem
import numpy as np
from src.lgp.algorithm.typed_lgp.ec import *

# import src.lgp.algorithm.typed_lgp.dsl.type_checker as tc
from src.lgp.algorithm.typed_lgp.individual.primitives.typed_interface import TypedInterface
from typing import Union

class TypedInput(GPNode, TypedInterface):

    def __init__(self):
        GPNode.__init__(self)
        TypedInterface.__init__(self)

    def __str__(self):
        return "I"

    def setup(self, state: TypedEvolutionState, base: Parameter):
        super().setup(state, base)
        self.agent = state.typeagent

    def expectedChildren(self):
        return 0

    def eval(self, state:EvolutionState, thread:int, input:TypedData, individual, problem, argval: list = None):
        input.type = self.output_type
        if not input.to_vectorize:
            input.value = problem.X
        else:
            input.values = problem.data

    # def __hash__(self):
    #     return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, TypedInput) \
                and self.output_type == other.output_type

    def resetNode(self, state, thread):
        super().resetNode(state, thread)

    def lightClone(self) -> 'TypedInput':
        obj:TypedInput = GPNode.lightClone(self)
        obj.lightCloneInterface(self)
        return obj

    def clone(self) -> 'TypedInput':
        obj:TypedInput = GPNode.clone(self)
        obj.lightCloneInterface(self)
        return obj
    
    def swapCompatibleWith(self, node:'TypedInterface', state:TypedEvolutionState=None) -> bool:
        res = GPNode.swapCompatibleWith(self, node) and TypedInterface.swapCompatibleWith(self, node, state)
        return res
