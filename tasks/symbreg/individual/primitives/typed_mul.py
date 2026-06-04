from src.ec import *
from src.ec.util.parameter import Parameter
from src.lgp.algorithm.typed_lgp.ec.typed_data import TypedData
from tasks.problem import Problem
from typing import override
import numpy as np
from src.lgp.algorithm.typed_lgp.ec import *
from src.lgp.algorithm.typed_lgp.individual.primitives.typed_interface import TypedInterface
from src.lgp.individual.primitive.mul import Mul
from tasks.symbreg.DSL.basic_dsl.sr_types import Float

class TypedMul(Mul, TypedInterface):
    def __init__(self):
        Mul.__init__(self)
        TypedInterface.__init__(self)
        self.output_type = Float
        self.input_types = (Float, Float)

    def setup(self, state: TypedEvolutionState, base: Parameter):
        super().setup(state, base)
        self.agent = state.typeagent

    def eval(self, state: EvolutionState, thread: int, input: TypedData,
             individual, problem: Problem, argval: list[TypedData] = None):
        input.type = self.output_type
        super().eval(state, thread, input, individual, problem, argval)

    def lightClone(self) -> 'TypedMul':
        obj:TypedMul = Mul.lightClone(self)
        obj.lightCloneInterface(self)
        return obj

    def clone(self) -> 'TypedMul':
        obj:TypedMul = Mul.clone(self)
        obj.lightCloneInterface(self)
        return obj
    
    def swapCompatibleWith(self, node:'TypedInterface', state:TypedEvolutionState=None) -> bool:
        res = GPNode.swapCompatibleWith(self, node) and TypedInterface.swapCompatibleWith(self, node, state)
        return res
        
