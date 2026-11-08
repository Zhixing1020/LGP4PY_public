from src.ec import *
# typed write register
from tasks.problem import Problem
from src.lgp.individual.primitive import ReadRegisterGPNode
from typing import override
# from tasks.ARC.DSL.arc_agent import ARCAgent as arc
from src.lgp.algorithm.typed_lgp.ec import *
import numpy as np
from src.lgp.algorithm.typed_lgp.individual.primitives.typed_interface import TypedInterface
from src.lgp.algorithm.typed_lgp.individual.primitives.typed_register import TypedRegister


class TypedReadRegister(ReadRegisterGPNode, TypedInterface):
    def __init__(self, index=0, range_=None):
        ReadRegisterGPNode.__init__(self, index, range_)
        TypedInterface.__init__(self)

    def eval(self, state:EvolutionState, thread:int, input:TypedData, individual, 
             problem:Problem, argval: list[TypedData] = None):
        
        register:TypedRegister = individual.getRegistersIndex(self.index)

        # set the returning types based on the input types of functions
        # if isinstance(self.parent, TypedInterface) and self.argposition is not None:
        #     input.type = self.output_type = self.parent.getInputType(no_callable=True)[self.argposition]
        # else:
        #     input.type = self.output_type = individual.outputType

        input.type = self.output_type = self.parent.getInputType(no_callable=False)[self.argposition]

        # set the returning value and types
        if not input.to_vectorize:
            input.value = register.getValue(input.type)
        else:
            input.values = register.getValue(input.type)

    def lightClone(self):
        obj:TypedReadRegister = ReadRegisterGPNode.lightClone(self)
        obj.lightCloneInterface(self)
        return obj
    
    def clone(self):
        obj:TypedReadRegister = ReadRegisterGPNode.clone(self)
        obj.lightCloneInterface(self)
        return obj
    
    def swapCompatibleWith(self, node:'TypedInterface', state:TypedEvolutionState=None) -> bool:
        res = GPNode.swapCompatibleWith(self, node) and TypedInterface.swapCompatibleWith(self, node, state)
        return res
    
    def collectInputTypes(self, s: set):
        # the output type of ReadRegister is also the input type for the register, so we also need to add it into the set of input types.
        s.add(self.getOutputType())
        super().collectInputTypes(s)
