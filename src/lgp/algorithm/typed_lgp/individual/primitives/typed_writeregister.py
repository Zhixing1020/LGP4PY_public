
from src.ec import *
# typed write register
from tasks.problem import Problem
from src.lgp.individual.primitive import WriteRegisterGPNode
from typing import override
# from tasks.ARC.DSL.arc_agent import ARCAgent as arc
from src.lgp.algorithm.typed_lgp.ec import *
import numpy as np
from src.lgp.algorithm.typed_lgp.individual.primitives.typed_interface import TypedInterface

class TypedWriteRegister(WriteRegisterGPNode, TypedInterface):

    def __init__(self, index=0, range_=None):
        WriteRegisterGPNode.__init__(self, index, range_)
        TypedInterface.__init__(self)
        

    def eval(self, state:EvolutionState, thread:int, input:TypedData, individual, 
             problem:Problem, argval: list[TypedData] = None):

        # collect arguments and type
        args = []
        tmp_type = None
        if argval is not None and len(argval) == self.expectedChildren():
            # If argval is provided, use it for evaluation
            args = argval 
            tmp_type = argval[0].type
        else:
            for i, child in enumerate(self.children):
                child.eval(state, thread, input, individual, problem)
                arg = input.value if not input.to_vectorize else input.values
                args.append(arg)
                tmp_type = input.type

        # set the type of WriteRegister (the input type is left None)
        self.output_type = tmp_type

        # set the type of input (because WriteRegister does not change the value of input, its type keeps the same)
        input.type = tmp_type

        # set the value and type into register
        register = individual.getRegistersIndex(self.index) # the same object

        register.setValue(args[0], tmp_type)

    def lightClone(self):
        obj:TypedWriteRegister = WriteRegisterGPNode.lightClone(self)
        obj.lightCloneInterface(self)
        return obj
    
    def clone(self):
        obj:TypedWriteRegister = WriteRegisterGPNode.clone(self)
        obj.lightCloneInterface(self)
        return obj
    
    def swapCompatibleWith(self, node:'TypedInterface', state:TypedEvolutionState=None) -> bool:
        res = GPNode.swapCompatibleWith(self, node) and TypedInterface.swapCompatibleWith(self, node, state)
        return res
