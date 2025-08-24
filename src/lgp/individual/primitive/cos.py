from src.ec import * 
import math
from tasks.problem import Problem
from typing import override
import numpy as np

class Cos(GPNode):
    @override
    def expectedChildren(self):
        return 1

    @override
    def eval(self, state: EvolutionState, thread: int, input: GPData,
             individual, problem: Problem, argval: list[float] = None):
        
        if argval is not None and len(argval) == self.expectedChildren():
            # If argval is provided, use it for evaluation
            if not input.to_vectorize: 
                input.value = math.cos(argval[0])
            else:
                input.values = np.cos(argval[0])
        else:
            self.children[0].eval(state, thread, input, individual, problem)
            if not input.to_vectorize: 
                input.value = math.cos(input.value)
            else:
                input.values = np.cos(input.values)

        # if not input.to_vectorize:
        #     child_result = input
            
        #     self.children[0].eval(state, thread, child_result, individual, problem)
        #     input.value = math.cos(child_result.value)
        # else:
        #     self.children[0].eval(state, thread, input, individual, problem)
        #     input.values = np.cos(input.values)

    def __str__(self):
        return "cos"