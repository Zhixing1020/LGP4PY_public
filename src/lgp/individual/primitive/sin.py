from src.ec import * 
import math
from tasks.problem import Problem
from typing import override
import numpy as np

class Sin(GPNode):

    @override
    def expectedChildren(self):
        return 1

    @override
    def eval(self, state: EvolutionState, thread: int, input: GPData,
             individual, problem: Problem, argval: list[float] = None):
        if argval is not None and len(argval) == self.expectedChildren():
            # If argval is provided, use it for evaluation
            if not input.to_vectorize:
                input.value = math.sin(argval[0])
            else:
                input.values = np.sin(argval[0])
        else:
            self.children[0].eval(state, thread, input, individual, problem)
            if not input.to_vectorize:
                input.value = math.sin(input.value)
            else:
                input.values = np.sin(input.values)

        # if not input.to_vectorize:
        #     child_result = input
            
        #     self.children[0].eval(state, thread, child_result, individual, problem)
        #     input.value = math.sin(child_result.value)
        # else:
        #     self.children[0].eval(state, thread, input, individual, problem)
        #     input.values = np.sin(input.values)

    def __str__(self):
        return "sin"