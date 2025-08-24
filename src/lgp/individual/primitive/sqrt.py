from src.ec import * 
import math
from tasks.problem import Problem
from typing import override
import numpy as np

class Sqrt(GPNode):

    @override
    def expectedChildren(self):
        return 1

    @override
    def eval(self, state: EvolutionState, thread: int, input: GPData,
             individual, problem: Problem, argval: list[float] = None):
        if argval is not None and len(argval) == self.expectedChildren():
            # If argval is provided, use it for evaluation
            if not input.to_vectorize:
                result = math.sqrt(abs(argval[0]))
            else:
                result = np.sqrt(np.abs(argval[0]))
        else:
            self.children[0].eval(state, thread, input, individual, problem)
            if not input.to_vectorize:
                result = math.sqrt(abs(input.value))
            else:
                result = np.sqrt(np.abs(input.values))
                
        # Clip the result to ±1e6
        if not input.to_vectorize:
            input.value = max(-1e6, min(1e6, result))
        else:
            input.values = np.clip(result, -1e6, 1e6)

            
        # if not input.to_vectorize:
        #     child_result = input
            
        #     self.children[0].eval(state, thread, child_result, individual, problem)
        #     result = child_result.value
            
        #     input.value = math.sqrt( abs(result))
        # else:
        #     self.children[0].eval(state, thread, input, individual, problem)
        #     result = input.values
        #     input.values = np.sqrt(np.abs(result))

    def __str__(self):
        return "sqr"