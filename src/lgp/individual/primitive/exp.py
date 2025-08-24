from src.ec import * 
import math
from tasks.problem import Problem
from typing import override
import numpy as np

class Exp(GPNode):

    @override
    def expectedChildren(self):
        return 1

    @override
    def eval(self, state: EvolutionState, thread: int, input: GPData,
             individual, problem: Problem, argval: list[float] = None):
    
        if argval is not None and len(argval) == self.expectedChildren():
            # If argval is provided, use it for evaluation
            if not input.to_vectorize:
                argval[0] = min(argval[0], 10)  # Clip to 10
                result = math.exp(argval[0])
            else:
                argval[0] = np.clip(argval[0], None, 10)  # Clip to 10
                result = np.exp(argval[0])
        else:
            self.children[0].eval(state, thread, input, individual, problem)
            if not input.to_vectorize:
                result = math.exp(min(input.value, 10)) 
            else:
                result = np.exp(np.clip(input.values, None, 10))

        # Clip the result to ±1e6
        if not input.to_vectorize:
            input.value = max(-1e6, min(1e6, result))
        else:
            input.values = np.clip(result, -1e6, 1e6)

        # if not input.to_vectorize:
        #     child_result = input
            
        #     self.children[0].eval(state, thread, child_result, individual, problem)
        #     result = child_result.value
        #     if child_result.value > 10:
        #         result = 10
            
        #     input.value = math.exp(result)
        #     if input.value > 1e6:
        #         input.value = 1e6
        # else:
        #     self.children[0].eval(state, thread, input, individual, problem)
        #     result = input.values
        #     np.clip(result, None, 10, out=result)
        #     input.values = np.exp(result)
        #     np.clip(input.values, -1e6, 1e6, out=input.values)

    def __str__(self):
        return "exp"