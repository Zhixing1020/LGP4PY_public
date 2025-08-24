from src.ec import *
from tasks.problem import Problem
from typing import override
import numpy as np
# from src.ec.gp_node import GPNode
# from src.ec.gp_data import GPData

class Sub(GPNode):
    
    def __str__(self):
        return "-"
    
    @override
    def expectedChildren(self)->int:
        return 2
    
    @override
    def eval(self, state: EvolutionState, thread: int, input: GPData,
             individual, problem: Problem, argval: list[float] = None):
        
        if argval is not None and len(argval) == self.expectedChildren():
            # If argval is provided, use it for evaluation
            result = argval[0] - argval[1]
        else:
            self.children[0].eval(state, thread, input, individual, problem)
            arg1 = input.value if not input.to_vectorize else input.values
            self.children[1].eval(state, thread, input, individual, problem)
            arg2 = input.value if not input.to_vectorize else input.values
            result = arg1 - arg2
            
        # Clip the result to ±1e6   
        if not input.to_vectorize:
            input.value = max(-1e6, min(1e6, result))
        else:
            input.values = np.clip(result, -1e6, 1e6)

        # if not input.to_vectorize:
        #     child_result = input
            
        #     self.children[0].eval(state, thread, child_result, individual, problem)
        #     result = child_result.value

        #     self.children[1].eval(state, thread, child_result, individual, problem)
        #     child_result.value = result - child_result.value

        #     # Clip the result to ±1e6
        #     if child_result.value > 1e6:
        #         child_result.value = 1e6
        #     elif child_result.value < -1e6:
        #         child_result.value = -1e6
        # else:
        #     self.children[0].eval(state, thread, input, individual, problem)
        #     result = input.values

        #     self.children[1].eval(state, thread, input, individual, problem)
        #     input.values = result - input.values
        #     input.values = np.clip(input.values, -1e6, 1e6)


        # input.value = child_result.value