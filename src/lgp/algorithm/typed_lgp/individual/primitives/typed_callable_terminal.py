# strongly typed callable function node. This node does not execute the function but return the function itself

from typing import Callable, override
from collections.abc import Container as runtimeContainer
from collections.abc import Sized
from src.ec import *
from src.ec.util import *
from tasks.problem import Problem
from src.lgp.algorithm.typed_lgp.dsl.agent import Agent
# import src.lgp.algorithm.typed_lgp.dsl.type_checker as tc
import numpy as np
from src.lgp.algorithm.typed_lgp.individual.primitives.typed_function import TypedFunction
from src.lgp.algorithm.typed_lgp.ec import *

class TypedCallableTerminal(TypedFunction):
    P_TYPEDCALLABLE = "typed_callable_terminal"

    def printRootedTreeInString(self) -> str:
        return str(self)
    
    def expectedChildren(self)->int:
        return 0
    
    def setProperties(self, name:str, function:Callable, child_fun:Callable, input_types:list, output_type):
        self.function_name = name
        self.eval_function = function
        self.child_func = child_fun
        self.input_types = input_types
        self.output_type = Callable[[tuple(input_types) if input_types else None], output_type] 
    
    def __eq__(self, other):
        return isinstance(other, TypedCallableTerminal) \
            and self.function_name == other.function_name \
            and self.eval_function == other.eval_function \
            and self.child_func == other.child_func \
            and self.input_types == other.input_types \
            and self.output_type == other.output_type
    
    def lightClone(self) -> 'TypedCallableTerminal':
        obj:TypedCallableTerminal = GPNode.lightClone(self)
        # obj.setfunction(self.function_name, self.child_func, self.type_T['input'])
        obj.lightCloneInterface(self)
        obj.setProperties(name=self.function_name,
                          function=self.eval_function,
                          child_fun=self.child_func,
                          input_types=self.input_types,
                          output_type=self.output_type
                          )
        return obj

    def clone(self) -> 'TypedCallableTerminal':
        obj:TypedCallableTerminal = GPNode.clone(self)
        # obj.setfunction(self.function_name, self.child_func, self.type_T['input'])
        obj.lightCloneInterface(self)
        obj.setProperties(name=self.function_name,
                          function=self.eval_function,
                          child_fun=self.child_func,
                          input_types=self.input_types,
                          output_type=self.output_type
                          )
        return obj
    
    @override
    def eval(self, state: TypedEvolutionState, thread: int, input: TypedData,
             individual, problem: Problem, argval: list = None):
        
        args = [] # non-callable arguments

        if argval is not None and len(argval) == self.expectedChildren():
            # If argval is provided, use it for evaluation
            args.append(argval)
        else:
            for i, child in enumerate(self.children):
                child.eval(state, thread, input, individual, problem)
                arg = input.value if not input.to_vectorize else input.values
                args.append(arg)
        
        # if self.eval_function is None:
        #     raise Exception("eval_function is not set for TypedGPNode. Set it before eval()")

        if not input.to_vectorize: # scalar model, args are scalars
            input.value = self.eval_function
            input.type = self.output_type
        else: # vectorized model, args are container for multiple argument containers
            # input.values = np.empty((len(args[0]), 1), dtype=self.output_type)
            # arg_list = [arg for arg in args]
            input.values = [self.eval_function] * len(args)
            input.type = self.output_type
            # for i in range(len(input.values)):
            #     argi = [arg[i] for arg in args]
            #     input.values[i] = self.evalfunction(self.eval_function, argi, self.child_func, to_check_type=False)
            #     input.type = self.output_type