# strongly typed function GPNode for ARC

# each node has a function name a type signature (input types, output type). The type is defined in arc_types.py

# a typed GPNode executes the function by getattr(dsl_module, fun)(inputs)

from typing import Any, Union, Container, Callable, Optional, override, get_origin, get_args, get_type_hints
from collections.abc import Container as runtimeContainer
from collections.abc import Sized
from src.ec import *
from src.ec.util import *
from tasks.problem import Problem
from src.lgp.algorithm.typed_lgp.dsl.agent import Agent
# import src.lgp.algorithm.typed_lgp.dsl.type_checker as tc
import numpy as np
from src.lgp.algorithm.typed_lgp.individual.primitives.typed_interface import TypedInterface
from src.lgp.algorithm.typed_lgp.ec import *

class TypedFunction(GPNode, TypedInterface):

    P_TYPEDFUNCTION = "typed_function"
    # P_ADD_TYPES_PROB = "add_type_prob" # the probability of increasing effective types
    # P_CHILD_FUN_PROB = "child_fun_prob" # the probability of using functions with child_function
    # add_type_prob = 0.67
    # child_fun_prob = 0.1

    def __init__(self):
        GPNode.__init__(self)
        TypedInterface.__init__(self)
        self.function_name:str = None
        self.eval_function:Callable = None
        self.child_func = None # the callable argument

    def __str__(self):
        return self.function_name
    
    def printRootedTreeInString(self) -> str:
        res = " (" if len(self.children) > 0 else " "
        res += str(self)
        if self.child_func != None:
            res += f"[{self.child_func.__name__}]"
        for child in self.children:
            res += child.printRootedTreeInString()
        if len(self.children) > 0:
            res += ")"
        return res

    def expectedChildren(self)->int:
    #     return len(self.input_types) - (self.input_types[0] == Callable) if self.input_types is not None else self.CHILDREN_UNKNOWN
        return len(self.input_types) if self.input_types is not None else self.CHILDREN_UNKNOWN
    
    @classmethod
    def defaultBase(cls) -> Parameter:
        return Parameter(cls.P_TYPEDFUNCTION)

    def setup(self, state:TypedEvolutionState, base):
        # if self.function_name is None:
        #     raise Exception("function_name is not set for TypedGPNode. Set it before setup()")
        super().setup(state, base)
        self.agent = state.typeagent

        def_param = self.defaultBase()

        # Fetch the constraint ID string
        # s = state.parameters.getString(base.push(self.P_NODECONSTRAINTS),
        #                             def_param.push(self.P_NODECONSTRAINTS))
        # self.add_type_prob = state.parameters.getDoubleWithDefault(base.push(self.P_ADD_TYPES_PROB),
        #                                                            def_param.push(self.P_ADD_TYPES_PROB),
        #                                                            0.67)
        # if self.add_type_prob < 0 or self.add_type_prob > 1:
        #     raise ValueError(f"The add_type_prob in TypedFunction cannot be < 0 or > 1")
        
        # self.child_fun_prob = state.parameters.getDoubleWithDefault(base.push(self.P_CHILD_FUN_PROB),
        #                                                            def_param.push(self.P_CHILD_FUN_PROB),
        #                                                            0.1)
        # if self.child_fun_prob < 0 or self.child_fun_prob > 1:
        #     raise ValueError(f"The child_fun_prob in TypedFunction cannot be < 0 or > 1")
        return 
    
    def setProperties(self, name:str, function:Callable, child_fun:Callable, input_types:list, output_type):
        self.function_name = name
        self.eval_function = function
        self.child_func = child_fun
        self.input_types = input_types
        self.output_type = output_type
    
    def __eq__(self, other):
        return isinstance(other, TypedFunction) \
            and self.function_name == other.function_name \
            and self.eval_function == other.eval_function \
            and self.child_func == other.child_func \
            and self.input_types == other.input_types \
            and self.output_type == other.output_type
    
    def lightClone(self) -> 'TypedFunction':
        obj:TypedFunction = GPNode.lightClone(self)
        # obj.setfunction(self.function_name, self.child_func, self.type_T['input'])
        obj.lightCloneInterface(self)
        obj.setProperties(name=self.function_name,
                          function=self.eval_function,
                          child_fun=self.child_func,
                          input_types=self.input_types,
                          output_type=self.output_type
                          )
        return obj

    def clone(self) -> 'TypedFunction':
        obj:TypedFunction = GPNode.clone(self)
        # obj.setfunction(self.function_name, self.child_func, self.type_T['input'])
        obj.lightCloneInterface(self)
        obj.setProperties(name=self.function_name,
                          function=self.eval_function,
                          child_fun=self.child_func,
                          input_types=self.input_types,
                          output_type=self.output_type
                          )
        return obj


    def resetNode(self, state: EvolutionState, thread: int, candidate:list=None):
        if candidate == None or len(candidate) == 0:
            # sample the function from all possible functions
            candidate = self.agent.getFunctionListWithOutput(None, None, state, thread)

        # to_add_type = state.random[thread].random() < self.add_type_prob
        # to_have_child_fun = state.random[thread].random() < self.child_fun_prob

        # def get_num_unique_type(input_types)->int:
        #     res = 0
        #     for ty in input_types:
        #         for tj in input_types:
        #             if ty != tj and not self.agent.tc.type_equal(ty, tj):
        #                 res += 1
        #     return res
        
        def random_choose_fun(candidate:list):
            """candidate: a list of function maps. This function returns a randomly sampling map.
            To reduce the bias caused by generic types (a function name has a lot of implementation maps),
            we first collect the function name, and sample based on the function name
            """
            names = set(can['name'] for can in candidate)

            name = state.random[thread].choice(list(names))

            filter_cand = [can for can in candidate if can['name'] == name]

            return state.random[thread].choice(filter_cand)


        # tmp_can = []
        # for can in candidate:
        #     num_callable = sum(self.agent.tc.type_equal(x, Callable) for x in can['input_type'])
        #     if not to_have_child_fun and num_callable > 0:
        #         continue
        #     if to_add_type and get_num_unique_type(can['input_type']) - num_callable <= 1:
        #         continue
        #     tmp_can.append(can)

        # if len(tmp_can) > 0:
        #     candidate = tmp_can   

        map = random_choose_fun(candidate)
        self.setProperties(
            name=map['name'],
            function=map['obj'],
            child_fun=map['child_fun'],
            input_types=list(map['input_type']),
            output_type=map['output_type']
        )

        length = self.expectedChildren()
        
        if length == 0:
            self.children = []
        else:
            self.children = [None] * length 


    # def swapCompatibleWith(self, node) -> bool:
    #     if not isinstance(node, TypedFunction):
    #         return False
    #     res:bool = super().swapCompatibleWith(node)
    #     res = res and tc.type_equal(self.output_type, node.output_type) and tc.type_equal(self.input_types, node.input_types)
    #     return res
    
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
            result = self.evalfunction(self.eval_function, args, self.child_func, to_check_type=False)
            input.value = result
            input.type = self.output_type
        else: # vectorized model, args are container for multiple argument containers
            # input.values = np.empty((len(args[0]), 1), dtype=self.output_type)
            arg_list = [arg for arg in args]
            input.values = self.evalfunction(self.eval_function, arg_list, self.child_func, to_check_type=False)
            input.type = self.output_type
            # for i in range(len(input.values)):
            #     argi = [arg[i] for arg in args]
            #     input.values[i] = self.evalfunction(self.eval_function, argi, self.child_func, to_check_type=False)
            #     input.type = self.output_type



    def evalfunction(self, func: Callable, args: list, childfunc:Callable = None, to_check_type:bool=True):

        # 1. check the type of all input arguments
        input_types = self.input_types if childfunc is None else self.input_types[1:]

        if to_check_type and (len(args) != len(input_types) or not all( self.agent.tc.check_type(value, typ) for value, typ in zip(args, input_types) ) ):
            raise Exception("eval_function gets inconsistent argument values and types")  

        # 2. execute the function
        res = func(*args[:]) if childfunc is None else func(childfunc, *args[:])
        
        # 3. check the output type of the result
        if to_check_type and not self.agent.tc.check_type(res, self.output_type):
            raise Exception("eval_function outputs inconsistent results and types")  
        
        if isinstance(res, Sized) and len(res) == 0:
            raise Exception("eval_function outputs an empty container")  

        return res
    
    def swapCompatibleWith(self, node:'TypedInterface', state:TypedEvolutionState=None) -> bool:
        res = GPNode.swapCompatibleWith(self, node) and TypedInterface.swapCompatibleWith(self, node, state)
        return res
