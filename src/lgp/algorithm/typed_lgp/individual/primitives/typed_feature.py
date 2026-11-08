from src.ec import *
from src.lgp.individual.primitive.readRegisterGPNode import ReadRegisterGPNode
from tasks.problem import Problem
from src.lgp.algorithm.typed_lgp.ec.typed_data import TypedData
from src.lgp.algorithm.typed_lgp.individual.primitives import TypedConstant
import numpy as np
# import tasks.ARC.DSL.constants as constants
# import tasks.ARC.DSL.arc_types as arctype
# import src.lgp.algorithm.typed_lgp.dsl.type_checker as tc
from src.lgp.algorithm.typed_lgp.individual.primitives.typed_interface import TypedInterface
from src.lgp.algorithm.typed_lgp.ec import *
from src.ec.util import *
from src.lgp.algorithm.typed_lgp.individual.primitives import TypedInput, TypedReadRegister
from src.lgp.algorithm.typed_lgp.dsl.agent import Agent
from typing import override, Callable
from collections.abc import Sized


class TypedFeature(GPNode, TypedInterface):

    # a special constant node fulfilled by simple DSL functions. it has no children. 
    # the only one input argument is fixed as the original input

    # color: most color, least color, possible color, color of the largest / smallest object
    # size: scaling size, object sizes
    # canvas:  input canvas height and width, coordinators of four corners
    # the most likely moving direction (a tuple, approximated by a linear model between objects with the same color in input and outputs)
    # input object: largest object (background as boundary), smallest object, the object with the most color, the object with the least color

    P_NUMREGISTERS = "numregisters"

    def __init__(self):
        GPNode.__init__(self)
        TypedInterface.__init__(self)
        self.function_name:str = None
        self.eval_function:Callable = None
        # self.input_arg:TypedInput = TypedInput()
        self.input_arg:GPNode = TypedInput()
        self.input_arg.parent = self
        self.input_arg.argposition = 0
        self.range:int = 1

    def __str__(self):
        return " ("+self.function_name + self.input_arg.printRootedTreeInString() + ")"

    def setup(self, state: TypedEvolutionState, base: Parameter):
        super().setup(state, base)
        self.agent:Agent = state.typeagent

        param_key = base.push(self.P_NUMREGISTERS)
        self.range = state.parameters.getInt(param_key, None)
        if self.range < 1:
            state.output.fatal("number of registers must be >=1")

    def expectedChildren(self):
        return 0

    def setProperties(self, name:str, function:Callable, input_arg:TypedInterface, input_types:list, output_type):
        self.function_name = name
        self.eval_function = function
        self.input_arg = input_arg
        self.input_types = input_types
        self.output_type = output_type

    def __eq__(self, other):
        return isinstance(other, TypedFeature) \
            and self.function_name == other.function_name \
            and self.eval_function == other.eval_function \
            and self.input_arg == other.input_arg \
            and self.input_types == other.input_types \
            and self.output_type == other.output_type \
            and self.agent == other.agent

    @override
    def eval(self, state:EvolutionState, thread:int, input:TypedData, individual, problem, argval: list = None):
        args = [] # non-callable arguments

        # if argval is not None and len(argval) == self.expectedChildren():
        #     # If argval is provided, use it for evaluation
        #     args.append(argval)
        # else:
        #     for i, child in enumerate(self.children):
        #         child.eval(state, thread, input, individual, problem)
        #         arg = input.value if not input.to_vectorize else input.values
        #         args.append(arg)

        if self.input_arg == None:
            raise Exception("input argument is not set for Typed Feature Node. Set it before eval()")
        
        self.input_arg.eval(state, thread, input, individual, problem)
        arg = input.value if not input.to_vectorize else input.values
        args.append(arg)
        
        if self.eval_function is None:
            raise Exception("eval_function is not set for TypedGPNode. Set it before eval()")

        if not input.to_vectorize: # scalar model, args are scalars
            result = self.evalfunction(self.eval_function, args, to_check_type=False)
            input.value = result
            input.type = self.output_type
        else: # vectorized model, args are container for multiple argument containers
            input.values = np.empty(len(args[0]), dtype=self.output_type)
            for i in range(len(input.values)):
                argi = [arg[i] for arg in args]
                input.values[i] = self.evalfunction(self.eval_function, argi)
                input.type = self.output_type

    # def __hash__(self):
    #     return hash(str(self))

    def resetNode(self, state: EvolutionState, thread: int, candidate:list=None):
        if candidate == None or len(candidate) == 0:
            # sample the function from all possible functions
            candidate = self.agent.getFeaturesWithOutput(None)

        map = state.random[thread].choice(candidate)

        regind = state.random[thread].randint(0, self.range - 1)

        self.setProperties(
            name=map['name'],
            function=map['obj'],
            input_arg=TypedReadRegister(regind, self.range) if state.random[thread].random()<0.5 else TypedInput(),
            input_types=list(map['input_type']),
            output_type=map['output_type']
        )

        self.input_arg.parent = self
        self.input_arg.argposition = 0
        
        self.children = []

    def lightClone(self) -> 'TypedFeature':
        obj:TypedFeature = GPNode.lightClone(self)
        obj.lightCloneInterface(self)
        obj.setProperties(
            name=self.function_name,
            function=self.eval_function,
            input_arg=self.input_arg.lightClone(),
            input_types=self.input_types,
            output_type=self.output_type
        )
        obj.input_arg.parent = obj
        obj.input_arg.argposition = 0
        obj.range = self.range
        return obj

    def clone(self) -> 'TypedFeature':
        obj:TypedFeature = GPNode.clone(self)
        obj.lightCloneInterface(self)
        obj.setProperties(
            name=self.function_name,
            function=self.eval_function,
            input_arg=self.input_arg.clone(),
            input_types=self.input_types,
            output_type=self.output_type
        )
        obj.input_arg.parent = obj
        obj.input_arg.argposition = 0
        return obj
    
    def swapCompatibleWith(self, node:'TypedInterface', state:TypedEvolutionState=None) -> bool:
        res = GPNode.swapCompatibleWith(self, node) and TypedInterface.swapCompatibleWith(self, node, state)
        return res
    
    def evalfunction(self, func: Callable, args: list, to_check_type:bool=True):

        # 1. check the type of all input arguments
        input_types = self.input_types 

        if to_check_type and (len(args) != len(input_types) or not all( self.agent.tc.check_type(value, typ) for value, typ in zip(args, input_types) )):
            for value, typ in zip(args, input_types):
                print(f"arg value: {value}, arg type: {type(value)}, expected type: {typ}")
                print(f"function name: {self.function_name}")
                print(f"input args: {self.input_arg}")
            raise Exception("eval_function gets inconsistent argument values and types")  

        # 2. execute the function
        res = func(*args[:]) 
        
        # 3. check the output type of the result
        if not self.agent.tc.check_type(res, self.output_type):
            raise Exception("eval_function outputs inconsistent results and types")  
        
        if isinstance(res, Sized) and len(res) == 0:
            raise Exception("eval_function outputs an empty container")  

        return res
    
    def collectInputTypes(self, s: set):
        # the input type of TypedFeature is determined by the output type of its input argument, so we also need to add it into the set of input types.
        if self.input_arg != None and isinstance(self.input_arg, TypedReadRegister):
            s.add(self.input_arg.getOutputType())
        super().collectInputTypes(s)

    def collectReadRegister(self, s: set[int]):
        # from src.lgp.individual.primitive import ReadRegisterGPNode
        # from src.lgp.algorithm.typed_lgp.individual.primitives.typed_feature import TypedFeature

        arg = self.input_arg
        if isinstance(arg, ReadRegisterGPNode):
            s.add(arg.index)

        # stack = [self]

        # while stack:
        #     node = stack.pop()

        #     arg = node.input_arg
        #     if isinstance(arg, ReadRegisterGPNode):
        #         s.add(arg.index)

        #     # Extend stack
        #     stack.extend(node.children)