
from tasks.ARC.DSL.arc_types import Callable
from src.lgp.algorithm.typed_lgp.dsl.agent import Agent
from src.lgp.algorithm.typed_lgp.ec import *

class TypedInterface:
    # all possible GP Node should inherit from this interface, specifying the input and output types

    def __init__(self):
        self.input_types:tuple = None
        self.output_type:type = None
        self.agent:Agent = None

    def getInputType(self, no_callable:bool=False)-> tuple:
        types = self.input_types
        if types is None or not no_callable or Callable not in types:
            return types
        
        return tuple(t for t in types if t is not Callable)

    def getOutputType(self):
        return self.output_type
    
    def setInputType(self, types):
        self.input_types = tuple(types)
    
    def setOutputType(self, type):
        self.output_type = type

    def collectInputTypes(self, s: set):
        # from src.lgp.algorithm.typed_lgp.individual.primitives.typed_readregister import TypedReadRegister
        # if isinstance(self, TypedReadRegister):
        #     s.add(self.getOutputType())  # Assumes getIndex() method exists in ReadRegisterGPNode

        # from src.lgp.algorithm.typed_lgp.individual.primitives.typed_feature import TypedFeature
        # if isinstance(self, TypedFeature) and isinstance(self.input_arg, TypedReadRegister):
        #     s.add(self.input_arg.getOutputType()) 

        for child in self.children:
            child.collectInputTypes(s)

    def lightCloneInterface(self, obj:'TypedInterface'):
        self.input_types = obj.input_types # if obj.input_types is not None else None
        self.output_type = obj.output_type
        self.agent = obj.agent

    def swapCompatibleWith(self, node:'TypedInterface', state: TypedEvolutionState=None) -> bool:
        tc = self.agent.tc if self.agent != None else state.typeagent.tc
        if tc is None:
            raise Exception("there is no type checker in swapCompatibleWith of TypedInterface")
        
        res = tc.type_equal(self.output_type, node.output_type) 
        
        if self.input_types != None and node.input_types != None:
            res = res and (len(self.input_types) == len(node.input_types) \
            and all(tc.type_equal(self.input_types[i], node.input_types[i]) for i in range(len(self.input_types))))
        else:
            res = res and (
                self.input_types == None and node.input_types == None 
                or self.expectedChildren() == node.expectedChildren() == 0
            )

        return res