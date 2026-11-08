from src.lgp.individual.gp_tree_struct import GPTreeStruct
from src.lgp.algorithm.typed_lgp.ec import *
from copy import deepcopy

class TypedGPTreeStruct(GPTreeStruct):
    __slots__ = ("effTypes", "effInputTypes", "effInputRegister")
    def __init__(self):
        super().__init__()
        self.effTypes:set = set()

        # self.forwardStatus:bool = False  # the effectiveness by inspecting the input types of the instructions
        self.effInputTypes:set = set()  # the effective types for registers (input type + manipulated types)
        self.effInputRegister:dict = dict()  # the effective indices of possible types  {type : [possible indices]}

    def updateEffType(self, s:set):
        self.child.collectInputTypes(s)

    def clone(self) -> 'TypedGPTreeStruct':
        t:TypedGPTreeStruct = super().clone()

        # t.forwardStatus = self.forwardStatus
        t.effTypes = set(self.effTypes)
        t.effInputTypes = set(self.effInputTypes)
        t.effInputRegister = dict(self.effInputRegister)
        return t
    
    def lightClone(self) -> 'TypedGPTreeStruct':
        t:TypedGPTreeStruct = super().lightClone()
        
        t.effTypes = set(self.effTypes)
        t.effInputTypes = set(self.effInputTypes)
        t.effInputRegister = dict(self.effInputRegister)
        # t.forwardStatus = self.forwardStatus
        return t
    

    def buildTree(self, state:TypedEvolutionState, thread:int, targetType=None, sourceType=None):
        if not isinstance(state.builder, Typed_LGPBuilder):
            raise Exception(f"the current builder {state.builder.__class__.__name__} is not {Typed_LGPBuilder}")
        
        # construct a GPTreeStruct returning the given target type
        self.child = state.builder.newRootedTree(state, 0, self, self.species.primitiveset, 0, 
                                                 expect_type=targetType, given_type=sourceType)
        
    def setReadRegisterOnTypes(self, n, state:TypedEvolutionState, thread:int):
        """find all read registers. If their output types are not consistent with effInputRegister, 
        reset their index. n: GPNode (the root of the tree)"""
        from src.lgp.algorithm.typed_lgp.individual.primitives import TypedReadRegister
        def _recursive_set(n):
            if isinstance(n, TypedReadRegister):
                tc = state.typeagent.tc
                if any(tc.type_equal(n.getOutputType(), t) for t in self.effInputRegister.keys()):
                    ty = next((t for t in self.effInputRegister.keys() if tc.type_equal(n.getOutputType(), t)), None)

                    if n.getIndex() not in self.effInputRegister[ty]:
                        n.setIndex(state.random[thread].choice(list(self.effInputRegister[ty])))

            else:
                for child in n.children:
                    _recursive_set(child)

        _recursive_set(n)
