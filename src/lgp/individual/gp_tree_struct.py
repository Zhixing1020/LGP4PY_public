from src.ec.gp_tree import GPTree
from src.ec.gp_node import GPNode
from src.ec.gp_data import GPData
from src.ec.evolution_state import EvolutionState
from tasks.problem import Problem

from typing import Set, List
# from copy import deepcopy

class GPTreeStruct(GPTree):
    ARITHMETIC = 0
    BRANCHING = 1
    ITERATION = 2

    # execution_list:list[GPNode] = None

    def __init__(self):
        super().__init__()
        self.status: bool = False  # False: non-effective, True: effective
        self.effRegisters: Set[int] = set()
        self.type: int = GPTreeStruct.ARITHMETIC  # default: ARITHMETIC

    def __repr__(self):
        address = hex(id(self))
        return f"{str(self)}<GPTreeStruct {address} status={self.status}>"


    def updateEffRegister(self, s: Set[int]):
        self.child.collectReadRegister(s)

    def collectReadRegister(self) -> Set[int]:
        s = set()
        self.child.collectReadRegister(s)
        return s

    # def collectReadRegister_list(self) -> List[int]:
    #     l = []
    #     self.child.collectReadRegister_list(l)
    #     return l

    def clone(self) -> 'GPTreeStruct':
        t = self.lightClone()
        t.child = self.child.clone()
        t.child.parent = t
        t.child.argposition = 0
        
        # t.execution_list = None  # Reset execution list for the new tree
        # if self.execution_list is not None:
        #     t.execution_list = [node.lightClone() for node in self.execution_list]  # Reset execution list for the new tree
        return t

    def lightClone(self) -> 'GPTreeStruct':
        t = super().lightClone()
        t.status = self.status
        t.type = self.type
        t.effRegisters = set(self.effRegisters)
        return t

    def assignfrom(self, tree: 'GPTree'):
        self.child = tree.child
        self.owner = tree.owner

    # def flatten_postorder(self, state:EvolutionState, thread:int) -> list[GPNode]:
    #     """
    #     Traverse the tree in postorder and return a flat list of nodes
    #     in the order they should be executed.
    #     """
    #     root = self.child

    #     self.execution_list = []
    #     stack = [(root, False)]  # (node, visited)

    #     while stack:
    #         node, visited = stack.pop()
    #         if visited:
    #             self.execution_list.append(node)  # append the node itself
    #         else:
    #             stack.append((node, True))  # mark node to be added after children
    #             for child in reversed(node.children):  # maintain left-to-right order
    #                 stack.append((child, False))

    #     return self.execution_list
    
    # def postorder_execution(self, state:EvolutionState, thread:int, 
    #                         input:GPData, individual, problem:Problem):
    #     if self.execution_list is None:
    #         self.flatten_postorder(state, thread, self.child)
        
    #     tmp_result = []

    #     def self_pop(A:list[None]):
    #         if A:
    #             return A.pop()
    #         else:
    #             return None
        
    #     for node in self.execution_list:
    #         argval = [self_pop(tmp_result) for _ in range(node.expectedChildren())]
    #         node.eval(state, thread, input, individual, problem, argval)
    #         tmp_result.append(input.value if not input.to_vectorize else input.values)

    #     if not input.to_vectorize:
    #         input.value = tmp_result[-1]
    #     else:
    #         input.values = tmp_result[-1]

        

