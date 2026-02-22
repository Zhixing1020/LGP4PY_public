from abc import abstractmethod
from typing import List, Set

from src.ec.util.parameter import Parameter
from src.ec.util.parameter_database import ParameterDatabase

from src.ec.evolution_state import EvolutionState 
from src.ec.gp_defaults import GPDefaults
from src.ec.gp_data import GPData
from src.ec.gp_node_parent import GPNodeParent

from tasks.problem import Problem
import numpy as np
# import copy

class GPNodeGather:
        def __init__(self):
            self.node:GPNode = None

class GPNode(GPNodeParent):

    '''define the functions and variables in every GP node'''
    
    __slots__ = ("children", "parent", "argposition")

    P_NODE: str = "node"
    P_NODECONSTRAINTS: str = "nc"
    GPNODEPRINTTAB: str = "    "
    MAXPRINTBYTES: int = 40

    NODESEARCH_ALL: int = 0
    NODESEARCH_TERMINALS: int = 1
    NODESEARCH_NONTERMINALS: int = 2
    NODESEARCH_CONSTANT: int = 5  # zhixing, LGP for JSS, 2021.4.10
    NODESEARCH_READREG: int = 4   # zhixing, LGP for JSS, 2021.4.10
    NODESEARCH_NULL: int = 6      # zhixing, LGP for JSS, 2021.4.10

    _NODESEARCH_CUSTOM: int = 3  # equivalent to package-private or private

    CHILDREN_UNKNOWN: int = -1

    def __init__(self):
        super()
        # the parent of this GP node
        self.parent:GPNode = None

        # the children of this GP node
        self.children:List[GPNode] = []

        # the constraints of this GP node, specifying how many children it has
        #self.constraints:int = None

        # the argument index of this GPNode
        self.argposition:int = -1

    @classmethod
    def defaultBase(cls) -> Parameter:
        return GPDefaults.base().push(cls.P_NODE)
    
    @property
    def expectedChildren(self)->int:
        return self.CHILDREN_UNKNOWN
    
    # def checkConstraints(self:GPNode, state: EvolutionState, tree: int,
    #                  typicalIndividual: GPIndividual, individualBase: Parameter):
    
    #     numChildren = self.expectedChildren()

    #     if numChildren >= 0 and len(self.children) != numChildren:
    #         state.output.error(
    #             f"Incorrect number of children for node {self.toStringForError()} at {individualBase}, "
    #             f"was expecting {numChildren} but got {len(self.children)}"
    #         )

    def setup(self, state: EvolutionState, base: Parameter):
        def_param = self.defaultBase()

        # Fetch the constraint ID string
        # s = state.parameters.getString(base.push(self.P_NODECONSTRAINTS),
        #                             def_param.push(self.P_NODECONSTRAINTS))

        # if s is None:
        #     state.output.fatal(
        #         f"No node constraints are defined for the GPNode {self.toStringForError()}",
        #         base.push(self.P_NODECONSTRAINTS),
        #         def_param.push(self.P_NODECONSTRAINTS)
        #     )
        # else:
        #     self.constraints = GPNodeConstraints.constraintsFor(s, state).constraintNumber

        # # Get the constraint object and determine children size
        # constraintsObj = self.constraintsObject(state.initializer)  # must implement this method

        length = self.expectedChildren()
        
        if length == 0:
            self.children = []
        else:
            self.children = [None] * length  # or a list of GPNode()

        self.parent = None

    def swapCompatibleWith(self, node) -> bool:
        # Fast check for atomic compatibility
        # if self.constraints(initializer).returntype == node.constraints(initializer).returntype:
        #     return True

        # # Set compatibility based on the parent
        # if isinstance(node.parent, GPNode):
        #     type_ = node.parent.constraints(initializer).childtypes[node.argposition]
        # else:  # GPTree root
        #     type_ = node.parent.constraints(initializer).treetype
        # if isinstance(node, GPNode):

        #     return self.expectedChildren() == node.expectedChildren()
        # else:
        #     return False
        return (self.expectedChildren() == node.expectedChildren()) if isinstance(node, GPNode) else False

    # def numNodes_with_gatherer(self, g: 'GPNodeGatherer') -> int:
    #     s = 0
    #     for child in self.children:
    #         s += child.numNodes_with_gatherer(g)
    #     return s + (1 if g.test(self) else 0)

    # def numNodes(self, nodesearch: int) -> int:
    #     from src.lgp.individual.primitive import WriteRegisterGPNode, ReadRegisterGPNode
    #     s = 0
    #     for child in self.children:
    #         if child is not None:
    #             s += child.numNodes(nodesearch)
    #         elif nodesearch == self.NODESEARCH_NULL:
    #             s += 1

    #     is_terminal = len(self.children) == 0
    #     is_nonterminal = len(self.children) > 0 and not isinstance(self, WriteRegisterGPNode)
    #     is_constant = len(self.children) == 0 and not isinstance(self, ReadRegisterGPNode)
    #     is_readreg = isinstance(self, ReadRegisterGPNode)

    #     include = (
    #         nodesearch == self.NODESEARCH_ALL or
    #         (nodesearch == self.NODESEARCH_TERMINALS and is_terminal) or
    #         (nodesearch == self.NODESEARCH_NONTERMINALS and is_nonterminal) or
    #         (nodesearch == self.NODESEARCH_CONSTANT and is_constant) or
    #         (nodesearch == self.NODESEARCH_READREG and is_readreg)
    #     )

    #     return s + (1 if include else 0)
    def numNodes(self, nodesearch: int) -> int:
        from src.lgp.individual.primitive import WriteRegisterGPNode, ReadRegisterGPNode

        count = 0
        stack = [self]

        while stack:
            node = stack.pop()

            children = node.children
            has_children = bool(children)

            # Push children
            for c in children:
                if c is not None:
                    stack.append(c)
                elif nodesearch == node.NODESEARCH_NULL:
                    count += 1

            # Classification (compute once)
            is_terminal = not has_children
            is_nonterminal = has_children and not isinstance(node, WriteRegisterGPNode)
            is_constant = is_terminal and not isinstance(node, ReadRegisterGPNode)
            is_readreg = isinstance(node, ReadRegisterGPNode)

            if (
                nodesearch == node.NODESEARCH_ALL or
                (nodesearch == node.NODESEARCH_TERMINALS and is_terminal) or
                (nodesearch == node.NODESEARCH_NONTERMINALS and is_nonterminal) or
                (nodesearch == node.NODESEARCH_CONSTANT and is_constant) or
                (nodesearch == node.NODESEARCH_READREG and is_readreg)
            ):
                count += 1

        return count

    def depth(self) -> int:
        d = 0
        for child in self.children:
            newdepth = child.depth()
            if newdepth > d:
                d = newdepth
        return d + 1
    
    def atDepth(self) -> int:
        cparent = self.parent
        count = 0
        while cparent is not None and isinstance(cparent, GPNode):
            count += 1
            cparent = cparent.parent
        return count
    
    
    # def nodeInPosition(self, p: int, g: GPNodeGather, nodesearch: int) -> int:
    #     from src.lgp.individual.primitive import WriteRegisterGPNode, ReadRegisterGPNode
    #     if (nodesearch == self.NODESEARCH_ALL or
    #         (nodesearch == self.NODESEARCH_TERMINALS and len(self.children) == 0) or
    #         (nodesearch == self.NODESEARCH_NONTERMINALS and len(self.children) > 0 and not isinstance(self, WriteRegisterGPNode)) or
    #         (nodesearch == self.NODESEARCH_CONSTANT and len(self.children) == 0 and not isinstance(self, ReadRegisterGPNode)) or
    #         (nodesearch == self.NODESEARCH_READREG and isinstance(self, ReadRegisterGPNode))):

    #         if p == 0:
    #             g.node = self
    #             return -1
    #         else:
    #             p -= 1

    #     for child in self.children:
    #         p = child.nodeInPosition(p, g, nodesearch)
    #         if p == -1:
    #             return -1

    #     return p
    def nodeInPosition(self, p: int, g: GPNodeGather, nodesearch: int) -> int:
        from src.lgp.individual.primitive import WriteRegisterGPNode, ReadRegisterGPNode

        stack = [self]

        while stack:
            node = stack.pop()

            children = node.children
            has_children = bool(children)

            include = (
                nodesearch == node.NODESEARCH_ALL or
                (nodesearch == node.NODESEARCH_TERMINALS and not has_children) or
                (nodesearch == node.NODESEARCH_NONTERMINALS and has_children and not isinstance(node, WriteRegisterGPNode)) or
                (nodesearch == node.NODESEARCH_CONSTANT and not has_children and not isinstance(node, ReadRegisterGPNode)) or
                (nodesearch == node.NODESEARCH_READREG and isinstance(node, ReadRegisterGPNode))
            )

            if include:
                if p == 0:
                    g.node = node
                    return -1
                p -= 1

            # Reverse to preserve original traversal order
            for c in reversed(children):
                stack.append(c)

        return p
    
    def rootParent(self) -> 'GPNodeParent':
        cparent = self
        while cparent is not None and isinstance(cparent, GPNode):
            cparent = cparent.parent
        return cparent

    def contains(self, subnode) -> bool:
        if subnode is self:
            return True
        for child in self.children:
            if child.contains(subnode):
                return True
        return False

    def resetNode(self, state: EvolutionState, thread: int):
        pass

    def lightClone(self) -> 'GPNode':
        obj:GPNode = super().clone()
        # obj = self.__class__()
        if len(self.children) == 0:
            obj.children = self.children  # share array (assumed to be reused zeroChildren)
        else:
            obj.children = [None] * len(self.children)
        obj.parent = None # the parent and argposition are set to None (or initial values) to imply a new node
        obj.argposition = -1
        return obj

    def clone(self) -> 'GPNode':
        newnode = self.lightClone()
        for x in range(len(self.children)):
            newnode.children[x] = self.children[x].clone()
            newnode.children[x].parent = newnode
            newnode.children[x].argposition = x
        return newnode
    
    def cloneReplacing(self, newSubtree: 'GPNode', oldSubtree: 'GPNode') -> 'GPNode':
        ''' Deep-clones the tree rooted at this node, and returns the entire
        copied tree.  If the node oldSubtree is located somewhere in this
        tree, then its subtree is replaced with a deep-cloned copy of
        newSubtree.  The result has everything set except for the root
        node's parent and argposition. '''
        if self is oldSubtree:
            return newSubtree.clone()
        else:
            newnode = self.lightClone()
            for x in range(len(self.children)):
                newnode.children[x] = self.children[x].cloneReplacing(newSubtree, oldSubtree)
                newnode.children[x].parent = newnode
                newnode.children[x].argposition = x
            return newnode

    def cloneReplacingNoSubclone(self, newSubtree: 'GPNode', oldSubtree: 'GPNode') -> 'GPNode':
        '''
        Deep-clones the tree rooted at this node, and returns the entire
        copied tree.  If the node oldSubtree is located somewhere in this
        tree, then its subtree is replaced with
        newSubtree (<i>not</i> a copy of newSubtree).  
        The result has everything set except for the root
        node's parent and argposition.
        '''
        if self is oldSubtree:
            return newSubtree
        else:
            newnode = self.lightClone()
            for x in range(len(self.children)):
                newnode.children[x] = self.children[x].cloneReplacingNoSubclone(newSubtree, oldSubtree)
                newnode.children[x].parent = newnode
                newnode.children[x].argposition = x
            return newnode
        
    def replaceWith(self, newNode: 'GPNode'):
        newNode.parent = self.parent
        newNode.argposition = self.argposition

        if isinstance(self.parent, GPNode):
            self.parent.children[self.argposition] = newNode
        else:
            # then, self.parent is a GPTree
            self.parent.child = newNode

        for x in range(len(self.children)):
            newNode.children[x] = self.children[x]
            newNode.children[x].parent = newNode
            newNode.children[x].argposition = x

    def __eq__(self, node):
        return (
            type(self) == type(node) and
            len(self.children) == len(node.children)
        )

    def rootedTreeEquals(self, node: 'GPNode') -> bool:
        if self is not node:
            return False
        for x in range(len(self.children)):
            if not self.children[x].rootedTreeEquals(node.children[x]):
                return False
        return True
    
    @abstractmethod
    def __str__(self)->str:
        pass
    
    def toGraphvizString(self)->str:
        return str(self)
    
    def makeGraphvizTree(self) -> str:
        return (
            "digraph g {\n"
            "graph [ordering=out];\n"
            "node [shape=rectangle];\n"
            + self.makeGraphvizSubtree("n")
            + "}\n"
        )

    def makeGraphvizSubtree(self, prefix: str) -> str:
        body = f'{prefix}[label = "{str(self)}"];\n'
        for x, child in enumerate(self.children):
            if x < 10:
                newprefix = f"{prefix}{x}"
            else:
                newprefix = f"{prefix}n{x}"
            body += child.makeGraphvizSubtree(newprefix)
            body += f"{prefix} -> {newprefix};\n"
        return body

    # def printRootedTreeInString(self) -> str:
    #     res = " (" if len(self.children) > 0 else " "
    #     res += str(self)
    #     for child in self.children:
    #         res += child.printRootedTreeInString()
    #     if len(self.children) > 0:
    #         res += ")"
    #     return res
    def printRootedTreeInString(self) -> str:
        parts = []
        stack = [(self, 0)]  # (node, state)

        while stack:
            node, state = stack.pop()

            if state == 0:
                if node.children:
                    parts.append(" (")
                else:
                    parts.append(" ")

                parts.append(str(node))

                if node.children:
                    stack.append((node, 1))
                    for c in reversed(node.children):
                        stack.append((c, 0))
            else:
                parts.append(")")

        return "".join(parts)
    
    @abstractmethod
    def eval(self, state: EvolutionState, thread: int, input: GPData,
             individual, problem: Problem, argval: list[float] = None):
        pass

    # def collectReadRegister(self, s: Set[int]):
    #     from src.lgp.individual.primitive import ReadRegisterGPNode
    #     if isinstance(self, ReadRegisterGPNode):
    #         s.add(self.getIndex())  # Assumes getIndex() method exists in ReadRegisterGPNode

    #     from src.lgp.algorithm.typed_lgp.individual.primitives.typed_feature import TypedFeature
    #     if isinstance(self, TypedFeature) and isinstance(self.input_arg, ReadRegisterGPNode):
    #         s.add(self.input_arg.getIndex()) 


    #     for child in self.children:
    #         child.collectReadRegister(s)
    def collectReadRegister(self, s: set[int]):
        from src.lgp.individual.primitive import ReadRegisterGPNode
        from src.lgp.algorithm.typed_lgp.individual.primitives.typed_feature import TypedFeature

        stack = [self]

        while stack:
            node = stack.pop()

            # Inline checks
            if isinstance(node, ReadRegisterGPNode):
                s.add(node.index)   # direct attribute if possible

            elif isinstance(node, TypedFeature):
                arg = node.input_arg
                if isinstance(arg, ReadRegisterGPNode):
                    s.add(arg.index)

            # Extend stack
            stack.extend(node.children)