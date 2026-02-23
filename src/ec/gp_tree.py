from typing import List

from src.ec.util.parameter import Parameter
from src.ec.util.parameter_database import ParameterDatabase

from src.ec.evolution_state import EvolutionState
from src.ec.gp_node import GPNode
# from src.ec.gp_individual import GPIndividual
from src.ec.gp_defaults import GPDefaults
from src.ec.gp_node_parent import GPNodeParent

class GPTree(GPNodeParent):
    
    __slots__ = ("child", "owner", "species")
    P_TREE: str = "tree"

    def __init__(self):
        # super()
        self.child:GPNode = None
        self.owner = None
        self.species = None
    
    @classmethod
    def defaultBase(cls) -> Parameter:
        return GPDefaults.base().push(cls.P_TREE)
    
    def __eq__(self, other):
        return self.treeEquals(other)

    def treeEquals(self, tree:'GPTree')->bool:
        return self.child.rootedTreeEquals(tree.child)
    
    def lightClone(self) -> 'GPTree':
        # Like shallow copy: just replicate the GPTree object and share child
        newtree:GPTree = super().clone()
        # newtree.constraints = self.constraints
        newtree.child = self.child  # NOTE: shared reference
        newtree.owner = self.owner
        # newtree.argposition = self.argposition
        # newtree.parent = self.parent
        newtree.species = self.species
        return newtree

    # def clone(self) -> 'GPTree':
    #     newtree = self.lightClone()
    #     if self.child is not None:
    #         newtree.child = self.child.clone()  # assumes GPNode.clone() exists
    #         newtree.child.parent = newtree
    #         newtree.child.argposition = 0
    #     # still share the same owner
    #     return newtree
    def clone(self) -> "GPTree":
        newtree = self.lightClone()

        root = self.child
        if root is None:
            return newtree

        # Clone root
        new_root = root.lightClone()
        new_root.parent = newtree
        new_root.argposition = 0
        newtree.child = new_root

        # Stack: (old_node, new_node)
        stack = [(root, new_root)]

        while stack:
            old, new = stack.pop()

            children = old.children
            if not children:
                continue

            new_children = []
            for i, c in enumerate(children):
                nc = c.lightClone()
                nc.parent = new
                nc.argposition = i
                new_children.append(nc)
                stack.append((c, nc))

            new.children = new_children

        return newtree
    
    def setup(self, state:EvolutionState, base:Parameter):
        def_param = self.defaultBase()

    def __str__(self)->str:
        return self.child.printRootedTreeInString()

    def buildTree(self, state:EvolutionState, thread:int):
        self.child = state.builder.newRootedTree(state, 0, self, self.species.primitiveset, 0)
