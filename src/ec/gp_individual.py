from src.ec.gp_tree import GPTree
from src.ec.gp_defaults import GPDefaults
from src.ec.gp_node import GPNode
from src.ec.evolution_state import EvolutionState
from src.ec.gp_data import GPData
from src.ec.fitness import Fitness
# from copy import deepcopy

from tasks.problem import Problem

from abc import ABC, abstractmethod

class GPIndividual(ABC):
    '''A simple GP individual with only one tree'''
    P_NUMTREES = "numtrees"
    P_TREE = "tree"
    P_INDIVIDUAL = "individual"
    EVALUATED_PREAMBLE = "Evaluated: "
    
    P_TOWRAP = "to-wrap"
    P_BATCHSIZE = "batch-size"

    def __init__(self):
        self.treelist:list[GPTree] = []*1  # storing all the tree in this individual
        self.evaluated = False  # evaluated or not
        self.fitness:Fitness = None  
        self.species = None   # the species that this individual belongs to
        self.breedingPipe = None   # the pipeline that produces this individuals
        
        self.towrap = False  # to wrap the output or not
        self.batchsize = 1   # the batch size in training


    @classmethod
    def defaultBase(cls):
        return GPDefaults.base().push(cls.P_INDIVIDUAL)

    def __eq__(self, other):
        return self.equals(other)
        
    def equals(self, ind:'GPIndividual'):
        if ind is None:
            return False
        if not isinstance(ind, self.__class__):
            return False
        if len(self.treelist) != len(ind.treelist):
            return False
        return all(self_tree.treeEquals(other_tree) for self_tree, other_tree in zip(self.treelist, ind.treelist))
    
    def setup(self, state, base):

        def_base = self.defaultBase()
        self.evaluated = False

        # t = state.parameters.getInt(base.push(self.P_NUMTREES), def_base.push(self.P_NUMTREES), 1)
        # if t <= 0:
        #     state.output.fatal("A GPIndividual must have at least one tree.", base.push(self.P_NUMTREES), def_base.push(self.P_NUMTREES))

        # self.trees = [None] * t
        # for x in range(t):
        #     p = base.push(self.P_TREE).push(str(x))
        #     self.trees[x] = state.parameters.getInstanceForParameterEq(p, def_base.push(self.P_TREE).push(str(x)), GPTree)
        #     self.trees[x].owner = self
        #     self.trees[x].setup(state, p)
        self.treelist = [GPTree]*1
        p = base.push(self.P_TREE).push(str(0))
        self.treelist[0] = state.parameters.getInstanceForParameter(p, def_base.push(self.P_TREE).push(str(0)), GPTree)
        self.treelist[0].owner = self
        self.treelist[0].setup(state, p)

        # initializer = state.initializer  # expected to be GPInitializer
        # for x in range(t):
        #     constraints = self.trees[x].constraints(initializer)
        #     for node_array in constraints.functionset.nodes:
        #         for node in node_array:
        #             node.check_constraints(state, x, self, base)

    def printTrees(self)->str:
        res = ""
        for i, tree in enumerate( self.treelist ):
            res += f"Tree {i}: {tree}"

        return res
    
    def printIndividualForHuman(self, state:EvolutionState)->str:
        res = ""
        res += self.EVALUATED_PREAMBLE + ("true" if self.evaluated else "false") + "\n"
        res += f"Fitness:\t[{ str( self.fitness )}]\n"
        
        res += self.printTrees() + "\n"

        return res
    
    def __str__(self)->str:
        return self.printIndividualForHuman()
    
    def clone(self):
        myobj = self.__class__()
        myobj.fitness = self.fitness.clone() if self.fitness is not None else None
        myobj.treelist = [tree.clone() for tree in self.treelist]
        myobj.towrap = self.towrap
        myobj.batchsize = self.batchsize
        myobj.breedingPipe = self.breedingPipe 
        for tree in myobj.treelist:
            tree.owner = myobj
        myobj.evaluated = self.evaluated
        return myobj

    def lightClone(self):
        myobj = self.__class__()
        myobj.fitness = self.fitness.clone() if self.fitness is not None else None
        myobj.treelist = [tree.lightClone() for tree in self.treelist]
        myobj.towrap = self.towrap
        myobj.batchsize = self.batchsize
        myobj.breedingPipe = self.breedingPipe
        for tree in myobj.treelist:
            tree.owner = myobj
        myobj.evaluated = self.evaluated
        return myobj
    
    # def __deepcopy__(self):
    #     self.clone()

    def size(self):
        return sum(tree.child.num_nodes(GPNode.NODESEARCH_ALL) for tree in self.treelist)
    
    def getTree(self, index:int)->GPTree:
        return self.treelist[index]
    
    def getTrees(self)->list[GPTree]:
        return self.treelist
    
    def setTree(self, index:int, tree:GPTree)->bool:
        if index < len(self.treelist):
            self.treelist[index] = tree
            return True
        else:
            print(f"setTree index: {index} is out of range " + len(self.treelist))
            return False
        
    def getTreesLength(self)->int:
        return len(self.treelist)
    
    @abstractmethod
    def rebuildIndividual(self, state: EvolutionState, thread: int):
        pass

    @abstractmethod
    def execute(self, state:EvolutionState, thread:int, input:GPData, individual:'GPIndividual', problem:Problem):
        pass

    @abstractmethod
    def preExecution(self, state:EvolutionState, thread:int):
        pass

    @abstractmethod
    def postExecution(self, state:EvolutionState, thread:int):
        pass

    @abstractmethod
    def makeGraphvizRule(self, outputRegs:list[int])->str:
        pass
    
    @abstractmethod
    def wrapper(self, predict_list:list[list[float]], target_list:list[list[float]], state:EvolutionState, thread:int, problem:Problem):
        pass

    def IsWrap(self)->bool:
        return self.towrap
    
    @abstractmethod
    def getWrapper(self):
        pass
    

if __name__ == "__main__":
    
    from src.ec.util.parameter import Parameter
    builder = GPBuilder()
    state = EvolutionState('D:\\zhixing\\科研\\LGP4PY\\LGP4PY\\tasks\\Symbreg\\parameters\\LGP_test.params')
    state.setup("")
    state.primitive_set = GPPrimitiveSet()
    state.primitive_set.setup(state, Parameter('gp.fs.0'))  # here, I need to use a default parameter name
    # fun_set = {Add(), InputFeatureGPNode()}
    tree = GPTree()
    
    individual = GPIndividual()
    individual.setup(state, Parameter('pop.subpop.0.species.ind'))
