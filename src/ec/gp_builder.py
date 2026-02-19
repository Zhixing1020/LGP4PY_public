# import sys
# sys.path.append('D:/zhixing/科研/LGP4PY/LGP4PY')

from src.ec.evolution_state import EvolutionState
from src.ec.gp_node import GPNode
from src.ec.gp_node_parent import GPNodeParent
from src.ec.gp_primitive_set import GPPrimitiveSet
from src.ec.gp_tree import GPTree
from src.ec.util.parameter import Parameter
from src.lgp.individual.primitive import *

class GPBuilder:
    '''
    A builder for GP trees, implementing the Koza-style half-and-half (full and grow) methods.
    '''

    P_MAXDEPTH = "max-depth"
    P_MINDEPTH = "min-depth"
    P_PROBCON = "prob_constant"
    P_BUILDER = "builder"

    def __init__(self):
        self.maxDepth:int = 3
        self.minDepth:int = 1
        self.probCons = 0.5

    @classmethod
    def default_base(cls)->Parameter:
        return Parameter(cls.P_BUILDER)

    def setup(self, state:EvolutionState, base:Parameter):
        # super().setup(state, base)
        def_base = self.default_base()

        self.maxDepth = state.parameters.getInt(base.push(self.P_MAXDEPTH), def_base.push(self.P_MAXDEPTH))
        if self.maxDepth <= 0:
            state.output.fatal("The Max Depth for a KozaBuilder must be at least 1.")

        self.minDepth = state.parameters.getInt(base.push(self.P_MINDEPTH), def_base.push(self.P_MINDEPTH))
        if self.minDepth <= 0:
            state.output.fatal("The Min Depth for a KozaBuilder must be at least 1.")

        if self.maxDepth < self.minDepth:
            state.output.fatal("Max Depth must be >= Min Depth for a KozaBuilder")

        self.probCons = state.parameters.getDoubleWithDefault(base.push(self.P_PROBCON), def_base.push(self.P_PROBCON), 0.5)
        if self.probCons < 0:
            state.output.fatal("The constant probability for a KozaBuilder must be >= 0.0")

    def canAddConstant(self, parent:GPNodeParent)->bool:
        root:GPNode = parent
        if isinstance(root, FlowOperator):
            return True
        terminalsize = root.numNodes(GPNode.NODESEARCH_TERMINALS)
        constnatsize = root.numNodes(GPNode.NODESEARCH_CONSTANT)
        nullsize = root.numNodes(GPNode.NODESEARCH_NULL)
        if terminalsize > 0:
            return (constnatsize + 1) / (terminalsize + nullsize) <= self.probCons
        else:
            return False
        
    def full_node(self, state:EvolutionState, current:int, max_depth:int, thread:int, 
    parent:GPNodeParent, argposition:int, func_set:GPPrimitiveSet):
        tried_terminals = False

        # t = type_.type
        terminals = func_set.terminals
        nonterminals = func_set.nonterminals
        nodes = func_set.nodes

        if len(nonterminals) == 0:
            state.output.warning("there is NO nonterminals")

        if len(nodes) == 0:
            state.output.error("there is no node for a certain type")

        if ((current + 1 >= max_depth or len(nonterminals) == 0)
            and (tried_terminals := True)
            and len(terminals) != 0):

            n:GPNode = state.random[thread].choice(terminals).lightClone()
            n.resetNode(state, thread)
            n.argposition = argposition
            n.parent = parent
            return n
        else:
            if tried_terminals:
                state.output.error("there is NO terminals")

            nodes_to_pick = func_set.nonterminals
            if nodes_to_pick is None or len(nodes_to_pick) == 0:
                nodes_to_pick = func_set.terminals

            n:GPNode = state.random[thread].choice(nodes_to_pick).lightClone()
            n.resetNode(state, thread)
            n.argposition = argposition
            n.parent = parent

            n.children = [self.full_node(state, current + 1, max_depth, thread, n, i, func_set) for i, _ in enumerate(n.children)]

            return n
        
    def grow_node(self, state:EvolutionState, current:int, max_depth:int, thread:int, 
                  parent:GPNodeParent, argposition:int, func_set:GPPrimitiveSet):
        
        tried_terminals = False
        terminals = func_set.terminals
        nodes = func_set.nodes

        if len(nodes) == 0:
            state.output.error("there is no node for a certain type")

        if (current + 1 >= max_depth) \
            and (tried_terminals := True) \
            and (terminals):  # pick terminal

            n:GPNode = state.random[thread].choice(terminals).lightClone()
            n.resetNode(state, thread)
            n.argposition = argposition
            n.parent = parent
            return n
        else:
            if tried_terminals:
                state.output.error(f"There is No terminals")

            n = state.random[thread].choice(nodes).lightClone()
            n.resetNode(state, thread)
            n.argposition = argposition
            n.parent = parent

            n.children = [
                self.grow_node(state, current + 1, max_depth, thread, n, i, func_set)
                for i, _ in enumerate(n.children)
            ]
            return n
        
    def full_node_reg(self, state:EvolutionState, current:int, max_depth:int, thread:int, 
                      parent:GPNodeParent, argposition:int, funcset:GPPrimitiveSet):
        tried_terminals = False
        terminals = funcset.terminals
        nonterminals = funcset.nonterminals
        registers = funcset.registers
        nonregisters = funcset.nonregisters
        constants = funcset.constants
        nonconstants = funcset.nonconstants

        # pick a write_register at the root
        if current == 0:
            n:GPNode = state.random[thread].choice(registers).lightClone()
            n.resetNode(state, thread)
            n.argposition = argposition
            n.parent = parent
            n.children = [self.full_node_reg(state, current + 1, max_depth, thread, n, i, funcset)
                        for i, _ in enumerate(n.children)]
            return n

        #  pick a terminal when we're at max depth or if there are NO nonterminals
        if (current + 1 >= max_depth or not nonterminals) \
            and (tried_terminals := True) \
            and terminals:

            if state.random[thread].random() < 0.5 and self.canAddConstant(parent):
                n = state.random[thread].choice(constants).lightClone()
            else:
                n = state.random[thread].choice(nonconstants).lightClone()
            n.resetNode(state, thread)
            n.argposition = argposition
            n.parent = parent
            return n

        if tried_terminals:
            state.output.warn(f"There is No terminals")

        # else force a nonregisters (function, constants, and ReadRegister ) unless we have no choice
        # filter feasible functions
        nodes_to_pick = []
        for cand in nonregisters: # we first force to pick a function
            if all( cand.__class__ != c.__class__ for c in terminals):
                nodes_to_pick.append(cand.lightClone())

        if not nodes_to_pick:
            state.output.warning("No functions available in full_node_reg, so we can only pick terminals")
            nodes_to_pick = terminals

        n = state.random[thread].choice(nodes_to_pick)
        n.resetNode(state, thread)
        n.argposition = argposition
        n.parent = parent

        # shuffle the initialization order of children
        indices = list(range(len(n.children)))
        state.random[thread].shuffle(indices)

        for i in indices:
            n.children[i] = self.full_node_reg(state, current + 1, max_depth, thread, n, i, funcset)
        return n
    
    def grow_node_reg(self, state:EvolutionState, current:int, max_depth:int, thread:int, 
                      parent:GPNodeParent, argposition:int, funcset:GPPrimitiveSet):
        
        tried_terminals = False
        terminals = funcset.terminals
        registers = funcset.registers
        nonregisters = funcset.nonregisters
        constants = funcset.constants
        nonconstants = funcset.nonconstants

        if current == 0:
            n:GPNode = state.random[thread].choice(registers).lightClone()
            n.resetNode(state, thread)
            n.argposition = argposition
            n.parent = parent
            n.children = [self.grow_node_reg(state, current + 1, max_depth, thread, n, i, funcset)
                        for i, _ in enumerate(n.children)]
            return n

        if (current + 1 >= max_depth) \
            and (tried_terminals := True) \
            and terminals:
            # pick a terminal or constant
            if state.random[thread].random() < 0.5 and self.canAddConstant(parent):
                n = state.random[thread].choice(constants).lightClone()
            else:
                n = state.random[thread].choice(nonconstants).lightClone()
            n.resetNode(state, thread)
            n.argposition = argposition
            n.parent = parent
            return n

        if tried_terminals:
            state.output.warn(f"There is No terminals")

        nodes_to_pick = []

        if current == 1:
            for cand in nonregisters: # we first force to pick a function
                if all( cand.__class__ != c.__class__ for c in terminals):
                    nodes_to_pick.append(cand.lightClone())

            if not nodes_to_pick:
                state.output.warning("No functions available in grow_node_reg, so we can only pick terminals")
                nodes_to_pick = terminals
        else:
            for cand in nonregisters:
                if all( cand.__class__ != c.__class__ for c in constants) or self.canAddConstant(parent):
                    #if it is not a constant or we can add a constant, we can pick it
                    nodes_to_pick.append(cand.lightClone())

            if not nodes_to_pick:
                state.output.fatal("No nonregisters available in grow_node_reg")

        n = state.random[thread].choice(nodes_to_pick)
        n.resetNode(state, thread)
        n.argposition = argposition
        n.parent = parent

        # shuffle the initialization order of children
        indices = list(range(len(n.children)))
        state.random[thread].shuffle(indices)

        for i in indices:
            n.children[i] = self.grow_node_reg(state, current + 1, max_depth, thread, n, i, funcset)
        return n


    
    def newRootedTree(self, state:EvolutionState, thread:int, parent:GPNodeParent, set:GPPrimitiveSet, argposition:int, position:int=0)->GPNode:
        # return self.full_node(state, 0, state.random[thread].randint(0, self.maxDepth-self.minDepth) + self.minDepth,
        #                  thread,parent,argposition,set)
        if state.random[thread].random() < 0.5:
            return self.grow_node_reg(state, position, state.random[thread].randint(self.minDepth, self.maxDepth), thread, parent, argposition, set)
        else:
            return self.full_node_reg(state, position, state.random[thread].randint(self.minDepth, self.maxDepth), thread, parent, argposition, set)
    

if __name__ == "__main__":
    
    from src.ec.util.parameter import Parameter
    builder = GPBuilder()
    
    state = EvolutionState('D:\\zhixing\\科研\\LGP4PY\\LGP4PY\\tasks\\Symbreg\\parameters\\LGP_test.params')
    state.setup("")
    builder.setup(state, Parameter("gp.koza.half"))
    state.primitive_set = GPPrimitiveSet()
    state.primitive_set.setup(state, Parameter('gp.fs.0'))  # here, I need to use a default parameter name
    # fun_set = {Add(), InputFeatureGPNode()}
    tree = GPTree()
    for _ in range(0,10):
        tree.child = builder.newRootedTree(state, 0, tree, state.primitive_set, 0)
        print(tree)
