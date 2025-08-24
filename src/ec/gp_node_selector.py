
from src.ec.util import *
from src.ec.evolution_state import EvolutionState
from src.ec.gp_individual import GPIndividual
from src.ec.gp_tree import GPTree
from src.ec.gp_node import GPNode,GPNodeGather

class GPNodeSelector:
    P_NODESELECTOR = "ns"
    P_TERMINAL_PROBABILITY = "terminals"
    P_NONTERMINAL_PROBABILITY = "nonterminals"
    P_ROOT_PROBABILITY = "root"

    def __init__(self):
        self.rootProbability = 0.0
        self.terminalProbability = 0.0
        self.nonterminalProbability = 0.0
        self.nonterminals = -1 # The number of nonterminals in the tree, -1 if unknown.
        self.terminals = -1 # The number of terminals in the tree, -1 if unknown.
        self.nodes = -1 #  The number of nodes in the tree, -1 if unknown. 
        self.reset()

    def defaultBase(self):
        return Parameter(self.P_NODESELECTOR)

    def clone(self):
        s = self.__class__()
        s.rootProbability = self.rootProbability
        s.terminalProbability = self.terminalProbability
        s.nonterminalProbability = self.nonterminalProbability
        self.reset()
        return s

    def setup(self, state:EvolutionState, base:Parameter):
        def_base = self.defaultBase()

        self.terminalProbability = state.parameters.getDoubleWithDefault(
            base.push(self.P_TERMINAL_PROBABILITY),
            def_base.push(self.P_TERMINAL_PROBABILITY), 0.35)
        if self.terminalProbability == -1.0:
            state.output.fatal("Invalid terminal probability for GPNodeSelector")

        self.nonterminalProbability = state.parameters.getDoubleWithDefault(
            base.push(self.P_NONTERMINAL_PROBABILITY),
            def_base.push(self.P_NONTERMINAL_PROBABILITY), 0.35)
        if self.nonterminalProbability == -1.0:
            state.output.fatal("Invalid nonterminal probability for GPNodeSelector")

        self.rootProbability = state.parameters.getDoubleWithDefault(
            base.push(self.P_ROOT_PROBABILITY),
            def_base.push(self.P_ROOT_PROBABILITY), 0.3)
        if self.rootProbability == -1.0:
            state.output.fatal("Invalid root probability for GPNodeSelector")

        if self.rootProbability + self.terminalProbability + self.nonterminalProbability > 1.0:
            state.output.fatal("Probabilities for root, terminals, and nonterminals must not sum to more than 1.0")

        self.reset()

    def reset(self):
        self.nonterminals = self.terminals = self.nodes = -1

    def pickNode(self, state:EvolutionState, subpopulation:int, thread:int, ind:GPIndividual, tree:GPTree, GPNodeType:int=None):
        
        pick = GPNodeGather()

        if GPNodeType is not None:
            if self.nodes == -1:
                self.nodes = tree.child.numNodes(GPNodeType)
            if self.nodes > 0:
                tree.child.nodeInPosition(state.random[thread].randint(0, self.nodes-1), pick, GPNodeType)
                return pick.node
            else: #  there ARE no the type of GPNode!  simply return the root node
                return tree.child

        rnd = state.random[thread].uniform(0, 1)
        if rnd > self.nonterminalProbability + self.terminalProbability + self.rootProbability:
            if self.nodes == -1:
                self.nodes = tree.child.numNodes(GPNode.NODESEARCH_ALL)
            tree.child.nodeInPosition(state.random[thread].randint(0, self.nodes-1), pick, GPNode.NODESEARCH_ALL)
            return pick.node
        
        elif rnd > self.nonterminalProbability + self.terminalProbability:
            return tree.child
        
        elif rnd > self.nonterminalProbability:
            if self.terminals == -1:
                self.terminals = tree.child.numNodes(GPNode.NODESEARCH_TERMINALS)
            tree.child.nodeInPosition(state.random[thread].randint(0, self.terminals-1), pick, GPNode.NODESEARCH_TERMINALS)
            return pick.node
        else:
            if self.nonterminals == -1:
                self.nonterminals = tree.child.numNodes(GPNode.NODESEARCH_NONTERMINALS)
            if self.nonterminals > 0:
                tree.child.nodeInPosition(state.random[thread].randint(0, self.nonterminals-1), pick, GPNode.NODESEARCH_NONTERMINALS)
                return pick.node
            else:
                return tree.child