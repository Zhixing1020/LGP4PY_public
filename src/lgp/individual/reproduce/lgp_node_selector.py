
from src.ec.gp_node_selector import GPNodeSelector
from src.ec.util import *
from src.ec.gp_node import GPNode, GPNodeGather
from src.ec.evolution_state import EvolutionState
from src.lgp.individual import *

class LGPNodeSelector(GPNodeSelector):
    
    P_CONSTANT_PROB = "constants"
    P_READREGISTER_PROB = "read_registers"
    
    def __init__(self):
        self.constantProbability = 0.0
        self.readRegProbability = 0.0
        self.constants = -1
        self.readregs = -1
        self.reset()
    
    def defaultBase(self):
        pp = Parameter("lgp")
        return pp.push(self.P_NODESELECTOR)
    
    def setup(self, state, base):
        def_base = self.defaultBase()
        
        self.constantProbability = state.parameters.getDoubleWithDefault(
            base.push(self.P_CONSTANT_PROB),
            def_base.push(self.P_CONSTANT_PROB), 0.25)
        if self.constantProbability == -1.0:
            state.output.fatal("Invalid constant probability for LGPNodeSelector ",
                base.push(self.P_CONSTANT_PROB),
                def_base.push(self.P_CONSTANT_PROB))
        
        self.readRegProbability = state.parameters.getDoubleWithDefault(
            base.push(self.P_READREGISTER_PROB), 
            def_base.push(self.P_READREGISTER_PROB), 0.25)
        if self.readRegProbability == -1.0:
            state.output.fatal("Invalid nonterminal probability for KozaNodeSelector ",
                base.push(self.P_READREGISTER_PROB), 
                def_base.push(self.P_READREGISTER_PROB))
        
        self.nonterminalProbability = state.parameters.getDoubleWithDefault(
                base.push(self.P_NONTERMINAL_PROBABILITY), 
                def_base.push(self.P_NONTERMINAL_PROBABILITY), 0.25)
        if self.nonterminalProbability == -1.0:
            state.output.fatal("Invalid nonterminal probability for KozaNodeSelector ",
                base.push(self.P_NONTERMINAL_PROBABILITY), 
                def_base.push(self.P_NONTERMINAL_PROBABILITY))

        self.rootProbability = state.parameters.getDoubleWithDefault(
            base.push(self.P_ROOT_PROBABILITY),
            def_base.push(self.P_ROOT_PROBABILITY), 0.25)
        if self.rootProbability == -1.0:
            state.output.fatal("Invalid root probability for KozaNodeSelector ",
                base.push(self.P_ROOT_PROBABILITY),
                def_base.push(self.P_ROOT_PROBABILITY))
        
        self.terminalProbability = self.constantProbability + self.readRegProbability
        
        if (self.rootProbability + self.nonterminalProbability + 
            self.readRegProbability + self.constantProbability > 1.0):
            state.output.fatal("The nonterminal, constant, readregister and root for LGPNodeSelector" + 
        str(base) + " may not sum to more than 1.0. (" + str(self.nonterminalProbability) + " " + str(self.rootProbability) + " " 
                + str(self.constantProbability) + " " + str(self.readRegProbability) + ")", base)
        
        self.reset()
    
    def reset(self):
        self.nonterminals = self.nodes = self.constants = self.readregs = -1
    
    def pickNode(self, state:EvolutionState, subpopulation:int, thread:int, ind:LGPIndividual, tree:GPTreeStruct):
        rnd = state.random[thread].uniform(0, 1)
        pick = GPNodeGather()
        if rnd > self.nonterminalProbability + self.constantProbability + self.readRegProbability + self.rootProbability:
            if self.nodes == -1:
                self.nodes = tree.child.numNodes(GPNode.NODESEARCH_ALL)
            tree.child.nodeInPosition(state.random[thread].randint(0, self.nodes-1), pick, GPNode.NODESEARCH_ALL)
            return pick.node
        
        elif rnd > self.nonterminalProbability + self.constantProbability + self.readRegProbability:
            return tree.child
        
        elif rnd > self.nonterminalProbability + self.readRegProbability:
            if self.constants == -1:
                self.constants = tree.child.numNodes(GPNode.NODESEARCH_CONSTANT)
            if self.constants > 0:
                tree.child.nodeInPosition(state.random[thread].randint(0, self.constants-1), pick, GPNode.NODESEARCH_CONSTANT)
                return pick.node
            else:
                if self.readregs == -1:
                    self.readregs = tree.child.numNodes(GPNode.NODESEARCH_READREG)
                tree.child.nodeInPosition(state.random[thread].randint(0, self.readregs-1), pick, GPNode.NODESEARCH_READREG)
                return pick.node
        
        elif rnd > self.nonterminalProbability:
            if self.readregs == -1:
                self.readregs = tree.child.numNodes(GPNode.NODESEARCH_READREG)
            if self.readregs > 0:
                tree.child.nodeInPosition(state.random[thread].randint(0, self.readregs-1), pick, GPNode.NODESEARCH_READREG)
                return pick.node
            else:
                self.constants = tree.child.numNodes(GPNode.NODESEARCH_CONSTANT)
                tree.child.nodeInPosition(state.random[thread].randint(0, self.constants-1), pick, GPNode.NODESEARCH_CONSTANT)
                return pick.node
        
        else:
            if self.nonterminals == -1:
                self.nonterminals = tree.child.numNodes(GPNode.NODESEARCH_NONTERMINALS)
            if self.nonterminals > 0:
                tree.child.nodeInPosition(state.random[thread].randint(0, self.nonterminals-1), pick, GPNode.NODESEARCH_NONTERMINALS)
                return pick.node
            else:
                return tree.child
    
    def clone(self):
        s = super().clone()
        s.constantProbability = self.constantProbability
        s.readRegProbability = self.readRegProbability
        s.reset()
        return s