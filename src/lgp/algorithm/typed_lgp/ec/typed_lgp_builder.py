from src.ec.evolution_state import EvolutionState
from src.lgp.algorithm.typed_lgp.ec.typed_evolution_state import TypedEvolutionState
from src.ec.gp_builder import GPBuilder
from src.ec.gp_node import GPNode
from src.ec.gp_node_parent import GPNodeParent
from src.ec.gp_primitive_set import GPPrimitiveSet
from src.ec.gp_tree import GPTree
from src.ec.util.parameter import Parameter
from src.lgp.algorithm.typed_lgp.individual.primitives import *
from typing import override, get_origin, get_args, Callable

import collections.abc

from src.lgp.algorithm.typed_lgp.dsl.agent import Agent
# from src.lgp.algorithm.typed_lgp.individual.primitives.typed_interface import TypedInterface
# import src.lgp.algorithm.typed_lgp.dsl.type_checker as tc


class Typed_LGPBuilder (GPBuilder):

    def __init__(self):
        super().__init__()
        self.agent = None

    def setup(self, state:TypedEvolutionState, base:Parameter):
        super().setup(state, base)
        self.agent:Agent = state.typeagent

    @override
    def full_node_reg(self, state:TypedEvolutionState, current:int, max_depth:int, thread:int, 
                      parent:GPNodeParent, argposition:int, funcset:GPPrimitiveSet, 
                      expect_type=None, given_type=None):
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
            n.setInputType([expect_type]) # we can limit the output type of an instruction by fixed the input type of the destination register
            n.setOutputType(expect_type)
            n.children = [self.full_node_reg(state, current + 1, max_depth, thread, n, i, funcset, 
                                             expect_type=expect_type, given_type=given_type)
                        for i, _ in enumerate(n.children)]
            if expect_type is None:
                expect_type = n.children[0].getOutputType() if n.children and n.children[0].getOutputType() != None else None
                n.setOutputType(expect_type)
                n.setInputType([expect_type])
            return n

        #  pick a terminal when we're at max depth or if there are NO nonterminals
        if (current + 1 >= max_depth or not nonterminals) \
            and (tried_terminals := True) \
            and terminals:

            # get the expected input type of the exact argument from the parent node
            # expect_type = parent.getInputType(no_callable=True)[argposition] if parent.getInputType(no_callable=True) != None else None
            
            # collect the GPNodes that might produce feasible types
            coherent_constants = self.getTypeCoherentList(constants, expect_type, given_type, state, thread)

            if state.random[thread].random() < 0.5 and self.canAddConstant(parent) and len(coherent_constants)>0:
                n = state.random[thread].choice(coherent_constants).lightClone()
            else:
                n = state.random[thread].choice(nonconstants).lightClone()

            if isinstance(n, TypedConstant):
                constant_dict = self.agent.getConstantSetWithOutput(expect_type, state, thread)
                n.resetNode(state, thread, constant_dict)
            elif isinstance(n, TypedFeature):
                feature_list = self.agent.getFeaturesWithOutput(expect_type)
                n.resetNode(state, thread, feature_list)
            elif isinstance(n, TypedCallableTerminal):
                # here, the expect_type should be a Callable type
                call_input_type, call_output_type = get_args(expect_type)
                func_list = self.agent.getCallableTerminalListWithOutputInput(call_output_type, call_input_type, state, thread)
                n.resetNode(state, thread, func_list)
            else:
                n.resetNode(state, thread)
                n.output_type = expect_type
            n.argposition = argposition
            n.parent = parent
            return n

        if tried_terminals:
            state.output.warn(f"There is No terminals")

        # else force a nonregisters (function, constants, and ReadRegister ) unless we have no choice
        # filter feasible functions
        nodes_to_pick = []
        # expect_type = parent.getInputType(no_callable=True)[argposition] if parent.getInputType(no_callable=True) != None else None

        coherent_nonreg = self.getTypeCoherentList(nonregisters, expect_type, given_type, state, thread)

        for cand in coherent_nonreg: # we first force to pick a function
            if all( cand.__class__ != c.__class__ for c in terminals) or isinstance(cand, TypedCallableTerminal):
                nodes_to_pick.append(cand.lightClone())

        if not nodes_to_pick:
            state.output.warning("No functions available in full_node_reg, so we can only pick terminals")
            coherent_terminals = self.getTypeCoherentList(terminals, expect_type, given_type, state, thread)
            nodes_to_pick = [c.lightClone() for c in coherent_terminals]

        n = state.random[thread].choice(nodes_to_pick)
        if isinstance(n, TypedConstant):
            constant_dict = self.agent.getConstantSetWithOutput(expect_type, state, thread)
            n.resetNode(state, thread, constant_dict)
        elif isinstance(n, TypedFeature):
            feature_list = self.agent.getFeaturesWithOutput(expect_type)
            n.resetNode(state, thread, feature_list)
        elif isinstance(n, TypedCallableTerminal):
            # here, the expect_type should be a Callable type
            call_input_type, call_output_type = get_args(expect_type) if expect_type is not None else (None, None)
            func_list = self.agent.getCallableTerminalListWithOutputInput(call_output_type, call_input_type, state, thread)
            n.resetNode(state, thread, func_list)
        elif isinstance(n, TypedFunction):
            func_list = self.agent.getFunctionListWithOutput(expect_type, given_type, state, thread)
            n.resetNode(state, thread, func_list)
        else: # typed registers
            n.resetNode(state, thread)
            n.output_type = expect_type
        n.argposition = argposition
        n.parent = parent

        # shuffle the initialization order of children
        indices = list(range(len(n.children)))
        state.random[thread].shuffle(indices)

        for i in indices:
            expect_type = n.getInputType(no_callable=False)[i] if n.getInputType(no_callable=False) != None else None
            n.children[i] = self.full_node_reg(state, current + 1, max_depth, thread, n, i, funcset,
                                               expect_type=expect_type, given_type=given_type)
        return n
    
    @override
    def grow_node_reg(self, state:TypedEvolutionState, current:int, max_depth:int, thread:int, 
                      parent:GPNodeParent, argposition:int, funcset:GPPrimitiveSet, 
                      expect_type=None, given_type=None):
        
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
            n.setInputType([expect_type]) # we can limit the output type of an instruction by fixed the input type of the destination register
            n.setOutputType(expect_type)
            n.children = [self.grow_node_reg(state, current + 1, max_depth, thread, n, i, funcset, 
                                             expect_type=expect_type, given_type=given_type)
                        for i, _ in enumerate(n.children)]
            if expect_type is None:
                expect_type = n.children[0].getOutputType() if n.children and n.children[0].getOutputType() != None else None
                n.setOutputType(expect_type)
                n.setInputType([expect_type])
            return n

        if (current + 1 >= max_depth) \
            and (tried_terminals := True) \
            and terminals:
            # pick a terminal or constant

            # get the expected input type of the exact argument from the parent node
            # expect_type = parent.getInputType(no_callable=True)[argposition] if parent.getInputType(no_callable=True) != None else None
            
            coherent_constants = self.getTypeCoherentList(constants, expect_type, given_type, state, thread)

            if state.random[thread].random() < 0.5 and self.canAddConstant(parent) and len(coherent_constants)>0:
                n = state.random[thread].choice(coherent_constants).lightClone()
            else:
                n = state.random[thread].choice(nonconstants).lightClone()

            if isinstance(n, TypedConstant):
                constant_dict = self.agent.getConstantSetWithOutput(expect_type, state, thread)
                n.resetNode(state, thread, constant_dict)
            elif isinstance(n, TypedFeature):
                feature_list = self.agent.getFeaturesWithOutput(expect_type)
                n.resetNode(state, thread, feature_list)
            elif isinstance(n, TypedCallableTerminal):
                # here, the expect_type should be a Callable type
                call_input_type, call_output_type = get_args(expect_type)
                func_list = self.agent.getCallableTerminalListWithOutputInput(call_output_type, call_input_type, state, thread)
                n.resetNode(state, thread, func_list)
            else: # typed registers
                n.resetNode(state, thread)
                n.output_type = expect_type
            n.argposition = argposition
            n.parent = parent
            return n

        if tried_terminals:
            state.output.warn(f"There is No terminals")

        nodes_to_pick = []
        # expect_type = parent.getInputType(no_callable=True)[argposition] if parent.getInputType(no_callable=True) != None else None

        coherent_nonreg = self.getTypeCoherentList(nonregisters, expect_type, given_type, state, thread)

        if current == 1:
            for cand in coherent_nonreg: # we first force to pick a function
                if all( cand.__class__ != c.__class__ for c in terminals) or isinstance(cand, TypedCallableTerminal):
                    nodes_to_pick.append(cand.lightClone())

            if not nodes_to_pick:
                state.output.warning("No functions available in grow_node_reg, so we can only pick terminals")
                nodes_to_pick = [c.lightClone() for c in coherent_nonreg]
        else:
            for cand in coherent_nonreg:
                if all( cand.__class__ != c.__class__ for c in constants) or self.canAddConstant(parent):
                    #if it is not a constant or we can add a constant, we can pick it
                    nodes_to_pick.append(cand.lightClone())

            if not nodes_to_pick:
                state.output.fatal("No nonregisters available in grow_node_reg")    
        

        n = state.random[thread].choice(nodes_to_pick)
        if isinstance(n, TypedConstant):
            constant_dict = self.agent.getConstantSetWithOutput(expect_type, state, thread)
            n.resetNode(state, thread, constant_dict)
        elif isinstance(n, TypedFeature):
            feature_list = self.agent.getFeaturesWithOutput(expect_type)
            n.resetNode(state, thread, feature_list)
        elif isinstance(n, TypedCallableTerminal):
            # here, the expect_type should be a Callable type
            call_input_type, call_output_type = get_args(expect_type) if expect_type is not None else (None, None)
            func_list = self.agent.getCallableTerminalListWithOutputInput(call_output_type, call_input_type, state, thread)
            n.resetNode(state, thread, func_list)
        elif isinstance(n, TypedFunction):
            func_list = self.agent.getFunctionListWithOutput(expect_type, given_type, state, thread)
            n.resetNode(state, thread, func_list)
        else: # typed registers
            n.resetNode(state, thread)
            n.output_type = expect_type
        n.argposition = argposition
        n.parent = parent

        # shuffle the initialization order of children
        indices = list(range(len(n.children)))
        state.random[thread].shuffle(indices)

        for i in indices:
            expect_type = n.getInputType(no_callable=False)[i] if n.getInputType(no_callable=False) != None else None
            n.children[i] = self.grow_node_reg(state, current + 1, max_depth, thread, n, i, funcset, 
                                               expect_type=expect_type, given_type=given_type)
        return n

    def getTypeCoherentList(self, candidates:list, expect_type, given_type=None, 
                            state:EvolutionState=None, thread:int=None)->list:
        """
        get the type-consistent elements from the candidate list.
        Note: the elements have not been clone. Clone (lightClone()) the element before you use it.
        """
        if expect_type is None:
            return candidates

        res = []
        for can in candidates:
            if isinstance(can, TypedConstant):
                # can we add constants?
                constant_dict = self.agent.getConstantSetWithOutput(expect_type, state, thread)
                if len(constant_dict) > 0:
                    res.append(can)
            elif isinstance(can, TypedFeature):
                feature_list = self.agent.getFeaturesWithOutput(expect_type)
                if len(feature_list) > 0:
                    res.append(can)
            elif isinstance(can, TypedInput):
                # can we add inputs?
                if can.getOutputType() == expect_type:
                    res.append(can)
            elif isinstance(can, TypedCallableTerminal):
                # can we add Callable nodes?
                if get_origin(expect_type) is collections.abc.Callable \
                    or expect_type is collections.abc.Callable \
                    or expect_type is Callable:
                    call_input_type, call_output_type = get_args(expect_type)
                    func_list = self.agent.getCallableTerminalListWithOutputInput(call_output_type, call_input_type, state, thread)
                    if len(func_list) > 0:
                        res.append(can)
            elif isinstance(can, TypedFunction):
                # can we add functions?
                func_list = self.agent.getFunctionListWithOutput(expect_type, given_type, state, thread)
                if len(func_list) > 0:
                    res.append(can)
            else:
                # "can" might be registers
                res.append(can)
        
        return res
    
    def newRootedTree(self, state:TypedEvolutionState, thread:int, parent:GPNodeParent, 
                      set:GPPrimitiveSet, argposition:int, position:int=0, 
                      expect_type=None, given_type=None)->GPNode:
        # return self.full_node(state, 0, state.random[thread].randint(0, self.maxDepth-self.minDepth) + self.minDepth,
        #                  thread,parent,argposition,set)
        if state.random[thread].random() < 0.5:
            return self.grow_node_reg(state, position, state.random[thread].randint(self.minDepth, self.maxDepth), thread, 
                                      parent, argposition, set, expect_type=expect_type, given_type=given_type)
        else:
            return self.full_node_reg(state, position, state.random[thread].randint(self.minDepth, self.maxDepth), thread, 
                                      parent, argposition, set, expect_type=expect_type, given_type=given_type)
    
    # def can_add_constant(self, parent:GPNodeParent)->bool:
    #     root:GPNode = parent
    #     # if isinstance(root, FlowOperator):
    #     #     return True
    #     terminalsize = root.numNodes(GPNode.NODESEARCH_TERMINALS)
    #     constnatsize = root.numNodes(GPNode.NODESEARCH_CONSTANT)
    #     nullsize = root.numNodes(GPNode.NODESEARCH_NULL)
    #     if terminalsize + nullsize > 1:
    #         return (constnatsize + 1) / (terminalsize + nullsize) <= self.probCons
    #     else:
    #         return True