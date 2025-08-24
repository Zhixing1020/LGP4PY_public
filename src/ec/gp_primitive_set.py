from typing import List
from src.ec.gp_node import GPNode
from src.ec.evolution_state import EvolutionState
from src.ec.util.parameter import Parameter

from src.lgp.individual.primitive import *

class GPPrimitiveSet:

    P_NAME = "name"
    P_FUNC = "func"
    P_SIZE = "size"
    P_NUMREGISTERS = "numregisters"

    def __init__(self):
        self.nodes:List[GPNode] = list()
        # self.nodes_by_name:Set[str] = set()

        self.nonterminals:List[GPNode] = list()
        self.terminals:List[GPNode] = list()

        self.registers:List[GPNode] = list()
        self.nonregisters:List[GPNode] = list()

        self.constants:List[GPNode] = list()
        self.nonconstants:List[GPNode] = list()

        self.flowoperators:List[GPNode] = list()
        

    def setup(self, state:EvolutionState, base:Parameter):
        self.name = state.parameters.getString(base.push("name"), None)
        if self.name is None:
            state.output.fatal("No name was given for this function set.", base.push("name"))

        # old_fs = state.initializer.function_set_repository.get(self.name)
        # if old_fs is not None:
        #     state.output.fatal(f"The GPFunctionSet \"{self.name}\" has been defined multiple times.", base.push("name"))
        # state.initializer.function_set_repository[self.name] = self

        num_funcs = state.parameters.getInt(base.push("size"), None)
        if num_funcs < 1:
            state.output.error(f"The GPFunctionSet \"{self.name}\" has no function.")

        for x in range(num_funcs):
            pp = base.push("func").push(str(x))
            gpfi = state.parameters.getInstanceForParameter(pp, None, GPNode)
            gpfi.setup(state, pp)

            # Special handling
            if isinstance(gpfi, InputFeatureGPNode):
                rng = state.parameters.getInt(pp.push("size"), None)
                gpfi.setRange(rng)

            elif isinstance(gpfi, ConstantGPNode):
                lb = state.parameters.getDoubleWithDefault(pp.push("lowbound"), None, 0.0)
                ub = state.parameters.getDoubleWithDefault(pp.push("upbound"), None, 1.0)
                step = state.parameters.getDoubleWithDefault(pp.push("step"), None, 0.1)
                if lb > ub:
                    state.output.fatal("the range of constants does not make sense")
                gpfi = ConstantGPNode(lb, ub, step)
                gpfi.setup(state, pp)

            elif isinstance(gpfi, FlowOperator):
                mbl1 = state.parameters.getInt(pp.push("maxbodylength"), None)
                if mbl1 < 1:
                    state.output.fatal("max body length must be >=1")
                mbl2 = state.parameters.getInt(pp.push("minbodylength"), None)
                if mbl2 < 1 or mbl2 > mbl1:
                    state.output.fatal(f"min body length is illegal, please check the setting of {pp.push('minbodylength')}")
                gpfi.set_max_body_length(mbl1)
                gpfi.set_min_body_length(mbl2)
            

            self.nodes.append(gpfi)
            # if str(gpfi) not in self.nodes_by_name:
            #     self.nodes_by_name.add(str(gpfi))
            # else:
            #     self.nodes_by_name[gpfi.name()].append(gpfi)

        for node in self.nodes:
            
            if node.expectedChildren() == 0:
                self.terminals.append(node)
            elif not isinstance(node, WriteRegisterGPNode):
                self.nonterminals.append(node)

            if isinstance(node, WriteRegisterGPNode):
                self.registers.append(node)
            else:
                self.nonregisters.append(node)
                
            if node.expectedChildren() == 0 and not isinstance(node, ReadRegisterGPNode):
                self.constants.append(node)
            elif isinstance(node, ReadRegisterGPNode):
                self.nonconstants.append(node)

            if isinstance(node, FlowOperator):
                self.flowoperators.append(node)

    def clone(self):
        newset = self.__class__()
        newset.name = self.name
        newset.nodes = [ n.lightClone() for n in self.nodes]
        # self.nodes_by_name:Set[str] = set()

        newset.nonterminals = [n.lightClone() for n in self.nonterminals]
        newset.terminals = [ n.lightClone() for n in self.terminals ]

        newset.registers = [n.lightClone() for n in self.registers]
        newset.nonregisters = [n.lightClone() for n in self.nonregisters]

        newset.constants = [n.lightClone() for n in self.constants]
        newset.nonconstants = [n.lightClone() for n in self.nonconstants]

        newset.flowoperators = [n.lightClone() for n in self.flowoperators]
        return newset