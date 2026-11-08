
from src.ec import *
from src.lgp.individual.lgp_individual import LGPIndividual, AtomicInteger
from src.lgp.individual.gp_tree_struct import GPTreeStruct
from tasks.supervisedproblem import SupervisedProblem
from typing import override, Optional, List
from src.lgp.algorithm.typed_lgp.individual.primitives.typed_register import TypedRegister
from src.lgp.algorithm.typed_lgp.individual.typed_gp_tree_struct import TypedGPTreeStruct
import numpy as np
from src.lgp.algorithm.typed_lgp.ec import *
import copy
from src.ec.util import *

init_reg_values:dict={} # shared initial values for all registers

class TypedLGPIndividual(LGPIndividual):

    P_REGISTER_PROTOTYPE = "register_prototype"
    P_INPUT_TYPE = "input_type"
    P_TARGET_TYPE = "target_type"
    

    def __init__(self):
        super().__init__()
        self.register_prototype:TypedRegister = None
        self.registers : list[TypedRegister] = []
        self.inputType = None
        self.outputType = None

    def setup(self, state: TypedEvolutionState, base: Parameter):
        
        def_base = self.defaultBase()

        # parameter get register prototype
        self.register_prototype = state.parameters.getInstanceForParameter(base.push(self.P_REGISTER_PROTOTYPE),
                                                                 def_base.push(self.P_REGISTER_PROTOTYPE),
                                                                 TypedRegister)

        self.register_prototype.setAgent(state.typeagent)

        # parameter get target type  (must be a type in agent.all types)

        input_type_name = state.parameters.getString(base.push(self.P_TARGET_TYPE),
                                               def_base.push(self.P_TARGET_TYPE))
        output_type_name = state.parameters.getString(base.push(self.P_TARGET_TYPE),
                                               def_base.push(self.P_TARGET_TYPE))
        self.inputType = state.typeagent.getTypeByName(input_type_name)
        self.outputType = state.typeagent.getTypeByName(output_type_name)

        super().setup(state, base)


    def getOutputType(self):
        return self.outputType
    
    def getInputType(self):
        return self.inputType
    
    def setRegister(self, ind:int, value, type=None):
        self.registers[ind].setValue(value, type)

    def setRegisters(self, values: list, type=None):
        """"
        because the value of a register is a dict of {type: value}, we need to specify the type and value.
        """
        if len(values) != self.numRegs:
            print(f"We are assigning multiple registers with inconsistent number to the original register number\n")
            exit(1)
        self.registers.clear()
        for i, item in enumerate(values): # for each register
            self.registers.append(self.register_prototype.lightClone_wo_val())
            if isinstance(item, TypedRegister):
                for k, v in item.value.items():
                    self.registers[i].setValue(v, k)
            else:
                self.registers[i].setValue(item, type)

    def resetRegisters(self, problem: SupervisedProblem, value=0.0, to_vectorize=False):
        if to_vectorize:
            # initialize registers as the background grid of the input
            # ini_array = np.zeros((len(value), 1))
            # for i, val in enumerate(value): # for every data instance, clone it into an np.ndarray.
            #     ini_val = val
            #     ini_array[i] = copy.deepcopy(ini_val)

            # regs = [] # a list of np.ndarray, each array includes the initial values for all examples
            # for _ in range(self.getNumRegs()):
            #     regs.append(copy.deepcopy(ini_array)) # broadcast to all registers

            # self.setRegisters(regs, self.outputType)
            raise Exception("typed LGP individual does not support vectorization")
        else:
            ini_val = value
            
            if len(self.registers) < self.getNumRegs():
                for r in range(self.getNumRegs()):
                    self.registers.append(self.register_prototype.lightClone_wo_val())

            for r in self.to_reset_register_set:   # r in range(self.getNumRegs())
                # self.registers[r] = self.register_prototype.lightClone_wo_val()
                self.setRegister(r, ini_val, self.inputType)


    # @staticmethod
    # def get_initial_value(data:tuple)->tuple:
    #     initial_val = np.array(data) # initialize the value as the input grid
    #     return tuple(map(tuple, initial_val))

    @override
    def rebuildIndividual(self, state, thread):

        numtrees = state.random[thread].randint(self.initMinNumTrees, self.initMaxNumTrees)

        self.treelist.clear()

        for _ in range(numtrees):
            tree = self.species.instr_prototype.lightClone()
            # tree = state.parameters.getInstanceForParameter(
            #     self.privateParameter,
            #     self.defaultBase().push(self.P_TREE).push("0"),
            #     GPTreeStruct
            # )
            tree.buildTree(state, thread)
            self.treelist.append(tree)

        self.updateStatus()

        if self.eff_initialize: # for typed individual, to ensure all instructions are effective, 
            # we construct individuals reversely based on the effective type and register
            self.removeIneffectiveInstr()

            # targetTypes = set([self.getOutputType()])
            # targetRegisters = set(self.getOutputRegisters())
            
            trial = 10 * self.initMaxNumTrees
            while self.countStatus() < numtrees and trial > 0:

                # if self.getTreesLength() == 0:
                #     targetTypes = set([self.getOutputType()])
                #     targetRegisters = set(self.getOutputRegisters())
                # else:
                #     targetTypes = self.getTree(0).effTypes
                #     targetRegisters = self.getTree(0).effRegisters
                #     index = self.getTree(0).child.getIndex()
                #     type = self.getTree(0).child.children[0].getOutputType() # the output type of the toppest function

                #     targetRegisters.discard(index)
                #     tree.updateEffRegister(targetRegisters)

                #     targetTypes.discard(type)
                #     tree.updateEffType(targetTypes)

                # tree:TypedGPTreeStruct = self.species.instr_prototype.lightClone()
                # tree.buildTree(state, thread, targetType= state.random[thread].choice(list(targetTypes)))
                # tree.child.setIndex(state.random[thread].choice(list(targetRegisters)))

                tree = self.getFeasibleInstruction(state, thread, 0)

                self.addTree(0, tree)
                # self.updateStatus()
                self.removeIneffectiveInstr()
                trial -= 1

    def getFeasibleInstruction(self, state, thread, index) -> TypedGPTreeStruct:
        # index: the position of the instruction
        # return a feasible typed instruction for a certain position

        #first, find the precedent effective instruction
        old_tree:TypedGPTreeStruct = None
        while index < self.getTreesLength() and not self.getTree(index).status:
            index += 1
        if index < self.getTreesLength():
            old_tree = self.getTree(index)
            
        if self.getTreesLength() == 0 or old_tree is None:
            # sourceTypes = self.register_prototype.agent.getSourceTypes()
            targetTypes = set([self.getOutputType()])
            targetRegisters = set(self.getOutputRegisters())
            eff_input_type = self.register_prototype.agent.getSourceTypes()
        else:
            # sourceTypes = old_tree.effInputTypes
            targetTypes = old_tree.effTypes
            targetRegisters = old_tree.effRegisters
            index = old_tree.child.getIndex()
            type = old_tree.child.children[0].getOutputType() # the output type of the toppest function
            eff_input_type = old_tree.effInputTypes


            targetRegisters.discard(index)
            old_tree.updateEffRegister(targetRegisters)

            targetTypes.discard(type)
            old_tree.updateEffType(targetTypes)
        
        tree:TypedGPTreeStruct = self.species.instr_prototype.lightClone()

        targetType = state.random[thread].choice(list(targetTypes)) if len(targetTypes) > 0 else None
        # tree.buildTree(state, thread, targetType= targetType, sourceType=sourceTypes)
        tree.buildTree(state, thread, targetType= targetType, sourceType=eff_input_type)

        targetRegiter = state.random[thread].choice(list(targetRegisters)) if len(targetRegisters) > 0 \
            else state.random[thread].randint(0, tree.child.getRange())
        tree.child.setIndex(targetRegiter)

        return tree

    @override
    def execute(self, state, thread, input:TypedData, individual, problem:SupervisedProblem, with_wrap:bool = False):

        # reset the registers
        if not input.to_vectorize:
            # initialize the value as the background with the same shape of the input grid
            self.resetRegisters(problem, value=problem.X, to_vectorize=input.to_vectorize)
        else:
            # self.resetRegisters(problem, value=problem.data, to_vectorize=input.to_vectorize)
            raise Exception("typed LGP individual does not support vectorization")

        # check if the individual can be fast executed
        if not self.fastFlag:
            # normal execution and take consider the flow control
            # reset the flow controller
            # self.getFlowctrl().reset()
            pass
        
        if not self.preevaluated or not self.fastFlag:
            for tree in self.getTreelist():
                if tree.status:
                    tree.child.eval(state, thread, input, individual, problem)
                    # tree.postorder_execution(state, thread, input, individual, problem)
        else:
            for tree in self.exec_trees:
                tree.child.eval(state, thread, input, individual, problem)
                # tree.postorder_execution(state, thread, input, individual, problem)

        # if self.IsWrap() and with_warp:
        #     # if the individual is wrapped, we need to execute the wrapper
        #     for instr in self.wraplist:
        #         instr.child.eval(state, thread, input, individual, problem)

        # the type of all output registers is Grid
        for r in self.getOutputRegisters():
            self.getRegistersIndex(r).setType(self.outputType)

        # return [ self.getRegistersIndex(r).getValue() for r in self.getOutputRegisters()] 
        if len(self.getOutputRegisters()) > 1:
            raise Exception(f"there are more than one output register in typed lgp individual: {len(self.getOutputRegisters())}")

        output_r = self.getOutputRegisters()[0]
        return self.getRegistersIndex(output_r).getValue(self.outputType)

    def postExecution(self, state, thread):
        return
    
    def copyLGPproperties(self, obj: 'TypedLGPIndividual'):
        self.register_prototype = obj.register_prototype
        self.inputType = obj.inputType
        self.outputType = obj.outputType
        super().copyLGPproperties(obj)
    
    @override
    def makeGraphvizRule(self, outputRegs: list[int]) -> str:
        """
        Convert the LGPIndividual into a Graphviz-compatible string,
        visualizing only effective instructions and connections.
        """
        # # collect terminal names
        # usedTerminals = [str(self.initVal) for _ in range(self.getNumRegs())]
        
        # # set of seen terminals (used for deduplication)
        # SRInputs = set()

        # # specify nodes (effective instructions)
        # nodeSpec = ""
        # for i, tree in enumerate(self.getTreelist()):
        #     if not tree.status:
        #         continue
        #     nodeSpec += f'{i}[label="{tree.child.children[0].toGraphvizString()}"];\n'

        # # prepare to collect connections
        # connection = ""
        # notUsed = set(outputRegs)
        # cntindex = AtomicInteger(len(self.getTreelist()))

        # for i in reversed(range(len(self.getTreelist()))):
        #     tree = self.getTree(i)
        #     if not tree.status:
        #         continue
        #     connection += self.makeGraphvizInstr(i, SRInputs, usedTerminals, notUsed, cntindex)

        # # assemble graph
        # result = (
        #     "digraph g {\n"
        #     "nodesep=0.2;\n"
        #     "ranksep=0;\n"
        #     "node[fixedsize=true,width=1.3,height=0.6,fontsize=\"30\",fontname=\"times-bold\",style=filled, fillcolor=lightgrey];\n"
        #     "edge[fontsize=\"25.0\",fontname=\"times-bold\"];\n"
        #     f"{nodeSpec}"
        #     f"{connection}"
        #     "}\n"
        # )

        # return result

        return
    
    def updateStatus(self, n: Optional[int] = None, tar: Optional[List[int]] = None, tar_type:list=None):
        '''
        identify which instructions are extrons and vise versa
        start to update the status from position n
        tar: the output register
        '''
        if n is None:
            n = len(self.getTreelist())
        if tar is None:
            tar = self.getOutputRegisters()
        if tar_type is None:
            tar_type = self.getOutputType()

        if n > len(self.getTreelist()):
            print("The n in updateStatus is larger than existing tree list")
            exit(1)

        self.preevaluated = False
        # statusArray = [False] * len(self.getTreelist())
        # sourceArray = [[False] * self.getNumRegs() for _ in self.getTreelist()]
        # destinationArray = [[False] * self.getNumRegs() for _ in self.getTreelist()]

        targetRegister = set(tar)
        targetType = set([tar_type])

        # Move iterator to position n
        treelist = self.getTreelist()
        idx = n - 1

        while idx >= 0:
            tree:TypedGPTreeStruct = treelist[idx]
            tree.effRegisters = set(targetRegister)
            tree.effTypes = set(targetType)

            # if isinstance(tree.child.children[0], Branching):
            #     tree.type = GPTreeStruct.BRANCHING
            #     tree.status = statusArray[idx] = False
            #     start = idx + 1
            #     end = min(len(treelist), idx + 1 + tree.child.children[0].getBodyLength())

            #     # Check if body has effective instructions
            #     for i in range(start, end):
            #         if statusArray[i]:
            #             tree.status = statusArray[idx] = True
            #             break

            #     # Determine body end (including nested blocks)
            #     bodyend = tree.child.children[0].getBodyLength() + idx
            #     for i in range(idx + 1, min(len(treelist), bodyend + 1)):
            #         if isinstance(treelist[i].child.children[0], FlowOperator):
            #             bodyend = max(bodyend, i + treelist[i].child.children[0].getBodyLength())

            #     effi = bodyend + 1
            #     if effi >= len(treelist):
            #         targetRegister.update(tar)
            #     else:
            #         targetRegister.update(treelist[effi].effRegisters)
            #         if statusArray[effi] and treelist[effi].type == GPTreeStruct.ARITHMETIC:
            #             targetRegister.discard(treelist[effi].child.getIndex())
            #             treelist[effi].updateEffRegister(targetRegister)

            #     tree.effRegisters.update(targetRegister)

            #     if statusArray[idx]:
            #         tree.updateEffRegister(targetRegister)
            #     self.collectReadRegister(tree.child, sourceArray[idx])

            # elif isinstance(tree.child.children[0], Iteration):
            #     tree.type = GPTreeStruct.ITERATION
            #     source, destination = set(), set()
            #     effective_block_exist = False

            #     for i in range(idx + 1, len(treelist)):
            #         if i > idx + tree.child.children[0].getBodyLength():
            #             break
            #         if statusArray[i]:
            #             if isinstance(treelist[i].child.children[0], FlowOperator):
            #                 effective_block_exist = True
            #                 i += treelist[i].child.children[0].getBodyLength() - 1
            #             else:
            #                 for r in range(self.getNumRegs()):
            #                     if sourceArray[i][r]:
            #                         source.add(r)
            #                     if destinationArray[i][r]:
            #                         destination.add(r)

            #     source &= destination  # Intersection

            #     bodyend = tree.child.children[0].getBodyLength() + idx
            #     for i in range(idx + 1, min(len(treelist), bodyend + 1)):
            #         if isinstance(treelist[i].child.children[0], FlowOperator):
            #             bodyend = max(bodyend, i + treelist[i].child.children[0].getBodyLength())

            #     effi = bodyend + 1
            #     if effi >= len(treelist):
            #         targetRegister.update(tar)
            #     else:
            #         targetRegister.update(treelist[effi].effRegisters)
            #         if statusArray[effi] and treelist[effi].type == GPTreeStruct.ARITHMETIC:
            #             targetRegister.discard(treelist[effi].child.getIndex())
            #             treelist[effi].updateEffRegister(targetRegister)

            #     tree.effRegisters.update(targetRegister)

            #     if source or effective_block_exist:
            #         tree.status = statusArray[idx] = True
            #         tree.updateEffRegister(targetRegister)
            #     else:
            #         tree.status = statusArray[idx] = False

            #     self.collectReadRegister(tree.child, sourceArray[idx])

            # else:
            #     tree.type = GPTreeStruct.ARITHMETIC
            #     index = tree.child.getIndex()
            #     if index in targetRegister:
            #         tree.status = statusArray[idx] = True
            #         targetRegister.discard(index)
            #         tree.updateEffRegister(targetRegister)
            #     else:
            #         tree.status = statusArray[idx] = False

            #     destinationArray[idx][index] = True
            #     self.collectReadRegister(tree.child, sourceArray[idx])

            tree.type = GPTreeStruct.ARITHMETIC
            index = tree.child.getIndex()
            type = tree.child.children[0].getOutputType() # the output type of the toppest function
            if index in targetRegister and type in targetType:
                tree.status = True
                targetRegister.discard(index)
                tree.updateEffRegister(targetRegister)

                targetType.discard(type)
                tree.updateEffType(targetType)
            else:
                tree.status = False

            idx -= 1

        # self.updateForwardStatus(index=n-1)

    # def updateForwardStatus(self, index:int=None, source_type:set = None):
    #     # determine the forward effectiveness and effective input types at index-th instruction
    #     # forwardly check all instructions before position n
    #     # if an instruction accepts constants or effective input registers only, it is effective

    #     if index is None:
    #         index = self.getTreesLength() - 1

    #     if source_type is None:
    #         source_type = self.register_prototype.agent.getSourceTypes()

    #     eff_input_type_register = {self.register_prototype.agent.input_type : set(range(0, self.getNumRegs())) }

    #     tree:TypedGPTreeStruct = None
    #     for tree in self.getTreelist()[:index+1]:
    #         tree.effInputTypes = set(source_type)
    #         tree.effInputRegister = dict(eff_input_type_register)

    #         # check the output type of every readRegister, 
    #         output_type_reg = set()
    #         tree.child.collectInputTypes(output_type_reg)

    #         # if all the output type of readRegisters are in effective source_type, then this instruction is effective
    #         tree.forwardStatus = all(ot in source_type for ot in output_type_reg)

    #         # update the source_type
    #         if tree.forwardStatus:
    #             source_type.add(tree.child.getOutputType())
    #             if tree.child.getOutputType() in eff_input_type_register.keys():
    #                 eff_input_type_register[tree.child.getOutputType()].add(tree.child.getIndex())
    #             else:
    #                 eff_input_type_register.update({tree.child.getOutputType() : set([tree.child.getIndex()])})

      

    def getProcessingTypeSet(self)->set:
        types = set()
        treelist = self.getTreelist()
        n = len(treelist)
        idx = n - 1

        while idx >= 0:
            tree:TypedGPTreeStruct = treelist[idx]

            if tree.status == True:
                type = tree.child.children[0].getOutputType() # the output type of the toppest function
                if type in self.register_prototype.agent.get_all_basic_types().keys():
                    types.add(type)

            idx -= 1

        return types