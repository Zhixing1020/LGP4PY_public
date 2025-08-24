from src.ec import *
from src.ec.util import *
from src.ec.gp_individual import GPIndividual
    
from abc import ABC, abstractmethod
from typing import List, Optional, Set, override, Union

from tasks.problem import Problem
from src.lgp.individual.gp_tree_struct import GPTreeStruct
from src.lgp.individual.primitive import *
# from src.lgp.util.linear_regression import LinearRegression

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

class LGPIndividual(GPIndividual):
    P_NUMREGISTERS = "numregisters"
    P_MAXNUMTREES = "maxnumtrees"
    P_MINNUMTREES = "minnumtrees"
    P_INITMAXNUMTREES = "init_maxnumtrees"
    P_INITMINNUMTREES = "init_minnumtrees"
    P_RATEFLOWOPERATOR = "rate_flowoperator"
    P_MAXITERTIMES = "max_itertimes"
    P_NUMOUTPUTREGISTERS = "num-output-register"
    P_OUTPUTREGISTER = "output-register"
    # P_FLOATOUTPUT = "to-float-outputs"
    P_EFFECTIVE_INITIAL = "effective_initial"
    INITIAL_VALUE = 0.0

    def __init__(self):
        super().__init__()
        self.privateParameter = None
        self.MaxNumTrees = 0
        self.MinNumTrees = 0
        self.initMaxNumTrees = 0
        self.initMinNumTrees = 0
        self.numRegs = 0
        self.numOutputRegs = 0
        self.eff_initialize = False
        self.registers = [] # each register can be a float or a numpy array
        self.wraplist = []
        # self.flowctrl = None
        # self.rateFlowOperator = 0.0
        # self.maxIterTimes = 0
        self.outputRegister = []
        self.fastFlag = False
        self.exec_trees:list[GPTreeStruct] = []
        self.preevaluated = False

        # self.tmp_numOutputRegs = 0
        # self.float_numOutputRegs = False
        # self.initReg = []
        # self.init_ConReg = []
        # self.initReg_values = []
        # self.constraintsNum = 0

    def setup(self, state: EvolutionState, base: Parameter):
        super().setup(state, base)
        def_base = self.defaultBase()

        self.evaluated = False

        # wrapper
        self.towrap = state.parameters.getBoolean(base.push(self.P_TOWRAP), def_base.push(self.P_TOWRAP), False)
        self.wraplist = []
        self.batchsize = state.parameters.getInt(base.push(self.P_BATCHSIZE), def_base.push(self.P_BATCHSIZE))
        if self.batchsize < 1:
            state.output.fatal("The wrap_max_sample must be larger than 1.",
                            base.push(self.P_BATCHSIZE), def_base.push(self.P_BATCHSIZE))

        # self.float_numOutputRegs = state.parameters.getBoolean(base.push(self.P_FLOATOUTPUT), def_base.push(self.P_FLOATOUTPUT), False)
        # self.normalize_wrap = state.parameters.getBoolean(base.push(self.P_NORMWRAP), def_base.push(self.P_NORMWRAP), False)
        # self.normalize_f = state.parameters.getDoubleWithDefault(base.push(self.P_NORMWRAP_F), def_base.push(self.P_NORMWRAP_F), 1e-3)

        self.MaxNumTrees = state.parameters.getInt(base.push(self.P_MAXNUMTREES), def_base.push(self.P_MAXNUMTREES))
        if self.MaxNumTrees <= 0:
            state.output.fatal("An LGPIndividual must have at least one tree.",
                            base.push(self.P_MAXNUMTREES), def_base.push(self.P_MAXNUMTREES))

        self.MinNumTrees = state.parameters.getInt(base.push(self.P_MINNUMTREES), def_base.push(self.P_MINNUMTREES))
        if self.MinNumTrees <= 0:
            state.output.fatal("An LGPIndividual must have at least one tree.",
                            base.push(self.P_MINNUMTREES), def_base.push(self.P_MINNUMTREES))

        self.initMaxNumTrees = state.parameters.getInt(base.push(self.P_INITMAXNUMTREES), def_base.push(self.P_INITMAXNUMTREES))
        if self.initMaxNumTrees <= 0:
            state.output.fatal("An LGPIndividual must have at least one tree.",
                            base.push(self.P_INITMAXNUMTREES), def_base.push(self.P_INITMAXNUMTREES))

        self.initMinNumTrees = state.parameters.getInt(base.push(self.P_INITMINNUMTREES), def_base.push(self.P_INITMINNUMTREES))
        if self.initMinNumTrees <= 0:
            state.output.fatal("An LGPIndividual must have at least one tree.",
                            base.push(self.P_INITMINNUMTREES), def_base.push(self.P_INITMINNUMTREES))
        if self.initMinNumTrees > self.initMaxNumTrees or self.initMinNumTrees < self.MinNumTrees:
            state.output.fatal("The initMinNumTrees must be >= MinNumTrees and <= initMaxNumTrees.")

        self.numRegs = state.parameters.getInt(base.push(self.P_NUMREGISTERS), def_base.push(self.P_NUMREGISTERS))
        if self.getNumRegs() <= 0:
            state.output.fatal("An LGPIndividual must have at least one register.",
                            base.push(self.P_NUMREGISTERS), def_base.push(self.P_NUMREGISTERS))

        # self.rateFlowOperator = state.parameters.getDoubleWithDefault(base.push(self.P_RATEFLOWOPERATOR), def_base.push(self.P_RATEFLOWOPERATOR), 0.)
        # if not (0 <= self.rateFlowOperator <= 1):
        #     state.output.fatal("The rate of flow operator must be >=0 and <=1.",
        #                     base.push(self.P_RATEFLOWOPERATOR), def_base.push(self.P_RATEFLOWOPERATOR))

        # self.maxIterTimes = state.parameters.getIntWithDefault(base.push(self.P_MAXITERTIMES), def_base.push(self.P_MAXITERTIMES), 100)
        # if self.maxIterTimes <= 0:
        #     state.output.fatal("max iteration times must be >=1",
        #                     base.push(self.P_MAXITERTIMES), def_base.push(self.P_MAXITERTIMES))

        self.eff_initialize = state.parameters.getBoolean(base.push(self.P_EFFECTIVE_INITIAL), def_base.push(self.P_EFFECTIVE_INITIAL), False)

        self.numOutputRegs = state.parameters.getIntWithDefault(base.push(self.P_NUMOUTPUTREGISTERS), def_base.push(self.P_NUMOUTPUTREGISTERS), 1)
        if self.numOutputRegs <= 0:
            state.output.fatal("An LGPIndividual must have at least one output register.",
                            base.push(self.P_NUMOUTPUTREGISTERS), def_base.push(self.P_NUMOUTPUTREGISTERS))

        self.outputRegister = [r for r in range(self.numOutputRegs)]
        # self.tmp_numOutputRegs = self.numOutputRegs

        self.treelist = []
        self.exec_trees = []

        for x in range(self.MaxNumTrees):
            p = base.push(self.P_TREE).push("0")
            self.privateParameter = p
            t: GPTreeStruct = state.parameters.getInstanceForParameter(
                p, def_base.push(self.P_TREE).push("0"), GPTreeStruct)
            t.owner = self
            t.status = False
            t.effRegisters = set()
            t.setup(state, p)
            self.treelist.append(t)

        self.setRegisters([0.0] * self.getNumRegs())

        # self.flowctrl = LGPFlowController()
        # self.getFlowctrl().maxIterTimes = self.maxIterTimes

        # x = 0
        # for tree in self.getTreelist():
        #     for w in range(len(tree.constraints(initializer).functionset.nodes)):
        #         gpfi = tree.constraints(initializer).functionset.nodes[w]
        #         for y in range(len(gpfi)):
        #             gpfi[y].checkConstraints(state, x, self, base)
        #             x += 1


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

        if self.eff_initialize:
            self.removeIneffectiveInstr()
            trial = 100 * self.initMaxNumTrees
            while self.countStatus() < numtrees and trial > 0:
                tree = self.species.instr_prototype.lightClone()
                # tree = state.parameters.getInstanceForParameter(
                #     self.privateParameter,
                #     self.defaultBase().push(self.P_TREE).push("0"),
                #     GPTreeStruct
                # )
                tree.buildTree(state, thread)
                self.addTree(0, tree)
                self.updateStatus()
                self.removeIneffectiveInstr()
                trial -= 1

    @override
    def execute(self, state:EvolutionState, thread:int, input:GPData, individual, 
                problem:Problem, with_warp:bool = False):
        # check if the individual is evaluated
        # if self.evaluated:
        #     return  [ self.getRegistersIndex(r) for r in self.getOutputRegisters()] 
        
        # reset the registers
        if not input.to_vectorize:
            self.resetRegisters(problem, self.INITIAL_VALUE)
        else:
            self.resetRegisters(problem, np.full((len(input.values),1), self.INITIAL_VALUE))

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

        if self.IsWrap() and with_warp:
            # if the individual is wrapped, we need to execute the wrapper
            for instr in self.wraplist:
                instr.child.eval(state, thread, input, individual, problem)

        return [ self.getRegistersIndex(r) for r in self.getOutputRegisters()] 
    
    @override
    def preExecution(self, state:EvolutionState, thread:int):
        
        # extract the executable trees
        self.exec_trees.clear()
        for tree in self.getTreelist():
            if tree.status:
                # tree.flatten_postorder(state, thread)
                self.exec_trees.append(tree)

        # set the fastFlag to 1 if the individual is fast executable (i.e., no flow control)
        self.fastFlag = all(tree.type == GPTreeStruct.ARITHMETIC for tree in self.exec_trees)
        self.preevaluated = True

    def getRegisters(self)->Union [List[float], List[np.array]]:
        return self.registers

    def getRegistersIndex(self, i: int) -> Union[float, np.array]:
        return self.registers[i]

    def getOutputRegisters(self):
        return self.outputRegister

    def setOutputRegisters(self, tar: List[int]):
        self.outputRegister = tar.copy()

    def getNumRegs(self) -> int:
        return self.numRegs

    def getNumOutputRegs(self) -> int:
        return self.numOutputRegs
    
    def getMaxNumTrees(self) -> int:
        return self.MaxNumTrees

    def getMinNumTrees(self) -> int:
        return self.MinNumTrees

    def getInitMaxNumTrees(self) -> int:
        return self.initMaxNumTrees

    def getInitMinNumTrees(self) -> int:
        return self.initMinNumTrees

    # def getFlowController(self):
    #     return self.getFlowctrl()

    # def getrateFlowOperator(self) -> float:
    #     return self.rateFlowOperator

    def resetIndividual(self, numReg: int, maxIterTime: int, outReg: Optional[List[int]] = None):
        self.numRegs = numReg
        self.maxIterTimes = maxIterTime
        self.evaluated = self.preevaluated = False

        self.setRegisters([self.INITIAL_VALUE for _ in range(self.numRegs)])

        # self.flowctrl = LGPFlowController()
        # self.flowctrl.maxIterTimes = self.maxIterTimes

        self.treelist = []

        if outReg is None:
            outReg = [0]

        self.outputRegister = outReg.copy()
        self.numOutputRegs = len(self.outputRegister)
        self.tmp_numOutputRegs = self.numOutputRegs

    def setRegister(self, ind:int, value: Union[float, np.array]):
        self.registers[ind] = value

    def setRegisters(self, registers: Union[list[float], list[np.array]]):
        if len(registers) != self.numRegs:
            print(f"We are assigning multiple registers with inconsistent number to the original register number\n")
            exit(1)
        self.registers = registers.copy()

    def resetRegisters(self, problem: Problem, value:Union[float, np.array]=0.0):
        if isinstance(value, float):
            self.setRegisters([value]*self.getNumRegs())
        elif isinstance(value, np.ndarray):
            regs = []
            for _ in range(self.getNumRegs()):
                regs.append(value.copy())
            self.setRegisters(regs)

    def printTrees(self, state: EvolutionState=None)->str:
        x = 0
        res = ""
        for x, tree in enumerate(self.treelist):
            if not tree.status:
                res += "//"
            res += (f"Ins {x}:\t{str(tree)}\n")

        if self.towrap:
            length = len(self.treelist)
            for x, tree in enumerate(self.wraplist):
                res += (f"Ins {x+length}:\t{str(tree)}\n")
        
        return res
    
    def printIndividualForHuman(self, state: 'EvolutionState')->str:
        res = self.EVALUATED_PREAMBLE + ("true" if self.evaluated else "false") + "\n"
        res += f"Fitness:\t[{ str( self.fitness )}]\n"
        res += self.printTrees(state)
        cnteff = self.countStatus()
        res += (f"# Effective instructions:\t{cnteff}\teffective %:\t{(cnteff / len(self.treelist)) * 100}\n")

        return res

    def getTreelist(self):
        return self.treelist

    def clone(self):
        # a deep clone
        myobj: LGPIndividual = super().clone()
        # myobj.treelist = []
        # for tree in self.getTreelist():
        #     t = tree.clone()
        #     t.owner = myobj
        #     myobj.getTreelist().append(t)
        myobj.copyLGPproperties(self)
        return myobj

    def copyLGPproperties(self, obj: 'LGPIndividual'):
        self.numRegs = len(obj.getRegisters())
        self.MaxNumTrees = obj.getMaxNumTrees()
        self.MinNumTrees = obj.getMinNumTrees()
        self.initMaxNumTrees = obj.getInitMaxNumTrees()
        self.initMinNumTrees = obj.getInitMinNumTrees()
        self.numOutputRegs = obj.getNumOutputRegs()
        self.outputRegister = obj.getOutputRegisters().copy()
        self.numOutputRegs = obj.getNumOutputRegs()
        self.towrap = obj.towrap
        self.batchsize = obj.batchsize
        self.eff_initialize = obj.eff_initialize
        self.setRegisters(obj.getRegisters())
        self.species = obj.species
        # self.flowctrl = LGPFlowController()
        # self.maxIterTimes = obj.getFlowController().maxIterTimes
        # self.flowctrl.maxIterTimes = self.maxIterTimes
        # self.rateFlowOperator = obj.getrateFlowOperator()

        self.wraplist = []
        for tree in obj.getWrapper():
            t = tree.clone()
            t.owner = self
            self.wraplist.append(t)

        # self.tmp_numOutputRegs = obj.getCurNumOutputRegs()
        # self.float_numOutputRegs = obj.isFloatingOutputs()

    def lightClone(self):
        myobj: LGPIndividual = super().lightClone()
        # myobj.treelist = []
        # for tree in self.getTreelist():
        #     t = tree.lightClone()
        #     t.owner = myobj
        #     myobj.getTreelist().append(t)
        myobj.copyLGPproperties(self)
        return myobj

    def size(self) -> int:
        return sum(tree.child.numNodes(GPNode.NODESEARCH_ALL) for tree in self.getTreelist())
    
    def getTree(self, index: int) -> 'GPTree':
        if index >= len(self.getTreelist()):
            print(f"The tree index {index} is out of range")
            exit(1)
        return self.getTreelist()[index]
    
    def setTree(self, index: int, tree: 'GPTreeStruct', update_status:bool=True) -> bool:
        if index < len(self.getTreelist()):
            # self.getTreelist().pop(index)
            treeStr: GPTreeStruct = tree if isinstance(tree, GPTreeStruct) else GPTreeStruct()
            if not isinstance(tree, GPTreeStruct):
                treeStr.assignfrom(tree)
                treeStr.status = False
                treeStr.effRegisters = set()
            # treeStr.type = GPTreeStruct.ARITHMETIC
            # if isinstance(treeStr.child.children[0], FlowOperator):
            #     treeStr.type = GPTreeStruct.BRANCHING if isinstance(treeStr.child.children[0], Branching) else GPTreeStruct.ITERATION
            # self.getTreelist().insert(index, treeStr)
            self.treelist[index] = treeStr
            self.evaluated = self.preevaluated = False
            if update_status:
                self.updateStatus()
            return True
        print(f"setTree index: {index} is out of range {len(self.getTreelist())}")
        return False

    def addTree(self, index: int, tree: 'GPTree', update_status:bool=True):
        treeStr = tree if isinstance(tree, GPTreeStruct) else GPTreeStruct()
        if not isinstance(tree, GPTreeStruct):
            treeStr.assignfrom(tree)
            treeStr.status = False
            treeStr.effRegisters = set()
        # treeStr.type = GPTreeStruct.ARITHMETIC
        # if isinstance(treeStr.child.children[0], FlowOperator):
        #     treeStr.type = GPTreeStruct.BRANCHING if isinstance(treeStr.child.children[0], Branching) else GPTreeStruct.ITERATION
        if index < 0:
            index = 0
        if index < len(self.getTreelist()):
            self.getTreelist().insert(index, treeStr)
        else:
            self.getTreelist().append(treeStr)
        self.evaluated = self.preevaluated = False
        if update_status:
            self.updateStatus()

    def removeTree(self, index: int, update_status:bool=True) -> bool:
        if index < len(self.getTreelist()):
            self.getTreelist().pop(index)
            self.evaluated = self.preevaluated = False
            if update_status:
                self.updateStatus()
            return True
        print(f"removeTree index: {index} is out of range {len(self.getTreelist())}")
        return False

    def removeIneffectiveInstr(self) -> bool:
        ii = 0
        while ii < self.getTreesLength():
            if not self.getTree(ii).status and self.getTreesLength() > self.getMinNumTrees():
                self.removeTree(ii)
            else:
                ii += 1
        return True

    def getTreesLength(self) -> int:
        return len(self.getTreelist())

    def getEffTreesLength(self, update_status:bool=True) -> int:
        if update_status:
            self.updateStatus()
        return sum(1 for tree in self.getTreelist() if tree.status)

    def getAvgNumEffFun(self) -> float:
        self.updateStatus()
        eff_trees = [tree for tree in self.getTreelist() if tree.status]
        if not eff_trees:
            return 0.0
        total = sum(tree.child.numNodes(GPNode.NODESEARCH_NONTERMINALS) for tree in eff_trees)
        return total / len(eff_trees)

    def getAvgNumFun(self) -> float:
        trees = self.getTreelist()
        if not trees:
            return 0.0
        total = sum(tree.child.numNodes(GPNode.NODESEARCH_NONTERMINALS) for tree in trees)
        return total / len(trees)

    def getNumEffNode(self) -> float:
        self.updateStatus()
        return sum(
            tree.child.numNodes(GPNode.NODESEARCH_ALL) -
            tree.child.numNodes(GPNode.NODESEARCH_READREG) - 1
            for tree in self.getTreelist() if tree.status
        )
    
    def getConditionIndex(self, state:EvolutionState, thread:int, predicate)->int:
        res = state.random[thread].randint(0, len(self.treelist)-1)
        if predicate(self.getTree(res)):
            return res
        
        indices:list[int] = []
        for i, x in enumerate(self.treelist):
            if predicate(x):
                indices.append(i)

        if len(indices) == 0:
            return state.random[thread].randint(0, len(self.treelist)-1)

        return state.random[thread].choice(indices)

    def getProgramSize(self) -> float:
        return self.getNumEffNode()
    
    def updateStatus(self, n: Optional[int] = None, tar: Optional[List[int]] = None):
        '''
        identify which instructions are extrons and vise versa
        start to update the status from position n
        tar: the output register
        '''
        if n is None:
            n = len(self.getTreelist())
        if tar is None:
            tar = self.getOutputRegisters()

        if n > len(self.getTreelist()):
            print("The n in updateStatus is larger than existing tree list")
            exit(1)

        self.preevaluated = False
        statusArray = [False] * len(self.getTreelist())
        sourceArray = [[False] * self.getNumRegs() for _ in self.getTreelist()]
        destinationArray = [[False] * self.getNumRegs() for _ in self.getTreelist()]

        targetRegister = set(tar)

        # Move iterator to position n
        treelist = self.getTreelist()
        idx = n - 1

        while idx >= 0:
            tree:GPTreeStruct = treelist[idx]
            tree.effRegisters = set(targetRegister)

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
            if index in targetRegister:
                tree.status = statusArray[idx] = True
                targetRegister.discard(index)
                tree.updateEffRegister(targetRegister)
            else:
                tree.status = statusArray[idx] = False

            destinationArray[idx][index] = True
            self.collectReadRegister(tree.child, sourceArray[idx])
            idx -= 1

    def countStatus(self, start: int=0, end: int=None) -> int:
        if end is None:
            end = self.getTreesLength()
        return sum(1 for tree in self.getTreelist()[start:end] if tree.status)
    
    def canAddFlowOperator(self) -> bool:
        return True
    #     cnt = sum(1 for tree in self.getTreelist() if isinstance(tree.child.children[0], FlowOperator))
    #     return cnt / len(self.getTreelist()) <= self.rateFlowOperator

    def collectReadRegister(self, node: GPNode, collect: List[bool]):
        if isinstance(node, ReadRegisterGPNode):
            collect[node.getIndex()] = True
        else:
            for child in node.children:
                self.collectReadRegister(child, collect)

    def makeGraphvizRule(self) -> str:
        return self.makeGraphvizRule(list(self.getOutputRegisters()))

    def wrapper(self, predict_array: np.ndarray, target_array: np.ndarray, 
            state: EvolutionState, thread: int, problem: Problem):
        
        #predict_array.shape = (N, F) where N = number of instances (samples) F = number of features
        #target_array.shape = (N, O) where O = number of outputs (targets)

        MAX_SAMPLE = self.batchsize

        num_samples, num_features = predict_array.shape
        num_outputs = target_array.shape[1]
        sample_size = min(num_samples, MAX_SAMPLE)

        # sample indices
        if num_samples > MAX_SAMPLE:
            indices = np.array([
                state.random[thread].randint(0, num_samples - 1)
                for _ in range(MAX_SAMPLE)
            ])
        else:
            indices = np.arange(sample_size)

        # sampled predictors and targets
        predict = predict_array[indices, :]         # shape (sample_size, num_features)
        self.wraplist.clear()

        for tar in range(num_outputs):
            # build target vector for this output
            target = target_array[indices, tar]     # shape (sample_size,)

            # fit linear regression
            lr = Ridge(alpha=0.1)
            lr.fit(predict, target)

            # combine intercept + coefficients
            W = np.concatenate(([lr.intercept_], lr.coef_))

            # construct instruction
            instr = self.constructInstr(self.outputRegister[tar], W)
            self.wraplist.append(instr)

            # update all predictions inplace
            tmp = W[0] + predict_array @ W[1:]   # shape (num_samples,)
            tmp = np.clip(tmp, -1e6, 1e6)
            predict_array[:, tar] = tmp

        # return updated predictions
        return predict_array.copy()


    def getWrapper(self):
        return self.wraplist
    
    def constructInstr(self, out_index: int, W: List[float]) -> GPTreeStruct:
        cand = GPTreeStruct()

        des_reg = WriteRegisterGPNode()
        des_reg.setIndex(out_index)
        des_reg.argposition = 0
        N = des_reg
        cand.child = N
        N.children = [None]

        for r in range(len(self.outputRegister) + 1):
            n = self.Add_Mul_Coef_R(W, r)
            N.children[0] = n
            N = n

        return cand
    
    def Add_Mul_Coef_R(self, W: List[float], index: int) -> GPNode:
        if index < len(W) - 1:
            A:ConstantGPNode = ConstantGPNode()
            A.setValue(W[index + 1])
            A.argposition = 0

            src_reg = ReadRegisterGPNode()
            src_reg.setIndex(self.outputRegister[index])
            src_reg.argposition = 0

            n = Add()
            n.children = [None, None]

            mul = Mul()
            mul.children = [A, src_reg]
            n.children[1] = mul

            return n
        else:
            A = ConstantGPNode()
            A.setValue(W[0])
            A.argposition = 0
            return A
        
    def makeGraphvizInstr(self, instr_index: int, inputs: Set[str], used_terminals: List[str],
                      not_used: Set[int], cntindex: 'AtomicInteger') -> str:
        connection = ""
        tree = self.getTree(instr_index)

        out_idx = tree.child.getIndex()
        if out_idx in not_used:
            connection += f"R{out_idx}[shape=box];\n"
            connection += f"R{out_idx}->" + str(instr_index) + ";\n"
            not_used.remove(out_idx)

        for c, child in enumerate(tree.child.children[0].children):
            connection += self.makeGraphvizSubInstr(child, instr_index, c,
                                                    inputs, used_terminals, str(instr_index), cntindex)
        return connection
    
    def makeGraphvizSubInstr(self, node: GPNode, instr_index: int, child_index: int,
                          inputs: Set[str], used_terminals: List[str], parent_label: str,
                          cntindex: 'AtomicInteger') -> str:
        res = ""
        if isinstance(node, (InputFeatureGPNode)):
            label = str(node)
            if label not in inputs:
                inputs.add(label)
                res += f"{label}[shape=box];\n"
            res += f"{parent_label}->{label}[label=\"{child_index}\"];\n"
            return res

        # if isinstance(node, Entity) and node.expectedChildren() == 0:
        #     label = node.toGraphvizString()
        #     if label not in inputs:
        #         inputs.add(label)
        #         res += f"\"{label}\"[shape=box];\n"
        #     res += f"{parent_label}->\"{label}\"[label=\"{child_index}\"];\n"
        #     return res

        if isinstance(node, ReadRegisterGPNode):
            reg_idx = node.getIndex()
            for j in range(instr_index - 1, -1, -1):
                visit:GPTreeStruct = self.getTree(j)
                if not visit.status:
                    continue
                if visit.child.getIndex() == reg_idx:
                    res += f"{parent_label}->{j}[label=\"{child_index}\"];\n"
                    break
            else: # if there is still source registers, connect the instruction with initial values
                if used_terminals[reg_idx] not in inputs:
                    inputs.add(used_terminals[reg_idx])
                    res += f"{used_terminals[reg_idx]}[shape=box];\n"
                res += f"{parent_label}->{used_terminals[reg_idx]}[label=\"{child_index}\"];\n"
            return res

        label = str(cntindex.value)
        cntindex.value = cntindex.get() + 1
        res += f"{label}[label=\"{node.toGraphvizString()}\"];\n"
        res += f"{parent_label}->{label}[label=\"{child_index}\"];\n"
        for x, child in enumerate(node.children):
            res += self.makeGraphvizSubInstr(child, instr_index, x, inputs, used_terminals, label, cntindex)
        return res



from dataclasses import dataclass

@dataclass
class AtomicInteger:
    value:int = 0

@dataclass
class LGPDefaults:
    base:Parameter = Parameter("lgp")