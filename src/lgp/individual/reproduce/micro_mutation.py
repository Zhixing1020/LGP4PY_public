from .mutation import MutationPipeline
from src.lgp.individual import *
from src.ec import *
from src.ec.util import *
from src.lgp.individual.primitive import *

class LGPMicroMutationPipeline(MutationPipeline):
    P_STEP = "step"  # the number of free mutation step size
    MICROMUT = "micromut"
    P_EFFFLAG = "effective"
    P_PROBFUNC = "probfunc"
    P_PROBCONS = "probcons"
    P_PROBWRIREG = "probwritereg"
    P_PROBREADREG = "probreadreg"
    P_CONSTSTEP = "conststep"
    
    functions = 0
    cons = 1
    writereg = 2
    readreg = 3
    
    def __init__(self):
        super().__init__()
        self.stepSize = 0
        self.effflag = False
        self.p_function = 0.0
        self.p_constant = 0.0
        self.p_writereg = 0.0
        self.p_readreg = 0.0
        self.cons_step = 0
        self.componenttype = 0
    
    def setup(self, state:EvolutionState, base:Parameter):
        super().setup(state, base)
        
        def_ = LGPDefaults.base.push(self.MICROMUT)
        
        self.stepSize = state.parameters.getInt(base.push(self.P_STEP), def_.push(self.P_STEP))
        if self.stepSize < 1:
            state.output.fatal("LGPFreeMutation Pipeline has an invalid number of step size (it must be >= 1).",
                             base.push(self.P_STEP), def_.push(self.P_STEP))
        
        self.effflag = state.parameters.getBoolean(base.push(self.P_EFFFLAG), def_.push(self.P_EFFFLAG), True)
        
        self.p_function = state.parameters.getDoubleWithDefault(base.push(self.P_PROBFUNC), def_.push(self.P_PROBFUNC), 0.25)
        if self.p_function == -1:
            state.output.fatal("LGPFreeMutation Pipeline has an invalid number of probfunc (it must be >= 0).",
                             base.push(self.P_PROBFUNC), def_.push(self.P_PROBFUNC))
        
        self.p_constant = state.parameters.getDoubleWithDefault(base.push(self.P_PROBCONS), def_.push(self.P_PROBCONS), 0.25)
        if self.p_constant == -1:
            state.output.fatal("LGPFreeMutation Pipeline has an invalid number of probcons (it must be >= 0).",
                             base.push(self.P_PROBCONS), def_.push(self.P_PROBCONS))
        
        self.p_writereg = state.parameters.getDoubleWithDefault(base.push(self.P_PROBWRIREG), def_.push(self.P_PROBWRIREG), 0.25)
        if self.p_writereg == -1:
            state.output.fatal("LGPFreeMutation Pipeline has an invalid number of probwritereg (it must be >= 0).",
                             base.push(self.P_PROBWRIREG), def_.push(self.P_PROBWRIREG))
        
        self.p_readreg = state.parameters.getDoubleWithDefault(base.push(self.P_PROBREADREG), def_.push(self.P_PROBREADREG), 0.25)
        if self.p_readreg == -1:
            state.output.fatal("LGPFreeMutation Pipeline has an invalid number of probreadreg (it must be >= 0).",
                             base.push(self.P_PROBREADREG), def_.push(self.P_PROBREADREG))
        
        self.cons_step = state.parameters.getInt(base.push(self.P_CONSTSTEP), def_.push(self.P_CONSTSTEP))
        if self.cons_step <= 0:
            state.output.fatal("LGPFreeMutation Pipeline has an invalid number of cons step (it must be > 0).",
                             base.push(self.P_CONSTSTEP), def_.push(self.P_CONSTSTEP))
    
    def produce(self, min, max, start, subpopulation, inds:list[LGPIndividual], state:EvolutionState, thread)->int:
        # grab individuals from our source
        n = self.sources[0].produce(min, max, start, subpopulation, inds, state, thread)

        # should we bother?
        # if not state.random[thread].nextBoolean(self.likelihood):
        #     return self.reproduce(n, start, subpopulation, inds, state, thread, False)

        # initializer = state.initializer
        
        # now let's mutate them
        for q in range(start, n + start):
            i = inds[q]
            inds[q] = self.produce_individual(subpopulation, i, state, thread)

        return n

    
    def produce_individual(self, subpopulation, ind, state, thread)-> LGPIndividual:
        # initializer = state.initializer

        parent:LGPIndividual = ind

        if (self.tree != self.TREE_UNFIXED and 
            (self.tree < 0 or self.tree >= parent.getTreesLength())):
            state.output.fatal("LGP Mutation Pipeline attempted to fix tree.0 to a value which was out of bounds of the array of the individual's trees. " \
            "Check the pipeline's fixed tree values -- they may be negative or greater than the number of trees in an individual")

        # get the function set
        set:GPPrimitiveSet = parent.species.primitiveset  # all trees have the same function set
        
        # get the mutation component
        self.randomGetComponentType(state, thread)

        j = None
        if isinstance(self.sources[0], BreedingPipeline):
            # it's already a copy
            j = parent
        else:
            # need to clone the individual
            j = parent.lightClone()
        
        # double pickNum = Math.max(state.random[thread].uniform(0, 1)*(i.getTreesLength()), 1)
        pickNum = state.random[thread].randint(0, self.stepSize-1) + 1.0
        for pick in range(int(pickNum)):
            # t = self.getLegalMutateIndex(j, state, thread)
            
            # validity result
            res = False
            
            # pick nodes
            p1 = None
            p2 = None
            cnt = 0  # number of primitives that satisfy the given component type
            cntdown = 0
            flag = -1  # whether it needs to reselect p1
            
            for x in range(self.numTries):
                t = self.getLegalMutateIndex(j, state, thread)
                self.randomGetComponentType(state, thread)
                
                # pick random tree
                if self.tree != self.TREE_UNFIXED:
                    t = self.tree
                
                # prepare the nodeselector
                self.nodeselect.reset()
                
                p1:GPNode = None
                p2:GPNode = None
                cnt = 0
                cntdown = 0
                oriTree = parent.getTree(t)
                flag = -1
                
                if self.componenttype == self.functions:
                    flag = GPNode.NODESEARCH_NONTERMINALS
                elif self.componenttype == self.cons:
                    flag = GPNode.NODESEARCH_CONSTANT
                elif self.componenttype == self.writereg:
                    flag = -1
                    cnt = 1
                    p1 = oriTree.child
                elif self.componenttype == self.readreg:
                    flag = GPNode.NODESEARCH_READREG
                
                if flag >= 0:
                    cnt = oriTree.child.numNodes(flag)
                
                if flag >= 0 and cnt > 0:
                    pick = GPNodeGather()
                    oriTree.child.nodeInPosition(state.random[thread].randint(0, cnt-1), pick, flag)
                    p1 = pick.node
                
                # size = GPNodeBuilder.NOSIZEGIVEN
                # if self.equalSize:
                #     size = p1.numNodes(GPNode.NODESEARCH_ALL) if p1 else 0
                
                if cnt > 0:
                    if self.componenttype == self.functions:
                        p2 = set.nonterminals[state.random[thread].randint(0,len(set.nonterminals)-1)].lightClone()
                        p2.resetNode(state, thread)
                    elif self.componenttype == self.cons:
                        if state.random[thread].uniform(0,1) < self.builder.probCons:
                            p2 = set.constants[state.random[thread].randint(0,len(set.constants)-1)].lightClone()
                            p2.resetNode(state, thread)
                        else:
                            p2 = set.nonconstants[state.random[thread].randint(0,len(set.nonconstants)-1)].lightClone()
                            p2.resetNode(state, thread)
                    elif self.componenttype == self.writereg:
                        p2 = set.registers[state.random[thread].randint(0,len(set.registers)-1)].lightClone()
                        p2.resetNode(state, thread)
                    elif self.componenttype == self.readreg:
                        p2 = self.builder.newRootedTree(
                            state,
                            thread,
                            p1.parent,
                            parent.species.primitiveset,
                            p1.argposition,
                            p1.atDepth())
                else:
                    # no suitable instruction found
                    p1 = self.nodeselect.pickNode(state, subpopulation, thread, parent, parent.getTree(t))
                    p2 = self.builder.newRootedTree(
                        state,
                        thread,
                        p1.parent,
                        parent.species.primitiveset,
                        p1.argposition,
                        p1.atDepth())
                
                # check for validity
                res = self.check_points(p1, p2, state, thread, parent, parent.getTree(t))
                if res:
                    break
            
            if res:  # we've got a valid mutation
                x = t
                tree = j.getTree(x)
                tree = parent.getTree(x).clone()
                tree.owner = j
                tree.child = parent.getTree(x).child.cloneReplacingNoSubclone(p2, p1)
                tree.child.parent = tree
                tree.child.argposition = 0
                j.setTree(x, tree)
                j.evaluated = False
            else:
                x = t
                tree = j.getTree(x)
                tree = parent.getTree(x).clone()
                tree.owner = j
                tree.child = parent.getTree(x).child.clone()
                tree.child.parent = tree
                tree.child.argposition = 0
                j.setTree(x, tree)
        
        j.breedingPipe = self
        return j
    
    def check_points(self, p1:GPNode, p2:GPNode, state:EvolutionState, thread:int, ind:LGPIndividual, treeStr:GPTreeStruct):
        res = False
        
        if p1.expectedChildren() == p2.expectedChildren():
            if str(p1) != str(p2):
                if self.effflag and len(treeStr.effRegisters) > 0:
                    if p1.atDepth() == 0 and not p2.getIndex() in treeStr.effRegisters:
                        # guarantee effectiveness
                        eff_list = list(treeStr.effRegisters)
                        p2.setIndex(eff_list[state.random[thread].randint(0,len(eff_list)-1)])
                        if str(p1) == str(p2) and len(treeStr.effRegisters) > 1:
                            res = False
                        else:
                            res = True
                    else:
                        res = True
                else:
                    res = True
                
                # further check for constant value
                if isinstance(p1, ConstantGPNode) and isinstance(p2, ConstantGPNode):
                    if abs(p1.getValue() - p2.getValue()) > self.cons_step:
                        if p1.getValue() - p2.getValue() > 0:
                            p2.setValue(p1.getValue() - 1 - state.random[thread].randint(0,self.cons_step - 1))
                        else:
                            p2.setValue(p1.getValue() + 1 + state.random[thread].randint(0,self.cons_step - 1))
                    res = True
                
                # further check for flow operator
                if isinstance(p2, FlowOperator) and not ind.canAddFlowOperator():
                    res = False
                
                for c in range(len(p1.children)):
                    p2.children[c] = p1.children[c].clone()
                    p2.children[c].parent = p2
        
        return res
    
    def getLegalMutateIndex(self, ind:LGPIndividual, state:EvolutionState, thread:int):
        res = state.random[thread].randint(0,ind.getTreesLength()-1)
        
        if self.effflag:  # guarantee effectiveness
            res = ind.getConditionIndex(state, thread,
                                         lambda x: x.status)
            # if self.componenttype != self.cons:
                # for x in range(self.numTries):
                    # if ind.getTree(res).status:
                    #     break
                    # res = state.random[thread].randint(0,ind.getTreesLength()-1) 
            # else:
            if self.componenttype == self.cons:
                res = ind.getConditionIndex(state, thread, 
                                                 lambda x: x.status and x.child.numNodes(GPNode.NODESEARCH_CONSTANT) > 0)
                # for x in range(self.numTries):
                    # if (ind.getTree(res).status and 
                    #     ind.getTree(res).child.numNodes(GPNode.NODESEARCH_CONSTANT) > 0):
                    #     break
                    # res = state.random[thread].randint(0,ind.getTreesLength()-1)
        else:
            if self.componenttype == self.cons:
                res = ind.getConditionIndex(state, thread, 
                                                 lambda x: x.child.numNodes(GPNode.NODESEARCH_CONSTANT) > 0)
                # for x in range(self.numTries):
                #     if ind.getTree(res).child.numNodes(GPNode.NODESEARCH_CONSTANT) > 0:
                #         break
                #     res = state.random[thread].randint(0,ind.getTreesLength()-1)
        
        return res
    
    def randomGetComponentType(self, state:EvolutionState, thread:int):
        rnd = state.random[thread].uniform(0,1)
        if rnd > self.p_function + self.p_constant + self.p_writereg + self.p_readreg:
            self.componenttype = state.random[thread].randint(0, 3)
        elif rnd > self.p_constant + self.p_writereg + self.p_readreg:
            self.componenttype = self.functions
        elif rnd > self.p_writereg + self.p_readreg:
            self.componenttype = self.cons
        elif rnd > self.p_readreg:
            self.componenttype = self.writereg
        else:
            self.componenttype = self.readreg

