from .mutation import MutationPipeline
from .micro_mutation import LGPMicroMutationPipeline
from src.lgp.individual import *
from src.ec import *
from src.ec.util import *
from src.lgp.individual.primitive import *

class LGPMacroMutationPipeline(MutationPipeline):
    P_MUT_TYPE = "type"
    P_STEP = "step"  # the number of free mutation step size
    P_INSERT = "prob_insert"
    P_DELETE = "prob_delete"
    MACROMUT = "macromut"
    P_MICROMUTBASE = "micro_base"
    
    FREEMACROMUT = 0
    EFFMACROMUT = 1
    EFFMACROMUT2 = 2
    EFFMACROMUT3 = 3
    
    def __init__(self):
        super().__init__()
        self.mutateType = ""
        self.mutateFlag = -1
        self.stepSize = 0
        self.probInsert = 0.0
        self.probDelete = 0.0
        self.microMutation:LGPMicroMutationPipeline = None
    
    def setup(self, state:EvolutionState, base:Parameter):
        super().setup(state, base)
        
        def_ = LGPDefaults.base.push(self.MACROMUT)
        
        self.mutateType = state.parameters.getString(base.push(self.P_MUT_TYPE),
                          def_.push(self.P_MUT_TYPE))
        if self.mutateType is None:
            state.output.fatal("LGPMacroMutation Pipeline has an invalid mutation type.",
                             base.push(self.P_MUT_TYPE), def_.push(self.P_MUT_TYPE))
        
        if self.mutateType == "freemut":
            self.mutateFlag = self.FREEMACROMUT
        elif self.mutateType == "effmut":
            self.mutateFlag = self.EFFMACROMUT
        elif self.mutateType == "effmut2":
            self.mutateFlag = self.EFFMACROMUT2
        elif self.mutateType == "effmut3":
            self.mutateFlag = self.EFFMACROMUT3
        
        self.stepSize = state.parameters.getInt(base.push(self.P_STEP), def_.push(self.P_STEP))
        if self.stepSize < 1:
            state.output.fatal("LGPMacroMutation Pipeline has an invalid number of step size (it must be >= 1).",
                             base.push(self.P_STEP), def_.push(self.P_STEP))
        
        self.probInsert = state.parameters.getDoubleWithDefault(base.push(self.P_INSERT), def_.push(self.P_INSERT), 0.67)
        if self.probInsert < 0:
            state.output.fatal("LGPMacroMutation Pipeline has an invalid number of prob_insert (it must be >= 0).",
                             base.push(self.P_INSERT), def_.push(self.P_INSERT))
        
        self.probDelete = state.parameters.getDoubleWithDefault(base.push(self.P_DELETE), def_.push(self.P_DELETE), 0.33)
        if self.probDelete < 0:
            state.output.fatal("LGPMacroMutation Pipeline has an invalid number of prob_delete (it must be >= 0).",
                             base.push(self.P_DELETE), def_.push(self.P_DELETE))
        
        microbase_str = state.parameters.getString(base.push(self.P_MICROMUTBASE), 
                            def_.push(self.P_MICROMUTBASE))
        microbase = Parameter(microbase_str) if microbase_str != None else None
        self.microMutation = None
        if microbase != None and str(microbase) != "null":
            self.microMutation = state.parameters.getInstanceForParameter(
                microbase, def_.push(self.P_MICROMUTBASE), MutationPipeline)
            self.microMutation.setup(state, microbase)
    
    def produce(self, min, max, start, subpopulation, inds:list[LGPIndividual], state:EvolutionState, thread)->int:
        # Grab individuals from our source
        n = self.sources[0].produce(min, max, start, subpopulation, inds, state, thread)

        # Should we bother?
        # if not state.random[thread].nextBoolean(self.likelihood):
        #     return self.reproduce(n, start, subpopulation, inds, state, thread, False)

        # initializer = state.initializer
        
        # Now let's mutate them
        for q in range(start, n + start):
            i = inds[q]
            inds[q] = self.produce_individual(subpopulation, i, state, thread)

        return n

    
    def produce_individual(self, subpopulation, ind, state, thread)->LGPIndividual:
        # initializer = state.initializer
            
        i:LGPIndividual = ind

        if (self.tree != self.TREE_UNFIXED and 
            (self.tree < 0 or self.tree >= i.getTreesLength())):
            state.output.fatal("LGP Mutation Pipeline attempted to fix tree.0 to a value which was out of bounds of the array of the individual's trees. Check the pipeline's fixed tree values -- they may be negative or greater than the number of trees in an individual")

        j:LGPIndividual = None
        if isinstance(self.sources[0], BreedingPipeline):
            # It's already a copy
            j = i
        else:
            # Need to clone the individual
            j = i.lightClone()
        
        # Copy all trees
        for v in range(i.getTreesLength()):
            x = v
            tree = j.getTree(x)
            tree = i.getTree(x).lightClone()
            tree.owner = j
            tree.child = i.getTree(x).child.clone()
            tree.child.parent = tree
            tree.child.argposition = 0
            j.setTree(x, tree, update_status=False)
        
        j.updateStatus()
        
        # Perform mutations
        pickNum = state.random[thread].randint(0, self.stepSize-1) + 1.0
        for _ in range(int(pickNum)):
            t = state.random[thread].randint(0, j.getTreesLength()-1)
            
            # Insert new instruction
            if (j.getTreesLength() < j.getMaxNumTrees() and 
                (state.random[thread].uniform(0, 1) < self.probInsert or 
                 j.getTreesLength() == j.getMinNumTrees())):
                
                t = self.getLegalInsertIndex(j, state, thread)
                
                res = False
                self.nodeselect.reset()
                
                p1 = j.getTree(t).child
                p2 = None
                
                for x in range(self.numTries):
                    # size = GPNodeBuilder.NOSIZEGIVEN
                    # if self.equalSize:
                    #     size = p1.numNodes(GPNode.NODESEARCH_ALL)
                    
                    p2 = self.builder.newRootedTree(
                            state,
                            thread,
                            p1.parent,
                            j.species.primitiveset,
                            p1.argposition,
                            p1.atDepth())
                    
                    # if isinstance(self.builder, LGPMutationGrowBuilder):
                    #     p2 = self.builder.newRootedTree(
                    #         state,
                    #         p1.parentType(initializer),
                    #         thread,
                    #         p1.parent,
                    #         j.getTree(t).constraints(initializer).functionset,
                    #         p1.argposition,
                    #         size,
                    #         p1.atDepth())
                    # else:
                    #     p2 = self.builder.newRootedTree(
                    #         state,
                    #         p1.parentType(initializer),
                    #         thread,
                    #         p1.parent,
                    #         j.getTree(t).constraints(initializer).functionset,
                    #         p1.argposition,
                    #         size)
                    
                    res = self.check_points(p1, p2, state, thread, j, j.getTree(t))
                    if res:
                        break
                
                x = t
                tree = j.getTree(x).clone()
                tree.owner = j
                tree.child = j.getTree(x).child.cloneReplacingNoSubclone(p2, p1)
                tree.child.parent = tree
                tree.child.argposition = 0
                j.addTree(x + 1, tree)
                j.evaluated = False
            
            # Delete instruction
            elif (j.getTreesLength() > j.getMinNumTrees() and
                  (state.random[thread].uniform(0, 1) < self.probInsert + self.probDelete or
                   j.getTreesLength() == j.getMaxNumTrees())):
                
                t = self.getLegalDeleteIndex(j, state, thread)
                j.removeTree(t)
                j.evaluated = False
            
            # Replace instruction (when min == max)
            elif (j.getTreesLength() == j.getMinNumTrees() and
                  j.getTreesLength() == j.getMaxNumTrees()):
                
                t = self.getLegalInsertIndex(j, state, thread)
                
                res = False
                self.nodeselect.reset()
                
                p1 = j.getTree(t).child
                p2 = None
                
                for x in range(self.numTries):
                    # size = GPNodeBuilder.NOSIZEGIVEN
                    # if self.equalSize:
                    #     size = p1.numNodes(GPNode.NODESEARCH_ALL)
                    
                    p2 = self.builder.newRootedTree(
                            state,
                            thread,
                            p1.parent,
                            j.species.primitiveset,
                            p1.argposition,
                            p1.atDepth())
                    
                    # if isinstance(self.builder, LGPMutationGrowBuilder):
                    #     p2 = self.builder.newRootedTree(
                    #         state,
                    #         p1.parentType(initializer),
                    #         thread,
                    #         p1.parent,
                    #         j.getTree(t).constraints(initializer).functionset,
                    #         p1.argposition,
                    #         size,
                    #         p1.atDepth())
                    # else:
                    #     p2 = self.builder.newRootedTree(
                    #         state,
                    #         p1.parentType(initializer),
                    #         thread,
                    #         p1.parent,
                    #         j.getTree(t).constraints(initializer).functionset,
                    #         p1.argposition,
                    #         size)
                    
                    res = self.check_points(p1, p2, state, thread, j, j.getTree(t))
                    if res:
                        break
                
                x = t
                tree = j.getTree(x).clone()
                tree.owner = j
                tree.child = j.getTree(x).child.cloneReplacingNoSubclone(p2, p1)
                tree.child.parent = tree
                tree.child.argposition = 0
                j.addTree(x + 1, tree, update_status=False)
                j.removeTree(x)
                j.evaluated = False
        
        # Post-processing based on mutation type
        if self.mutateFlag == self.EFFMACROMUT3:
            j.removeIneffectiveInstr()
        elif self.mutateFlag in [self.EFFMACROMUT, self.FREEMACROMUT]:
            if self.microMutation is not None:
                j = self.microMutation.produce_individual(subpopulation, j, state, thread)
        
        j.breedingPipe = self

        if j.getEffTreesLength() == 0:
            j.rebuildIndividual(state, thread)

        return j
    
    def getLegalInsertIndex(self, ind:LGPIndividual, state:EvolutionState, thread):
        res = 0
        if self.mutateFlag == self.FREEMACROMUT:
            res = state.random[thread].randint(0, ind.getTreesLength()-1)
        elif self.mutateFlag in [self.EFFMACROMUT, self.EFFMACROMUT2, self.EFFMACROMUT3]:
            res = ind.getConditionIndex(state, thread, lambda x:len(x.effRegisters)>0)
            # res = state.random[thread].randint(0,ind.getTreesLength()-1)
            # for x in range(self.numTries):
            #     if len(ind.getTree(res).effRegisters) > 0:
            #         break
            #     res = state.random[thread].randint(0,ind.getTreesLength()-1)
        else:
            state.output.fatal("illegal mutateFlag in LGP macro mutation")
        return res
    
    def getLegalDeleteIndex(self, ind:LGPIndividual, state:EvolutionState, thread):
        res = 0
        if self.mutateFlag in [self.FREEMACROMUT, self.EFFMACROMUT]:
            res = state.random[thread].randint(0,ind.getTreesLength()-1)
        elif self.mutateFlag in [self.EFFMACROMUT2, self.EFFMACROMUT3]:
            res = ind.getConditionIndex(state, thread, lambda x: x.status)
            # res = state.random[thread].randint(0,ind.getTreesLength()-1)
            # for x in range(self.numTries):
            #     if ind.getTree(res).status:
            #         break
            #     res = state.random[thread].randint(0,ind.getTreesLength()-1)
        else:
            state.output.fatal("illegal mutateFlag in LGP macro mutation")
        return res
    
    def check_points(self, p1:GPNode, p2:GPNode, state:EvolutionState, thread:int, ind:LGPIndividual, treeStr:GPTreeStruct):
        if self.mutateFlag == self.FREEMACROMUT:
            return True
        
        # if mutateFlag != self.FREEMACROMUT, we have to check and maintain the effectiveness
        res = False

        if len(treeStr.effRegisters) > 0 and not p2.getIndex() in treeStr.effRegisters:
            p2.setIndex(state.random[thread].choice(list(treeStr.effRegisters)))
            if p1.printRootedTreeInString() == p2.printRootedTreeInString() \
                and len(treeStr.effRegisters) > 1:
                res = False
            else:
                res = True
        
        return res