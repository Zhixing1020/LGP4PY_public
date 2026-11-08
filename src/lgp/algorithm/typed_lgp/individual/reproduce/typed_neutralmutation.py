from src.lgp.individual.reproduce import *
from src.lgp.algorithm.typed_lgp.ec import *
from src.lgp.algorithm.typed_lgp.individual.typed_lgpindividual import TypedLGPIndividual, TypedGPTreeStruct
from src.ec import *
from src.ec.util import *
from src.lgp.algorithm.typed_lgp.individual.primitives import *

class TypedNeutralMutation(LGPNeutralMutationPipeline):
    def produce_individual(self, subpopulation, ind, state:TypedEvolutionState, thread:int, no_clone:bool=False)->TypedLGPIndividual:
        # initializer = state.initializer
            
        i:TypedLGPIndividual = ind

        if (self.tree != self.TREE_UNFIXED and 
            (self.tree < 0 or self.tree >= i.getTreesLength())):
            state.output.fatal("LGP Mutation Pipeline attempted to fix tree.0 to a value which was out of bounds of the array of the individual's trees. Check the pipeline's fixed tree values -- they may be negative or greater than the number of trees in an individual")

        j:TypedLGPIndividual = None
        if isinstance(self.sources[0], BreedingPipeline) or no_clone:
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
        
        # j.updateStatus()
        
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
                    tmp_inds:list = [None]
                    n = self.sources[0].produce(1, 1, 0, subpopulation, tmp_inds, state, thread)
                    source_ind = tmp_inds[0]
                    source_t = state.random[thread].randint(0, source_ind.getTreesLength()-1)
                    p2 = source_ind.getTree(source_t).clone().child
                    if source_ind.getTree(source_t).status and self.check_points(p1, p2, state, thread, j, j.getTree(t)):
                        break
                
                # for x in range(self.numTries):
                #     # size = GPNodeBuilder.NOSIZEGIVEN
                #     # if self.equalSize:
                #     #     size = p1.numNodes(GPNode.NODESEARCH_ALL)
                    
                #     if isinstance(self.builder, Typed_LGPBuilder):
                #         # randomly choose one possible type as target type
                #         target_type = state.random[thread].choice(list(j.register_prototype.agent.getTypeSet()))
                #         source_type = j.getTree(t).effInputTypes
                #         p2 = self.builder.newRootedTree(
                #                 state,
                #                 thread,
                #                 p1.parent,
                #                 j.species.primitiveset,
                #                 p1.argposition,
                #                 p1.atDepth(),
                #                 expect_type=target_type,
                #                 given_type=source_type)
                #     else:    
                #         p2 = self.builder.newRootedTree(
                #                 state,
                #                 thread,
                #                 p1.parent,
                #                 j.species.primitiveset,
                #                 p1.argposition,
                #                 p1.atDepth())
                    
                #     # if isinstance(self.builder, LGPMutationGrowBuilder):
                #     #     p2 = self.builder.newRootedTree(
                #     #         state,
                #     #         p1.parentType(initializer),
                #     #         thread,
                #     #         p1.parent,
                #     #         j.getTree(t).constraints(initializer).functionset,
                #     #         p1.argposition,
                #     #         size,
                #     #         p1.atDepth())
                #     # else:
                #     #     p2 = self.builder.newRootedTree(
                #     #         state,
                #     #         p1.parentType(initializer),
                #     #         thread,
                #     #         p1.parent,
                #     #         j.getTree(t).constraints(initializer).functionset,
                #     #         p1.argposition,
                #     #         size)
                    
                #     res = self.check_points(p1, p2, state, thread, j, j.getTree(t))
                #     if res:
                #         break
                
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
                  j.getEffTreesLength(update_status=False) < j.getTreesLength() and
                  (state.random[thread].uniform(0, 1) < self.probInsert + self.probDelete or
                   j.getTreesLength() == j.getMaxNumTrees())):
                
                t = self.getLegalDeleteIndex(j, state, thread)
                if j.getTree(t).status:
                    continue  # skip effective instruction
                j.removeTree(t)
                j.evaluated = False
            
            # Replace instruction (when min == max)
            elif (j.getTreesLength() > j.getEffTreesLength(update_status=False)):
                
                t = self.getLegalDeleteIndex(j, state, thread)
                
                res = False
                self.nodeselect.reset()
                
                p1 = j.getTree(t).child
                p2 = None

                for x in range(self.numTries):
                    tmp_inds:list = [None]
                    n = self.sources[0].produce(1, 1, 0, subpopulation, tmp_inds, state, thread)
                    source_ind = tmp_inds[0]
                    source_t = state.random[thread].randint(0, source_ind.getTreesLength()-1)
                    p2 = source_ind.getTree(source_t).clone().child
                    if source_ind.getTree(source_t).status and self.check_points(p1, p2, state, thread, j, j.getTree(t)):
                        break
                
                # for x in range(self.numTries):
                #     # size = GPNodeBuilder.NOSIZEGIVEN
                #     # if self.equalSize:
                #     #     size = p1.numNodes(GPNode.NODESEARCH_ALL)
                    
                #     if isinstance(self.builder, Typed_LGPBuilder):
                #         target_type = state.random[thread].choice(list(j.register_prototype.agent.getTypeSet()))
                #         source_type = j.getTree(t).effInputTypes
                #         p2 = self.builder.newRootedTree(
                #                 state,
                #                 thread,
                #                 p1.parent,
                #                 j.species.primitiveset,
                #                 p1.argposition,
                #                 p1.atDepth(),
                #                 expect_type=target_type,
                #                 given_type=source_type)
                #     else:    
                #         p2 = self.builder.newRootedTree(
                #                 state,
                #                 thread,
                #                 p1.parent,
                #                 j.species.primitiveset,
                #                 p1.argposition,
                #                 p1.atDepth())
                    
                #     # if isinstance(self.builder, LGPMutationGrowBuilder):
                #     #     p2 = self.builder.newRootedTree(
                #     #         state,
                #     #         p1.parentType(initializer),
                #     #         thread,
                #     #         p1.parent,
                #     #         j.getTree(t).constraints(initializer).functionset,
                #     #         p1.argposition,
                #     #         size,
                #     #         p1.atDepth())
                #     # else:
                #     #     p2 = self.builder.newRootedTree(
                #     #         state,
                #     #         p1.parentType(initializer),
                #     #         thread,
                #     #         p1.parent,
                #     #         j.getTree(t).constraints(initializer).functionset,
                #     #         p1.argposition,
                #     #         size)
                    
                #     res = self.check_points(p1, p2, state, thread, j, j.getTree(t))
                #     if res:
                #         break
                
                x = t
                tree = j.getTree(x).clone()
                tree.owner = j
                tree.child = j.getTree(x).child.cloneReplacingNoSubclone(p2, p1)
                tree.child.parent = tree
                tree.child.argposition = 0
                j.addTree(x + 1, tree, update_status=False)
                j.removeTree(x)
                j.evaluated = False
        
        # Post-processing 
        if self.microMutation is not None:
            j = self.microMutation.produce_individual(subpopulation, j, state, thread, no_clone=True)
       
        j.breedingPipe = self

        return j
    
    # def getLegalDeleteIndex(self, ind:TypedLGPIndividual, state:EvolutionState, thread):
    #     """get the ineffective instructions, to be removed"""
    #     res = ind.getConditionIndex(state, thread, lambda x: not x.status and not x.forwardStatus)
    #     if ind.getTree(res).status:
    #         res = ind.getConditionIndex(state, thread, lambda x: not x.status)
    #     return res
    
    def check_points(self, p1:GPNode, p2:GPNode, state:EvolutionState, thread:int, ind:TypedLGPIndividual, treeStr:TypedGPTreeStruct):
        """
        make sure the newly generated instructions are intron while the read register are effective -- 
        accepting effective source types
        """
        res = super().check_points(p1, p2, state, thread, ind, treeStr)


        # check if the type of readRegister is consistent with their manipulated types
        treeStr.setReadRegisterOnTypes(p2, state, thread)        

        if p1.printRootedTreeInString() != p2.printRootedTreeInString():
            res = True
        else:
            res = False
        
        return res