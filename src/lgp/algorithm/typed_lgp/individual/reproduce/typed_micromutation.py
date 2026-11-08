
from src.lgp.individual.reproduce import *
from src.lgp.algorithm.typed_lgp.ec import *
from src.lgp.algorithm.typed_lgp.individual.typed_lgpindividual import TypedLGPIndividual, TypedGPTreeStruct
from src.ec import *
from src.ec.util import *
from src.lgp.algorithm.typed_lgp.individual.primitives import *

from typing import get_args

class TypedMicroMutation(LGPMicroMutationPipeline):
    def produce_individual(self, subpopulation, ind, state, thread, no_clone:bool=False)-> TypedLGPIndividual:
        # initializer = state.initializer

        parent:TypedLGPIndividual = ind

        if (self.tree != self.TREE_UNFIXED and 
            (self.tree < 0 or self.tree >= parent.getTreesLength())):
            state.output.fatal("LGP Mutation Pipeline attempted to fix tree.0 to a value which was out of bounds of the array of the individual's trees. " \
            "Check the pipeline's fixed tree values -- they may be negative or greater than the number of trees in an individual")

        # get the function set
        set:GPPrimitiveSet = parent.species.primitiveset  # all trees have the same function set
        
        # get the mutation component
        self.randomGetComponentType(state, thread)

        j = None
        if isinstance(self.sources[0], BreedingPipeline) or no_clone:
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
                target_type = None
                source_type = j.getTree(t).effInputTypes
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
                    target_type = p1.output_type
                
                # size = GPNodeBuilder.NOSIZEGIVEN
                # if self.equalSize:
                #     size = p1.numNodes(GPNode.NODESEARCH_ALL) if p1 else 0
                
                if cnt > 0:
                    if self.componenttype == self.functions:
                        p2:TypedInterface = set.nonterminals[state.random[thread].randint(0,len(set.nonterminals)-1)].lightClone()

                        candidate = p2.agent.getFunctionListWithOutput(target_type, source_type, state, thread)

                        p2.resetNode(state, thread, candidate)
                    
                    elif self.componenttype == self.cons:
                        if state.random[thread].uniform(0,1) < self.builder.probCons:
                            if isinstance(self.builder, Typed_LGPBuilder):
                                # to pick a constant first
                                coherent_constants = self.builder.getTypeCoherentList(set.constants, target_type, state, thread)

                                if len(coherent_constants) > 0:
                                    p2 = state.random[thread].choice(coherent_constants).lightClone()
                                else:
                                    p2 = p1.clone()
                            else:
                                p2 = set.constants[state.random[thread].randint(0,len(set.constants)-1)].lightClone()

                            if isinstance(p2, TypedConstant):
                                candidate = p2.agent.getConstantSetWithOutput(target_type, state, thread)
                                if candidate == None or len(candidate) == 0:
                                    p2 = set.nonconstants[state.random[thread].randint(0,len(set.nonconstants)-1)].lightClone()
                                    p2.resetNode(state, thread)
                                else:
                                    p2.resetNode(state, thread, candidate)
                            elif isinstance(p2, TypedFeature):
                                candidate = p2.agent.getFeaturesWithOutput(target_type)
                                if candidate == None or len(candidate) == 0:
                                    p2 = set.nonconstants[state.random[thread].randint(0,len(set.nonconstants)-1)].lightClone()
                                    p2.resetNode(state, thread)
                                else:
                                    p2.resetNode(state, thread, candidate)
                            elif isinstance(p2, TypedCallableTerminal):
                                call_input_type, call_output_type = get_args(target_type)
                                func_list = p2.agent.getCallableTerminalListWithOutputInput(call_output_type, call_input_type, state, thread)
                                p2.resetNode(state, thread, func_list)
                            else:
                                p2.resetNode(state, thread)
                        else:
                            p2 = set.nonconstants[state.random[thread].randint(0,len(set.nonconstants)-1)].lightClone()
                            p2.resetNode(state, thread)
                    
                    elif self.componenttype == self.writereg:
                        p2 = set.registers[state.random[thread].randint(0,len(set.registers)-1)].lightClone()
                        p2.resetNode(state, thread)

                    elif self.componenttype == self.readreg:
                        if isinstance(self.builder, Typed_LGPBuilder):
                            p2 = self.builder.newRootedTree(
                                state,
                                thread,
                                p1.parent,
                                parent.species.primitiveset,
                                p1.argposition,
                                p1.atDepth(),
                                expect_type=target_type,
                                given_type=source_type)
                        else:
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
                    target_type = p1.output_type
                    if isinstance(self.builder, Typed_LGPBuilder):
                        p2 = self.builder.newRootedTree(
                            state,
                            thread,
                            p1.parent,
                            parent.species.primitiveset,
                            p1.argposition,
                            p1.atDepth(),
                            expect_type=target_type,
                            given_type=source_type)
                    else:
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
                # tree = parent.getTree(x).clone()
                tree.owner = j

                # tree.child = parent.getTree(x).child.cloneReplacingNoSubclone(p2, p1)
                p1.replaceWith(p2)
                
                tree.child.parent = tree
                tree.child.argposition = 0

                # ======
                # args = [] # non-callable arguments
                # tmp = TypedData()
                # tmp.to_vectorize = False
                # for i, child in enumerate(tree.child.children[0].children):
                #     child.eval(state, thread, tmp, j, None)
                #     arg = tmp.value
                #     args.append(arg)

                # n = tree.child.children[0]
                # n.evalfunction(n.eval_function, args, n.child_func)
                # ======

                j.setTree(x, tree, update_status=False)
                if x < j.getTreesLength()-1:
                    j.updateStatus(n=x+1, tar=j.getTree(x).effRegisters)
                else:
                    j.updateStatus()
                j.evaluated = False
            # else:
            #     x = t
            #     tree = j.getTree(x)
            #     tree = parent.getTree(x).clone()
            #     tree.owner = j
            #     tree.child = parent.getTree(x).child.clone()
            #     tree.child.parent = tree
            #     tree.child.argposition = 0
            #     j.setTree(x, tree)
        
        if j.getEffTreesLength(update_status=False) == 0:
            j.rebuildIndividual(state, thread)

        if self.microMutation is not None:
            j = self.microMutation.produce_individual(subpopulation, j, state, thread, no_clone=True)

        j.breedingPipe = self

        return j
    
    def check_points(self, p1:GPNode, p2:GPNode, state:EvolutionState, thread:int, ind:TypedLGPIndividual, treeStr:TypedGPTreeStruct):
        res = False

        if p1.expectedChildren() == p2.expectedChildren() and p1.swapCompatibleWith(p2, state=state):
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
                    elif isinstance(p2, TypedReadRegister):
                        ty = next((t for t in treeStr.effInputRegister.keys() if state.typeagent.tc.type_equal(p2.getOutputType(), t)), None)
                        if ty != None \
                            and p2.getIndex() not in treeStr.effInputRegister[ty]:
                            p2.setIndex(state.random[thread].choice(list(treeStr.effInputRegister[ty])))

                            if str(p1) == str(p2) and len(treeStr.effInputRegister[ty]) > 1:
                                res = False
                            else:
                                res = True
                        else:
                            res = True
                    else:
                        res = True
                else:
                    res = True
                
                # further check for constant value
                # if isinstance(p1, TypedConstant) and isinstance(p2, TypedConstant):
                #     if abs(p1.getValue() - p2.getValue()) > self.cons_step:
                #         if p1.getValue() - p2.getValue() > 0:
                #             p2.setValue(p1.getValue() - 1 - state.random[thread].randint(0,self.cons_step - 1))
                #         else:
                #             p2.setValue(p1.getValue() + 1 + state.random[thread].randint(0,self.cons_step - 1))
                #     res = True
                
                # further check for flow operator
                # if isinstance(p2, FlowOperator) and not ind.canAddFlowOperator():
                #     res = False
                
                for c in range(len(p1.children)):
                    p2.children[c] = p1.children[c].clone()
                    p2.children[c].parent = p2
        
        return res