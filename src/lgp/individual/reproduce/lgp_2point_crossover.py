
from src.ec.breeding_pipeline import BreedingPipeline
from src.ec.gp_node_selector import GPNodeSelector, GPNode
from src.ec.gp_individual import GPIndividual
from src.ec import *
from src.ec.util import *
from .crossover import CrossoverPipeline
from .micro_mutation import LGPMicroMutationPipeline
from .mutation import MutationPipeline
from src.lgp.individual import *
from src.lgp.individual.primitive import *
import builtins


class LGP2PointCrossoverPipeline(CrossoverPipeline):
    # This crossover operator will insert a whole sequence of instructions from the other individual
    P_MAXLENGTH_SEG = "maxseglength"
    P_MAXLENDIFF_SEG = "maxlendiffseg"
    P_MAXDIS_CROSS_POINT = "maxdistancecrosspoint"
    TWOPOINT_CROSSOVER = "2pcross"
    P_MICROMUTBASE = "micro_base"
    P_EFFECTIVE = "effective"
    
    def __init__(self):
        super().__init__()
        self.MaxSegLength = 0
        self.MaxLenDiffSeg = 0
        self.MaxDistanceCrossPoint = 0
        self.microMutation:LGPMicroMutationPipeline = None
        self.eff_flag = False  # whether we need effective crossover
    
    def setup(self, state:EvolutionState, base:Parameter):
        super().setup(state, base)
        
        def_ = LGPDefaults.base.push(self.TWOPOINT_CROSSOVER)
        
        self.MaxSegLength = state.parameters.getInt(base.push(self.P_MAXLENGTH_SEG),def_.push(self.P_MAXLENGTH_SEG))
        if self.MaxSegLength < 1:
            state.output.fatal(
                "LGPCrossover Pipeline has an invalid number of maxseglength (it must be >= 1).",
                base.push(self.P_MAXLENGTH_SEG), def_.push(self.P_MAXLENGTH_SEG))
        
        self.MaxLenDiffSeg = state.parameters.getInt(base.push(self.P_MAXLENDIFF_SEG),def_.push(self.P_MAXLENDIFF_SEG))
        if self.MaxLenDiffSeg < 0:
            state.output.fatal(
                "LGPCrossover Pipeline has an invalid maximum length difference of segments (it must be >= 0).",
                base.push(self.P_MAXLENDIFF_SEG), def_.push(self.P_MAXLENDIFF_SEG))
        
        self.MaxDistanceCrossPoint = state.parameters.getInt(
            base.push(self.P_MAXDIS_CROSS_POINT),
            def_.push(self.P_MAXDIS_CROSS_POINT))
        if self.MaxDistanceCrossPoint < 0:
            state.output.fatal(
                "LGPCrossover Pipeline has an invalid maximum distance of crossover points (it must be >= 0).",
                base.push(self.P_MAXDIS_CROSS_POINT), def_.push(self.P_MAXDIS_CROSS_POINT))
        
        microbase_str = state.parameters.getString(base.push(self.P_MICROMUTBASE), 
                            def_.push(self.P_MICROMUTBASE))
        microbase = Parameter(microbase_str) if microbase_str != None else None
        self.microMutation = None
        if microbase != None and str(microbase) != "null":
            self.microMutation = state.parameters.getInstanceForParameter(
                microbase, def_.push(self.P_MICROMUTBASE), MutationPipeline)
            self.microMutation.setup(state, microbase)
        
        self.eff_flag = state.parameters.getBoolean(
            base.push(self.P_EFFECTIVE),
            def_.push(self.P_EFFECTIVE), False)
    
    def verifyPoints(self, inner1:GPNode, inner2:GPNode):
        # first check to see if inner1 is swap-compatible with inner2
        if not inner1.swapCompatibleWith(inner2):
            return False

        # check depth
        if inner1.depth() + inner2.atDepth() > self.maxDepth:
            return False

        # check size
        if self.maxSize != self.NO_SIZE_LIMIT:
            inner1size = inner1.numNodes(GPNode.NODESEARCH_ALL)
            inner2size = inner2.numNodes(GPNode.NODESEARCH_ALL)
            if inner1size > inner2size:
                root2 = inner2.rootParent().child
                root2size = root2.numNodes(GPNode.NODESEARCH_ALL)
                if root2size - inner2size + inner1size > self.maxSize:
                    return False
        
        # check depth equality for root nodes
        if (inner1.atDepth() != inner2.atDepth() and 
            (inner1.atDepth() == 0 or inner2.atDepth() == 0)):
            return False

        return True
    
    def produce(self, min, max, start, subpopulation, inds:list[GPIndividual], 
                state:EvolutionState, thread)->int:
        # how many individuals should we make?
        n = self.typicalIndsProduced()
        if n < min:
            n = min
        if n > max:
            n = max

        # should we bother?
        # if not state.random[thread].nextBoolean(self.likelihood):
        #     return self.reproduce(n, start, subpopulation, inds, state, thread, True)

        # initializer = state.initializer
        
        q = start

        while q < n + start:
            # grab two individuals from our sources
            if self.sources[0] == self.sources[1]:
                # grab from the same source
                self.sources[0].produce(2, 2, 0, subpopulation, self.parents, state, thread)
            else:
                # grab from different sources
                self.sources[0].produce(1, 1, 0, subpopulation, self.parents, state, thread)
                self.sources[1].produce(1, 1, 1, subpopulation, self.parents, state, thread)
            
            # parents[] now contains our two selected individuals
            parent1 = self.parents[0]
            parent2 = self.parents[1]
            
            nw = self.produce_individual(min, max, q, subpopulation, inds, state, thread, [parent1, parent2])
            
            q += nw

        return n
    
    def produce_individual(self, min, max, start, subpopulation, 
                            inds, state:EvolutionState, thread, parents:list[GPIndividual])->int:
        # how many individuals should we make?
        n = self.typicalIndsProduced()
        if n < min:
            n = min
        if n > max:
            n = max

        # should we bother?
        # if not state.random[thread].nextBoolean(self.likelihood):
        #     return self.reproduce(n, start, subpopulation, inds, state, thread, True)

        # initializer = state.initializer
        
        q = start
        parnt = 0
        while q < n + start:
            # check tree values validity
            if (self.tree1 != self.TREE_UNFIXED and 
                (self.tree1 < 0 or self.tree1 >= parents[0].getTreesLength())):
                state.output.fatal(
                    "LGP Crossover Pipeline attempted to fix tree.0 to a value which was out of bounds of the array of the individual's trees. "
                    "Check the pipeline's fixed tree values -- they may be negative or greater than the number of trees in an individual")
            
            if (self.tree2 != self.TREE_UNFIXED and 
                (self.tree2 < 0 or self.tree2 >= parents[1].getTreesLength())):
                state.output.fatal(
                    "LGP Crossover Pipeline attempted to fix tree.1 to a value which was out of bounds of the array of the individual's trees. "
                    "Check the pipeline's fixed tree values -- they may be negative or greater than the number of trees in an individual")

            t1 = 0
            t2 = 0
            j1:LGPIndividual = None
            j2:LGPIndividual = None
            
            if parents[0].getTreesLength() <= parents[1].getTreesLength():
                j1 = parents[0].lightClone()
                t1 = parnt
                j2 = parents[(parnt + 1) % len(parents)].lightClone()
                t2 = (parnt + 1) % len(parents)
            else:
                j2 = parents[parnt].lightClone()
                t2 = parnt
                j1 = parents[(parnt + 1) % len(parents)].lightClone()
                t1 = (parnt + 1) % len(parents)
            
            # Select crossover points
            begin1 = state.random[thread].randint(0, j1.getTreesLength()-1)
            pickNum1 = state.random[thread].randint(0,
                builtins.min(j1.getTreesLength() - begin1, self.MaxSegLength)-1) + 1
            
            feasibleLowerB = builtins.max(0, begin1 - self.MaxDistanceCrossPoint)
            feasibleUpperB = builtins.min(j2.getTreesLength() - 1, begin1 + self.MaxDistanceCrossPoint)
            
            begin2 = feasibleLowerB + state.random[thread].randint(0, feasibleUpperB - feasibleLowerB)
            pickNum2 = 1 + state.random[thread].randint(0, 
                builtins.min(j2.getTreesLength() - begin2, self.MaxSegLength)-1)
            
            # Adjust segment lengths to meet length difference constraint
            eff = abs(pickNum1 - pickNum2) <= self.MaxLenDiffSeg
            if not eff:
                if j2.getTreesLength() - begin2 > pickNum1 - self.MaxLenDiffSeg:
                    compensate = 1 if self.MaxLenDiffSeg == 0 else 0
                    pickNum2 = builtins.max(1, pickNum1 - self.MaxLenDiffSeg) + state.random[thread].randint(0, 
                        builtins.min(self.MaxSegLength, 
                            builtins.min(j2.getTreesLength() - begin2, pickNum1 + self.MaxLenDiffSeg)) - 
                        builtins.max(0, pickNum1 - self.MaxLenDiffSeg) + compensate-1)
            
            # Ensure tree count constraints are met
            if pickNum1 <= pickNum2:
                if (j2.getTreesLength() - (pickNum2 - pickNum1) < j2.getMinNumTrees() or
                    j1.getTreesLength() + (pickNum2 - pickNum1) > j1.getMaxNumTrees()):
                    if state.random[thread].uniform(0,1) < 0.5:
                        pickNum1 = pickNum2
                    else:
                        pickNum2 = pickNum1
                    if begin1 + pickNum1 > j1.getTreesLength():
                        pickNum1 = pickNum2 = j1.getTreesLength() - begin1
            else:
                if (j2.getTreesLength() + (pickNum1 - pickNum2) > j2.getMaxNumTrees() or
                    j1.getTreesLength() - (pickNum1 - pickNum2) < j1.getMinNumTrees()):
                    if state.random[thread].uniform(0,1) < 0.5:
                        pickNum2 = pickNum1
                    else:
                        pickNum1 = pickNum2
                    if begin2 + pickNum2 > j2.getTreesLength():
                        pickNum1 = pickNum2 = j2.getTreesLength() - begin2
            
            # Perform crossover for j1
            for pick in range(parents[t1].getTreesLength()):
                if pick == begin1:
                    # Remove trees in j1
                    for p in range(pickNum1):
                        j1.removeTree(pick, update_status=False)
                        j1.evaluated = False
                    
                    # Add trees from parent2 to j1
                    for p in range(pickNum2):
                        tree = parents[t2].getTree(begin2 + p).clone()
                        tree.owner = j1
                        tree.child = parents[t2].getTree(begin2 + p).child.clone()
                        tree.child.parent = tree
                        tree.child.argposition = 0
                        j1.addTree(pick + p, tree, update_status=False)
                        j1.evaluated = False
                    
                    j1.updateStatus()
            
            # Apply micro mutation if configured
            if self.microMutation is not None:
                j1 = self.microMutation.produce_individual(subpopulation, j1, state, thread, no_clone=True)

            if self.eff_flag:
                j1.removeIneffectiveInstr()

            # rebuild the individual if it has no effective instructions
            if j1.getEffTreesLength(update_status=False) == 0:
                j1.rebuildIndividual(state, thread)
            
            # Process second child if needed
            if n - (q - start) >= 2 and not self.tossSecondParent:
                for pick in range(parents[t2].getTreesLength()):
                    if pick == begin2:
                        # Remove trees in j2
                        for p in range(pickNum2):
                            j2.removeTree(pick, update_status=False)
                            j2.evaluated = False
                        
                        # Add trees from parent1 to j2
                        for p in range(pickNum1):
                            tree = parents[t1].getTree(begin1 + p).clone()
                            tree.owner = j2
                            tree.child = parents[t1].getTree(begin1 + p).child.clone()
                            tree.child.parent = tree
                            tree.child.argposition = 0
                            j2.addTree(pick + p, tree, update_status=False)
                            j2.evaluated = False
                        
                        j2.updateStatus()
                
                if self.microMutation is not None:
                    j2 = self.microMutation.produce_individual(subpopulation, j2, state, thread, no_clone=True)
                if self.eff_flag:
                    j2.removeIneffectiveInstr()

                # rebuild the individual if it has no effective instructions
                if j2.getEffTreesLength(update_status=False) == 0:
                    j2.rebuildIndividual(state, thread)
            
            # Validate and add children to population
            if (j1.getTreesLength() < j1.getMinNumTrees() or 
                j1.getTreesLength() > j1.getMaxNumTrees()):
                state.output.fatal("illegal tree number in linear cross j1")
            
            j1.breedingPipe = self
            inds[q] = j1
            q += 1
            parnt += 1
            
            if q < n + start and not self.tossSecondParent:
                if (j2.getTreesLength() < j2.getMinNumTrees() or 
                    j2.getTreesLength() > j2.getMaxNumTrees()):
                    state.output.fatal("illegal tree number in linear cross j2")
                j2.breedingPipe = self
                inds[q] = j2
                q += 1
                parnt += 1
        
        return n