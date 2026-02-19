from src.ec.breeding_pipeline import BreedingPipeline
from src.ec.gp_node_selector import GPNodeSelector, GPNode
from src.ec.gp_individual import GPIndividual
from src.ec import *
from src.ec.util import *

class CrossoverPipeline(BreedingPipeline):
    P_NUM_TRIES = "tries"
    P_MAXDEPTH = "maxdepth"
    P_MAXSIZE = "maxsize"
    P_CROSSOVER = "xover"
    P_TOSS = "toss"
    INDS_PRODUCED = 2
    NUM_SOURCES = 2
    

    def __init__(self):
        super().__init__()
        self.nodeselect1:GPNodeSelector = None
        self.nodeselect2:GPNodeSelector = None
        self.tree1 = self.TREE_UNFIXED
        self.tree2 = self.TREE_UNFIXED
        self.numTries = 1
        self.maxDepth = 1
        self.maxSize = self.NO_SIZE_LIMIT
        self.tossSecondParent = False
        self.parents = [None, None]

    def defaultBase(self):
        return Parameter(self.P_CROSSOVER)

    def numSources(self):
        return self.NUM_SOURCES

    def typicalIndsProduced(self):
        return self.minChildProduction() if self.tossSecondParent else self.minChildProduction()*2
    
    def setup(self, state, base):
        super().setup(state, base)

        def_base = self.defaultBase()

        p0 = base.push(self.P_NODESELECTOR).push("0")
        d0 = def_base.push(self.P_NODESELECTOR).push("0")
        self.nodeselect1 = state.parameters.getInstanceForParameter(p0, d0, GPNodeSelector)
        self.nodeselect1.setup(state, p0)

        p1 = base.push(self.P_NODESELECTOR).push("1")
        d1 = def_base.push(self.P_NODESELECTOR).push("1")
        if state.parameters.exists(p1, d1) and state.parameters.getString(p1, d1) == self.V_SAME:
            self.nodeselect2 = self.nodeselect1.clone()
        else:
            self.nodeselect2 = state.parameters.getInstanceForParameter(p1, d1, GPNodeSelector)
            self.nodeselect2.setup(state, p1)

        self.numTries = state.parameters.getInt(base.push(self.P_NUM_TRIES), def_base.push(self.P_NUM_TRIES))
        if self.numTries <= 0:
            state.output.fatal("Invalid number of crossover tries.", base.push(self.P_NUM_TRIES), def_base.push(self.P_NUM_TRIES))

        self.maxDepth = state.parameters.getInt(base.push(self.P_MAXDEPTH), def_base.push(self.P_MAXDEPTH))
        if self.maxDepth <= 0:
            state.output.fatal("Invalid crossover max depth.", base.push(self.P_MAXDEPTH), def_base.push(self.P_MAXDEPTH))

        if state.parameters.exists(base.push(self.P_MAXSIZE), def_base.push(self.P_MAXSIZE)):
            self.maxSize = state.parameters.getInt(base.push(self.P_MAXSIZE), def_base.push(self.P_MAXSIZE))
            if self.maxSize < 1:
                state.output.fatal("Invalid crossover max size.", base.push(self.P_MAXSIZE), def_base.push(self.P_MAXSIZE))

        if state.parameters.exists(base.push(self.P_TREE).push("0"), def_base.push(self.P_TREE).push("0")):
            self.tree1 = state.parameters.getInt(base.push(self.P_TREE).push("0"), def_base.push(self.P_TREE).push("0"))

        if state.parameters.exists(base.push(self.P_TREE).push("1"), def_base.push(self.P_TREE).push("1")):
            self.tree2 = state.parameters.getInt(base.push(self.P_TREE).push("1"), def_base.push(self.P_TREE).push("1"))

        self.tossSecondParent = state.parameters.getBoolean(base.push(self.P_TOSS), def_base.push(self.P_TOSS), False)

    def clone(self):
        c = super().clone()
        c.nodeselect1 = self.nodeselect1.clone()
        c.nodeselect2 = self.nodeselect2.clone()
        c.parents = [None, None]   # self.parents.clone()
        return c

    def verifyPoints(self, inner1:GPNode, inner2:GPNode):
        if not inner1.swapCompatibleWith(inner2):
            return False

        if inner1.depth() + inner2.atDepth() > self.maxDepth:
            return False

        if self.maxSize != self.NO_SIZE_LIMIT:
            inner1size = inner1.numNodes(GPNode.NODESEARCH_ALL)
            inner2size = inner2.numNodes(GPNode.NODESEARCH_ALL)
            if inner1size > inner2size:
                root2 = inner2.rootParent().child
                root2size = root2.numNodes(GPNode.NODESEARCH_ALL)
                if root2size - inner2size + inner1size > self.maxSize:
                    return False

        return True

    def produce(self, min, max, start, subpopulation, inds:list[GPIndividual], state:EvolutionState, thread)->int:
        n = self.typicalIndsProduced()
        if n < min:
            n = min
        if n > max:
            n = max

        # if not state.random[thread].uniform(0, 1) < self.likelihood:
        #     return self.reproduce(n, start, subpopulation, inds, state, thread, True)

        q = start
        while q < n + start:
            if self.sources[0] == self.sources[1]:
                self.sources[0].produce(2, 2, 0, subpopulation, self.parents, state, thread)
            else:
                self.sources[0].produce(1, 1, 0, subpopulation, self.parents, state, thread)
                self.sources[1].produce(1, 1, 1, subpopulation, self.parents, state, thread)

            parent1 = self.parents[0]
            parent2 = self.parents[1]

            nw = self.produce_individual(min, max, q, subpopulation, inds, state, thread, [parent1, parent2])

            q += nw

        return n

    def produce_individual(self, min, max, start, subpopulation, 
                            inds, state:EvolutionState, thread, parents:list[GPIndividual])->int:
        # How many individuals should we make?
        n = self.typicalIndsProduced()
        if n < min:
            n = min
        if n > max:
            n = max

        # Should we bother?
        # if not state.random[thread].nextBoolean(self.likelihood):
        #     return self.reproduce(n, start, subpopulation, inds, state, thread, True)

        # initializer = state.initializer
        
        q = start
        parnt = 0
        while q < n + start:
            # Check tree values validity
            if (self.tree1 != self.TREE_UNFIXED and 
                (self.tree1 < 0 or self.tree1 >= parents[0].getTreesLength())):
                state.output.fatal("GP Crossover Pipeline attempted to fix tree.0 to a value " \
                "which was out of bounds of the array of the individual's trees. Check the pipeline's fixed tree values" \
                " -- they may be negative or greater than the number of trees in an individual")
            
            if (self.tree2 != self.TREE_UNFIXED and 
                (self.tree2 < 0 or self.tree2 >= parents[1].getTreesLength())):
                state.output.fatal("GP Crossover Pipeline attempted to fix tree.1 to a value " \
                "which was out of bounds of the array of the individual's trees. Check the pipeline's fixed tree values" \
                " -- they may be negative or greater than the number of trees in an individual")


            # Make sure the constraints are okay
            if parents[0].species == parents[1].species:
                state.output.fatal("GP Crossover Pipeline's two parents have different GPSpecies")
            t1 = 0
            t2 = 0
            if self.tree1 == self.TREE_UNFIXED or self.tree2 == self.TREE_UNFIXED:
                if self.tree1 == self.TREE_UNFIXED:
                    t1 = (state.random[thread].randint(0, parents[0].getTreesLength()-1) 
                            if parents[0].getTreesLength() > 1 else 0)
                else:
                    t1 = self.tree1

                if self.tree2 == self.TREE_UNFIXED:
                    t2 = (state.random[thread].randint(0, parents[1].getTreesLength()-1) 
                            if parents[1].getTreesLength() > 1 else 0)
                else:
                    t2 = self.tree2
            else:
                t1 = self.tree1
                t2 = self.tree2
                

            # Validity results
            res1 = False
            res2 = False
            
            # Prepare the nodeselectors
            self.nodeselect1.reset()
            self.nodeselect2.reset()
            
            # Pick some nodes
            p1 = None
            p2 = None
            
            for x in range(self.numTries):
                # Pick a node in individual 1
                p1 = self.nodeselect1.pickNode(state, subpopulation, thread, parents[0], parents[0].getTree(t1))
                
                # Pick a node in individual 2
                p2 = self.nodeselect2.pickNode(state, subpopulation, thread, parents[1], parents[1].getTree(t2))
                
                # Check for depth and swap-compatibility limits
                res1 = self.verifyPoints(p2, p1)  # p2 can fill p1's spot
                if n - (q - start) < 2 or self.tossSecondParent:
                    res2 = True
                else:
                    res2 = self.verifyPoints(p1, p2)  # p1 can fill p2's spot
                
                # Did we get something that had both nodes verified?
                if res1 and res2:
                    break

            # Create new individuals based on the old ones
            j1 = parents[0].lightClone()
            j2 = None
            if n - (q - start) >= 2 and not self.tossSecondParent:
                j2 = parents[1].lightClone()
            
            # Process first child
            for x in range(j1.getTreesLength()):
                tree = j1.getTree(x)
                if x == t1 and res1:  # We've got a tree with a valid cross position
                    tree = parents[0].getTree(x).lightClone()
                    tree.owner = j1
                    tree.child = parents[0].getTree(x).child.cloneReplacing(p2, p1)
                    tree.child.parent = tree
                    tree.child.argposition = 0
                    j1.setTree(x, tree)
                    j1.evaluated = False
                else:
                    tree = parents[0].getTree(x).lightClone()
                    tree.owner = j1
                    tree.child = parents[0].getTree(x).child.clone()
                    tree.child.parent = tree
                    tree.child.argposition = 0
                    j1.setTree(x, tree)
            
            # Process second child if needed
            if n - (q - start) >= 2 and not self.tossSecondParent and j2 is not None:
                for x in range(j2.getTreesLength()):
                    tree = j2.getTree(x)
                    if x == t2 and res2:  # We've got a tree with a valid cross position
                        tree = parents[1].getTree(x).lightClone()
                        tree.owner = j2
                        tree.child = parents[1].getTree(x).child.cloneReplacing(p1, p2)
                        tree.child.parent = tree
                        tree.child.argposition = 0
                        j2.setTree(x, tree)
                        j2.evaluated = False
                    else:
                        tree = parents[1].getTree(x).lightClone()
                        tree.owner = j2
                        tree.child = parents[1].getTree(x).child.clone()
                        tree.child.parent = tree
                        tree.child.argposition = 0
                        j2.setTree(x, tree)
            
            # Add the individuals to the population
            j1.breedingPipe = self
            inds[q] = j1
            q += 1
            if q < n + start and not self.tossSecondParent and j2 is not None:
                j2.breedingPipe = self
                inds[q] = j2
                q += 1

        return n