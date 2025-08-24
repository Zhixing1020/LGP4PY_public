from src.ec.breeding_pipeline import BreedingPipeline
from src.ec.gp_node_selector import GPNodeSelector, GPNode
from src.ec.gp_individual import GPIndividual
from src.ec import *
from src.ec.util import *


class MutationPipeline(BreedingPipeline):

    P_NUM_TRIES = "tries"
    P_MAXDEPTH = "maxdepth"
    P_MAXSIZE = "maxsize"
    P_MUTATION = "mutate"
    P_BUILDER = "build"
    P_EQUALSIZE = "equal"
    INDS_PRODUCED = 1
    NUM_SOURCES = 1
    NO_SIZE_LIMIT = -1

    def __init__(self):
        super().__init__()
        self.nodeselect:GPNodeSelector = None
        self.builder:GPBuilder = None
        self.numTries = 1
        self.maxDepth = 1
        self.maxSize = self.NO_SIZE_LIMIT
        self.equalSize = False
        self.tree = self.TREE_UNFIXED

    def defaultBase(self):
        return Parameter(self.P_MUTATION)

    def numSources(self):
        return self.NUM_SOURCES

    def clone(self):
        c = super().clone()
        # Deep-cloned stuff
        c.nodeselect = self.nodeselect.clone()
        return c

    def setup(self, state, base):
        super().setup(state, base)
        
        def_ = self.defaultBase()
        p = base.push(self.P_NODESELECTOR).push("0")
        d = def_.push(self.P_NODESELECTOR).push("0")

        self.nodeselect = state.parameters.getInstanceForParameter(p, d, GPNodeSelector)
        self.nodeselect.setup(state, p)

        p = base.push(self.P_BUILDER).push("0")
        d = def_.push(self.P_BUILDER).push("0")

        self.builder = state.parameters.getInstanceForParameter(p, d, GPBuilder)
        self.builder.setup(state, p)

        self.numTries = state.parameters.getInt(base.push(self.P_NUM_TRIES), def_.push(self.P_NUM_TRIES))
        if self.numTries < 1:
            state.output.fatal("Mutation Pipeline has an invalid number of tries (it must be >= 1).", 
                             base.push(self.P_NUM_TRIES), def_.push(self.P_NUM_TRIES))

        self.maxDepth = state.parameters.getInt(base.push(self.P_MAXDEPTH), def_.push(self.P_MAXDEPTH))
        if self.maxDepth < 1:
            state.output.fatal(f"The Mutation Pipeline {base} has an invalid maximum depth (it must be >= 1).",
                             base.push(self.P_MAXDEPTH), def_.push(self.P_MAXDEPTH))

        self.maxSize = self.NO_SIZE_LIMIT
        if state.parameters.exists(base.push(self.P_MAXSIZE), def_.push(self.P_MAXSIZE)):
            self.maxSize = state.parameters.getInt(base.push(self.P_MAXSIZE), def_.push(self.P_MAXSIZE))
            if self.maxSize < 1:
                state.output.fatal("Maximum tree size, if defined, must be >= 1")

        self.equalSize = state.parameters.getBoolean(base.push(self.P_EQUALSIZE), def_.push(self.P_EQUALSIZE), False)

        self.tree = self.TREE_UNFIXED
        if state.parameters.exists(base.push(self.P_TREE).push("0"), def_.push(self.P_TREE).push("0")):
            self.tree = state.parameters.getInt(base.push(self.P_TREE).push("0"),
                                             def_.push(self.P_TREE).push("0"), 0)
            if self.tree == -1:
                state.output.fatal("Tree fixed value, if defined, must be >= 0")

    def verifyPoints(self, inner1:GPNode, inner2:GPNode):
        # Check depth
        if inner1.depth() + inner2.atDepth() > self.maxDepth:
            return False

        # Check size
        if self.maxSize != self.NO_SIZE_LIMIT:
            inner1size = inner1.numNodes(GPNode.NODESEARCH_ALL)
            inner2size = inner2.numNodes(GPNode.NODESEARCH_ALL)
            if inner1size > inner2size:
                root2 = inner2.rootParent().child
                root2size = root2.numNodes(GPNode.NODESEARCH_ALL)
                if root2size - inner2size + inner1size > self.maxSize:
                    return False
        return True

    def produce(self, min, max, start, subpopulation, inds, state, thread)->int:
        # Grab individuals from our source
        n = self.sources[0].produce(min, max, start, subpopulation, inds, state, thread)

        # Should we bother?
        # if not state.random[thread].nextBoolean(self.likelihood):
        #     return self.reproduce(n, start, subpopulation, inds, state, thread, False)

        # Mutate them
        for q in range(start, n + start):
            i = inds[q]
            inds[q] = self.produce_individual(subpopulation, i, state, thread)

        return n

    def produce_individual(self, subpopulation, ind, state:EvolutionState, thread)->GPIndividual:
        # initializer = state.initializer

        parent:GPIndividual = ind

        if (self.tree != self.TREE_UNFIXED and 
            (self.tree < 0 or self.tree >= parent.getTreesLength())):
            state.output.fatal("GP Mutation Pipeline attempted to fix tree.0 to a value which was out of bounds of the array of the individual's trees. " \
            "Check the pipeline's fixed tree values -- they may be negative or greater than the number of trees in an individual")

        # Pick random tree
        if self.tree == self.TREE_UNFIXED:
            t = state.random[thread].randint(0, parent.getTreesLength()-1) if parent.getTreesLength() > 1 else 0
        else:
            t = self.tree

        # Validity result
        res = False
        
        # Prepare the nodeselector
        self.nodeselect.reset()
        
        # Pick nodes
        p1 = None
        p2 = None
        
        for x in range(self.numTries):
            # Pick a node in individual
            p1 = self.nodeselect.pickNode(state, subpopulation, thread, parent, parent.getTree(t))
            
            # Generate a tree swap-compatible with p1's position
            # size = GPNodeBuilder.NOSIZEGIVEN if not self.equalSize else p1.numNodes(GPNode.NODESEARCH_ALL)
            size = p1.numNodes(GPNode.NODESEARCH_ALL)
            
            p2 = self.builder.newRootedTree(
                state,
                thread,
                p1.parent,
                parent.species.primitiveset,
                p1.argposition)
            
            # Check for depth and swap-compatibility limits
            res = self.verifyPoints(p2, p1)
            if res:
                break
        
        if isinstance(self.sources[0], BreedingPipeline):
            # It's already a copy, so just modify the tree
            j = parent
            if res:
                p2.parent = p1.parent
                p2.argposition = p1.argposition
                if isinstance(p2.parent, GPNode):
                    p2.parent.children[p2.argposition] = p2
                else:
                    p2.parent.child = p2
                j.evaluated = False
        else:
            # Need to clone the individual
            j = parent.lightClone()
            
            for x in range(j.getTreesLength()):
                tree = j.getTree(x)
                if x == t and res:
                    tree = parent.getTree(x).lightClone()
                    tree.owner = j
                    tree.child = parent.getTree(x).child.cloneReplacingNoSubclone(p2, p1)
                    tree.child.parent = tree
                    tree.child.argposition = 0
                    j.setTree(x, tree)
                    j.evaluated = False
                else:
                    tree = parent.getTree(x).lightClone()
                    tree.owner = j
                    tree.child = parent.getTree(x).child.clone()
                    tree.child.parent = tree
                    tree.child.argposition = 0
                    j.setTree(x, tree)
        j.breedingPipe = self
        return j