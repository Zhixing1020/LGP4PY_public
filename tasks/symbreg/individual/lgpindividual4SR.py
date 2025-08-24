from src.lgp.individual.lgp_individual import LGPIndividual, AtomicInteger

from typing import override

class LGPIndividual4SR(LGPIndividual):

    def __init__(self):
        super().__init__()
        self.dataindex = 0

    @override
    def execute(self, state, thread, input, individual, problem, with_wrap:bool = False):
        return super().execute(state, thread, input, individual, problem, with_wrap)
    
    def postExecution(self, state, thread):
        return
    
    def getDataIndex(self): return self.dataindex

    def setDataIndex(self, index:int):
        self.dataindex = index

    @override
    def makeGraphvizRule(self, outputRegs: list[int]) -> str:
        """
        Convert the LGPIndividual into a Graphviz-compatible string,
        visualizing only effective instructions and connections.
        """
        # collect terminal names
        usedTerminals = [str(self.initVal) for _ in range(self.getNumRegs())]
        
        # set of seen terminals (used for deduplication)
        SRInputs = set()

        # specify nodes (effective instructions)
        nodeSpec = ""
        for i, tree in enumerate(self.getTreelist()):
            if not tree.status:
                continue
            nodeSpec += f'{i}[label="{tree.child.children[0].toGraphvizString()}"];\n'

        # prepare to collect connections
        connection = ""
        notUsed = set(outputRegs)
        cntindex = AtomicInteger(len(self.getTreelist()))

        for i in reversed(range(len(self.getTreelist()))):
            tree = self.getTree(i)
            if not tree.status:
                continue
            connection += self.makeGraphvizInstr(i, SRInputs, usedTerminals, notUsed, cntindex)

        # assemble graph
        result = (
            "digraph g {\n"
            "nodesep=0.2;\n"
            "ranksep=0;\n"
            "node[fixedsize=true,width=1.3,height=0.6,fontsize=\"30\",fontname=\"times-bold\",style=filled, fillcolor=lightgrey];\n"
            "edge[fontsize=\"25.0\",fontname=\"times-bold\"];\n"
            f"{nodeSpec}"
            f"{connection}"
            "}\n"
        )

        return result
