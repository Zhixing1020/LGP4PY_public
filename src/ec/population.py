

from src.ec.evolution_state import EvolutionState
from src.ec.util.parameter import Parameter
# from typing import List
from src.ec.subpopulation import Subpopulation

class Population:
    __slots__ = ("subpops",)
    P_SIZE = "subpops"
    P_SUBPOP = "subpop"

    def __init__(self):
        self.subpops:list[Subpopulation] = [None]

    def clear(self):
        for subp in self.subpops:
            subp.clear()

    def setup(self, state:EvolutionState, base:Parameter):
        p = base.push(self.P_SIZE)
        size = state.parameters.getInt(p,None)
        if size == 0:
            state.output.fatal("Population size must be >0.\n")

        from src.ec.subpopulation import Subpopulation

        self.subpops = [None] * size

        for x in range(size):
            pp = base.push(self.P_SUBPOP).push(""+str(x))

            if not state.parameters.exists(str(pp)):
                state.output.fatal(f"The parameter {pp} is not defined\n")

            self.subpops[x] = state.parameters.getInstanceForParameter(pp, None, Subpopulation)
            self.subpops[x].setup(state, pp)

    def populate(self, state:EvolutionState, thread: int):
        for subpop in self.subpops:
            subpop.populate(state, thread)

    def emptyclone(self)->'Population':
        # from src.ec.subpopulation import Subpopulation
        pop = self.__class__()
        pop.subpops = [None] * len(self.subpops)

        for i, p in enumerate(self.subpops):
            pop.subpops[i] = p.emptyclone()
        
        return pop