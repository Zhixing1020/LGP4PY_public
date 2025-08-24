
from src.ec.util import *
from src.ec.evolution_state import EvolutionState

class Statistics:
    P_NUMCHILDREN = "num-children"
    P_CHILD = "child"
    P_SILENT = "silent"
    P_MUZZLE = "muzzle"  # deprecated
    P_SILENT_PRINT = "silent.print"
    P_SILENT_FILE = "silent.file"
    
    def __init__(self):
        self.children:list[Statistics] = []
        self.silentFile = False
        self.silentPrint = False
    
    def setup(self, state:EvolutionState, base:Parameter):
        t = state.parameters.getIntWithDefault(base.push(self.P_NUMCHILDREN), None, 0)
        if t < 0:
            state.output.fatal(
                "A Statistics object cannot have negative number of children",
                base.push(self.P_NUMCHILDREN)
            )
        
        self.silentFile = self.silentPrint = state.parameters.getBoolean(
            base.push(self.P_SILENT), None, False
        )
        
        # Second assignment is intentional (as per Java comment)
        self.silentFile = state.parameters.getBoolean(
            base.push(self.P_SILENT_FILE), None, self.silentFile
        )
        self.silentPrint = state.parameters.getBoolean(
            base.push(self.P_SILENT_PRINT), None, self.silentPrint
        )
        
        if state.parameters.exists(f"{base}.{self.P_MUZZLE}", None):
            state.output.warning(
                f"{base}.{self.P_MUZZLE} has been deprecated. We suggest you use "
                f"{base}.{self.P_SILENT} or similar newer options."
            )
        self.silentFile = self.silentFile or state.parameters.getBoolean(
            base.push(self.P_MUZZLE), None, False
        )

        # Load the children
        self.children = [None] * t
        for x in range(t):
            p:Parameter = base.push(self.P_CHILD).push(x)
            self.children[x] = state.parameters.getInstanceForParameter(
                p, None, Statistics
            )
            self.children[x].setup(state, p)
    
    def preInitializationStatistics(self, state):
        for child in self.children:
            child.preInitializationStatistics(state)
    
    def postInitializationStatistics(self, state):
        for child in self.children:
            child.postInitializationStatistics(state)
    
    def preEvaluationStatistics(self, state):
        for child in self.children:
            child.preEvaluationStatistics(state)
    
    def postEvaluationStatistics(self, state):
        for child in self.children:
            child.postEvaluationStatistics(state)
    
    def preBreedingStatistics(self, state):
        for child in self.children:
            child.preBreedingStatistics(state)
    
    def postBreedingStatistics(self, state):
        for child in self.children:
            child.postBreedingStatistics(state)
    
    def prePostBreedingExchangeStatistics(self, state):
        for child in self.children:
            child.prePostBreedingExchangeStatistics(state)
    
    def postPostBreedingExchangeStatistics(self, state):
        for child in self.children:
            child.postPostBreedingExchangeStatistics(state)
    
    def finalStatistics(self, state, result):
        for child in self.children:
            child.finalStatistics(state, result)