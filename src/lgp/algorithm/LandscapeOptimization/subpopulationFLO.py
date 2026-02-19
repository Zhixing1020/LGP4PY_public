from typing import override

from src.ec.subpopulation import Subpopulation
from src.ec import *
from src.ec.util import *
from src.lgp.individual.gp_tree_struct import GPTreeStruct
from src.lgp.algorithm.LandscapeOptimization.indexing.indexList import IndexList
from src.lgp.algorithm.LandscapeOptimization.indexing.board import Board

class SubpopulationFLO(Subpopulation):
    
    # Static string constants
    P_SUBPOPULATION = "subpop_FLO"
    P_INDEXLIST = "indexlist"
    P_BOARD = "board"
    P_LEADINGBOARD = "leadingboard"
    P_LOSINGBOARD = "losingboard"
    P_UPDATEINTERVAL = "updateinterval"
    P_LOGMETINTERVAL = "logmetricinterval"
    
    def __init__(self):
        super().__init__()
        self.IndList:IndexList[GPTreeStruct] = None  # IndexList<GPTreeStruct>
        self.fullBoard:Board = None  # Board
        
        self.updateInterval:int = 1
        self.logMetricInterval:int = 10

    def defaultBase(self):
        return Parameter(self.P_SUBPOPULATION)

    def setup(self, state:EvolutionState, base:Parameter):
        super().setup(state, base)
        
        def_param = self.defaultBase()
        
        # In Python, we instantiate the class specified in the parameters
        self.IndList = state.parameters.getInstanceForParameter(
            base.push(self.P_INDEXLIST), def_param.push(self.P_INDEXLIST), IndexList
        )
        self.IndList.setup(state, base.push(self.P_INDEXLIST))
        
        self.fullBoard = state.parameters.getInstanceForParameter(
            base.push(self.P_BOARD), def_param.push(self.P_BOARD), Board
        )
        self.fullBoard.setup(state, base.push(self.P_BOARD))

        self.updateInterval = state.parameters.getIntWithDefault(
            base.push(self.P_UPDATEINTERVAL), def_param.push(self.P_UPDATEINTERVAL), 1
        )
        if self.updateInterval < 1:
            state.output.fatal("The update interval of SubpopulationFLO is at least 1")
            
        self.logMetricInterval = state.parameters.getIntWithDefault(
            base.push(self.P_LOGMETINTERVAL), def_param.push(self.P_LOGMETINTERVAL), 10
        )
        if self.logMetricInterval < 1:
            state.output.fatal("The log metric interval of SubpopulationFLO is at least 1")

    def updateBoard(self, state:EvolutionState, thread:int):
        
        self.fullBoard.clear()
        
        # Add individuals to the board
        for i in range(len(self.individuals)):
            # CpxGPIndividual cast is implicit in Python
            self.fullBoard.addIndividual(self.individuals[i])
            
        # Sort the board (assumes Board implements __lt__ or is a sortable list)
        self.fullBoard.sort()
        

    def optimizeIndexes(self, state:EvolutionState, thread:int):
        self.IndList.optimizeIndex(state, thread, self, self.fullBoard)

    @override
    def emptyclone(self)->'Subpopulation':
        spop:SubpopulationFLO = super().emptyclone()

        # emptyclone only means clone without individuals
        spop.IndList = self.IndList
        spop.fullBoard = self.fullBoard
        
        spop.updateInterval = self.updateInterval
        spop.logMetricInterval = self.logMetricInterval

        return spop