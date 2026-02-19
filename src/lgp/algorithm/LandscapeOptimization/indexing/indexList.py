from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List
from src.ec.util import *
from src.ec import *
# from src.lgp.algorithm.LandscapeOptimization.subpopulationFLO import SubpopulationFLO
from src.lgp.algorithm.LandscapeOptimization.objectives.objective4FLO import Objective4FLO
from src.lgp.algorithm.LandscapeOptimization.indexing.indexSymbolBuilder import IndexSymbolBuilder
from src.lgp.algorithm.LandscapeOptimization.indexing.index import Index
from src.lgp.algorithm.LandscapeOptimization.indexing.board import Board

T = TypeVar("T")


class IndexList(list[Index], Generic[T], ABC):

    INDEXLIST = "IndexList"
    BUILDER = "builder"
    ITEMPROTOTYPE = "itemprototype"
    NUMOBJECTIVES = "num_objectives"
    OBJECTIVES = "objectives"
    P_COEFFICIENCY = "coef"
    P_BOARDSIZE = "boardsize"

    NUMITERATIONS = "numiterations"
    P_STEP = "step"
    P_MINSTEP = "minstep"
    P_BATCHSIZE = "batchsize"

    # static
    DiffNeighbor = None
    usedItemHistory = None

    def __init__(self):
        super().__init__()

        self.builder:IndexSymbolBuilder = None
        self.prototype:Index = None

        self.numobjectives = 0
        self.objectives:list[Objective4FLO] = None
        self.coefficiency = None
        self.boardsize = None

        self.numiterations = 0
        self.step_rate = 0.0
        self.step = -1
        self.min_step = 0.0
        self.batchsize = 0

    def setup(self, state:EvolutionState, base:Parameter):
        default = Parameter(self.INDEXLIST)

        self.builder = state.parameters.getInstanceForParameter(
            base.push(self.BUILDER),
            default.push(self.BUILDER),
            IndexSymbolBuilder
        )
        self.builder.setup(state, base.push(self.BUILDER))

        self.prototype = state.parameters.getInstanceForParameter(
            base.push(self.ITEMPROTOTYPE),
            default.push(self.ITEMPROTOTYPE),
            Index
        )
        self.prototype.setup(state, base.push(self.ITEMPROTOTYPE))

        self.numobjectives = state.parameters.getInt(
            base.push(self.NUMOBJECTIVES),
            default.push(self.NUMOBJECTIVES)
        )
        if self.numobjectives <= 0:
            state.output.fatal(
                "the number of objectives for fitness landscape optimization must be at least 1",
                base.push(self.NUMOBJECTIVES),
                default.push(self.NUMOBJECTIVES)
            )

        self.objectives = [None] * self.numobjectives
        for obj in range(self.numobjectives):
            self.objectives[obj] = state.parameters.getInstanceForParameter(
                base.push(self.OBJECTIVES).push(str(obj)),
                default.push(self.OBJECTIVES).push(str(obj)),
                Objective4FLO
            )

        self.coefficiency = [0.0] * self.numobjectives
        for obj in range(self.numobjectives):
            self.coefficiency[obj] = state.parameters.getDoubleWithDefault(
                base.push(self.OBJECTIVES).push(str(obj)).push(self.P_COEFFICIENCY),
                default.push(self.OBJECTIVES).push(str(obj)).push(self.P_COEFFICIENCY),
                1.0
            )
            if self.coefficiency[obj] <= 0:
                state.output.fatal(
                    "the coefficiency of objectives for fitness landscape optimization must be larger than 0.0",
                    base.push(self.OBJECTIVES).push(str(obj)).push(self.P_COEFFICIENCY),
                    default.push(self.OBJECTIVES).push(str(obj)).push(self.P_COEFFICIENCY)
                )

        self.boardsize = [0] * self.numobjectives
        for obj in range(self.numobjectives):
            self.boardsize[obj] = state.parameters.getIntWithDefault(
                base.push(self.OBJECTIVES).push(str(obj)).push(self.P_BOARDSIZE),
                default.push(self.OBJECTIVES).push(str(obj)).push(self.P_BOARDSIZE),
                10
            )
            if self.boardsize[obj] <= 0:
                state.output.fatal(
                    "the boardsize of objectives for fitness landscape optimization must be larger than 0.0",
                    base.push(self.OBJECTIVES).push(str(obj)).push(self.P_BOARDSIZE),
                    default.push(self.OBJECTIVES).push(str(obj)).push(self.P_BOARDSIZE)
                )

        self.numiterations = state.parameters.getInt(
            base.push(self.NUMITERATIONS),
            default.push(self.NUMITERATIONS),
            1
        )
        if self.numiterations <= 0:
            state.output.fatal(
                "the number of iterations for fitness landscape optimization must be at least 1",
                base.push(self.NUMITERATIONS),
                default.push(self.NUMITERATIONS)
            )

        self.step_rate = state.parameters.getDoubleWithDefault(
            base.push(self.P_STEP),
            default.push(self.P_STEP),
            0.0
        )
        if self.step_rate <= 0 or self.step_rate > 1:
            state.output.fatal(
                "the step for fitness landscape optimization must be larger than 0.0 and not larger than 1.0",
                base.push(self.P_STEP),
                default.push(self.P_STEP)
            )

        self.min_step = state.parameters.getDoubleWithDefault(
            base.push(self.P_MINSTEP),
            default.push(self.P_MINSTEP),
            1.0
        )
        if self.min_step <= 0:
            state.output.fatal(
                "the minimum step for fitness landscape optimization must be larger than 0.0",
                base.push(self.P_MINSTEP),
                default.push(self.P_MINSTEP)
            )

        self.batchsize = state.parameters.getIntWithDefault(
            base.push(self.P_BATCHSIZE),
            default.push(self.P_BATCHSIZE),
            1
        )
        if self.batchsize <= 0:
            state.output.fatal(
                "the batch size for fitness landscape optimization must be larger than 0.0",
                base.push(self.P_BATCHSIZE),
                default.push(self.P_BATCHSIZE)
            )

        self.initialize(state, 0)

    def initialize(self, state:EvolutionState, thread:int):
        self.basic_initialize(state, thread)

    def basic_initialize(self, state:EvolutionState, thread:int):
        # self.initializeDiffNeighbor(state, thread)
        self.step = max(1.0, round(self.step_rate * len(self)))

    # def getSymbolsByIndex(self, index:int)->List[T]:
    #     for it in self:
    #         if it.index == index:
    #             return it.symbols
    #     return None

    def getSymbolByIndex(self, index:int, state:EvolutionState, thread:int)->T:
        for it in self:
            if it.index == index:
                return it.symbols[state.random[thread].randint(0, len(it.symbols)-1)]

        print(f"The index list cannot find the index {index}")
        exit(1)

    def getIndexBySymbol(self, symbol:T)->int:
        name = str(symbol)
        for it in self:
            # for sym in it.symbols:
            #     if str(sym) == str(symbol):
            #         return it.index
            if name in it.symbol_names:
                return it.index 
        return -1

    @abstractmethod
    def getGenoVector(self, ind:GPIndividual):
        pass

    @abstractmethod
    def optimizeIndex(self, state:EvolutionState, thread:int, subpop:Subpopulation, board:Board):
        pass

    @abstractmethod
    def cloneIndexList(self):
        pass

    @abstractmethod
    def evaluateObjectives(self, list:"IndexList", board:Board):
        pass

    # @abstractmethod
    # def initializeDiffNeighbor(self, state, thread):
    #     pass

    # def getDiffNeighbor(self):
    #     return IndexList.DiffNeighbor

    # def setDiffNeighbor(self, DN):
    #     if DN is None:
    #         return
    #     IndexList.DiffNeighbor = [[0.0] * len(DN) for _ in range(len(DN))]
    #     for i in range(len(DN)):
    #         for j in range(i + 1, len(DN)):
    #             IndexList.DiffNeighbor[i][j] = DN[i][j]
    #             IndexList.DiffNeighbor[j][i] = DN[j][i]

    @staticmethod
    def shuffleIndex(state:EvolutionState, thread:int, list_:List):
        state.random[thread].shuffle(list_)
        # n = len(list_)
        # for _ in range(n):
        #     a = state.random[thread].randint(0, n-1)
        #     b = state.random[thread].randint(0, n-1)
        #     list_[a], list_[b] = list_[b], list_[a]

    def clear_tabu_frequency(self):
        for item in self:
            item.set_tabu_frequency(0)
