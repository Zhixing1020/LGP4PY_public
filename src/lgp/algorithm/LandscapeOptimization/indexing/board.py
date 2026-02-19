
from src.ec import *
from src.ec.util import *
# from zhixing.cpxInd.algorithm.LandscapeOptimization.objectives.norm2Q import norm2Q
# from zhixing.cpxInd.individual.CpxGPIndividual import CpxGPIndividual
from src.lgp.algorithm.LandscapeOptimization.indexing.boardItem import BoardItem
# from src.lgp.algorithm.LandscapeOptimization.indexing.indexList import IndexList
from src.lgp.algorithm.LandscapeOptimization.indexing.genoVector import GenoVector

class Board(list[BoardItem]):

    BOARD = "board"

    P_MAXSIZE = "maxsize"

    minsize = 3

    def __init__(self):
        super().__init__()
        self.maxsize = 0

        self.genoArray:list[GenoVector] = None
        self.thetaArray = None

        self.maxdiff = 0.0
        self.avgdiff = 0.0
        self.mindiff = 0.0
        self.mincosine = 0.0
        self.avgcosine = 0.0
        self.maxcosine = 0.0

        self.cur_generation_maxdiff = -1
        self.cur_generation_mincosine = -1


    def getDefault(self):
        return Parameter(Board.BOARD)

    def setup(self, state:EvolutionState, base:Parameter):
        def_param = self.getDefault()

        self.maxsize = state.parameters.getInt(
            base.push(Board.P_MAXSIZE),
            def_param.push(Board.P_MAXSIZE),
        )

        if self.maxsize < Board.minsize:
            raise RuntimeError(
                f"borad at least record {Board.minsize} fitness and individuals"
            )

        # self.anchorRate = state.parameters.getDoubleWithDefault(
        #     base.push(Board.P_ANCHORRATE),
        #     def_param.push(Board.P_ANCHORRATE),
        #     0.2
        # )

        # if self.anchorRate <= 0 or self.anchorRate > 1:
        #     raise RuntimeError(
        #         f"the borad got an illegal anchor rate: {self.anchorRate}. It must be (0,1]."
        #     )

    # def getIndividualRank(self, ind:GPIndividual):
    #     res = 0
    #     prog_str = str(ind)
    #     found = False

    #     for item in self:
    #         for in_ind in item:
    #             if str(in_ind) == prog_str:
    #                 found = True
    #                 break
    #         if found:
    #             break
    #         res += 1

    #     return res

    def getMaxDiff(self):
        return self.maxdiff

    def getAvgDiff(self):
        return self.avgdiff

    def getMinDiff(self):
        return self.mindiff

    def getMinCosine(self):
        return self.mincosine

    def getAvgCosine(self):
        return self.avgcosine

    def getMaxCosine(self):
        return self.maxcosine

    def addIndividual(self, ind:GPIndividual):
        if not ind.evaluated:
            print("please evaluate the individual before adding to the board")
            return

        i = 0
        for i in range(len(self)):
            if self[i].add(ind):
                break
        else:
            i = len(self)

        if i >= len(self):
            n = BoardItem(ind)
            super().append(n)

    def add(self, item:BoardItem)->bool:
        for i in range(len(self)):
            if self[i].fitness.equivalentTo(item.fitness):
                self[i].extend(item)
                return True

        super().append(item)
        return True

    def add_at(self, index:int, item:BoardItem):
        for i in range(len(self)):
            if self[i].fitness.equivalentTo(item.fitness):
                self[i].extend(item)
                print(
                    "there has been an item with the equivalent fitness, "
                    "Board.add(...) has put individuals to that place"
                )
                return

        super().insert(index, item)

    def set(self, index:int, item:BoardItem):
        if self[index].fitness.equivalentTo(item.fitness):
            self[index].extend(item)
            return self[index]

        for i in range(len(self)):
            if self[i].fitness.equivalentTo(item.fitness):
                self[i].extend(item)
                print(
                    "there has been an item with the equivalent fitness, "
                    "Board.set(...) has put individuals to that place"
                )
                return item

        old = self[index]
        self[index] = item
        return old

    def boardSize(self)->int:
        res = 0
        for item in self:
            res += len(item)
        return res

    def toGenoArray(self, indexes:'IndexList')->list[GenoVector]:
        if len(self) <= 0:
            print("there is no items on the board but we try to get a Geno Array from it")

        self.genoArray = [None] * self.boardSize()

        i = 0
        for item in self:
            for ind in item:
                self.genoArray[i] = indexes.getGenoVector(ind)
                i += 1

        return self.genoArray

    def toGenoArrayByIndex(self, indexes:'IndexList', boarditem_ind:int)->list[GenoVector]:
        """indexes: the given index system,  boarditem_ind: the index of board item that is to be converted into a geno array"""
        
        if len(self) <= 0:
            print("there is no items on the board but we try to get a Geno Array from it")

        if boarditem_ind >= len(self):
            print("index of Board is out of range")

        item = self[boarditem_ind]
        subgenoArray = [None] * len(item)

        i = 0
        for ind in item:
            subgenoArray[i] = indexes.getGenoVector(ind)
            i += 1

        return subgenoArray

    def getWeightArray(self):
        res = [0.0] * self.boardSize()

        i = 0
        for item in self:
            for _ in range(len(item)):
                res[i] = item.weight
                i += 1

        return res

    def getThetaArray(self, indexes:'IndexList'):
        if self.genoArray is None:
            self.toGenoArray(indexes)

        self.thetaArray = [
            [
                [0 for _ in range(len(indexes))]
                for _ in range(self.genoArray[0].length)
            ]
            for _ in range(len(self.genoArray))
        ]

        for i, Gi in enumerate(self.genoArray):
            for j in range(Gi.length):
                if Gi.G[j] >= 0:
                    pos = 0
                    for ind in indexes:
                        if ind.index == Gi.G[j]:
                            break
                        pos += 1
                    self.thetaArray[i][j][pos] = 1
                else:
                    break

        return self.thetaArray

    def getThetaArrayByIndex(self, indexes:'IndexList', boarditem_ind:int):
        subgenoArray = self.toGenoArrayByIndex(indexes, boarditem_ind)

        subthetaArray = [
            [
                [0 for _ in range(len(indexes))]
                for _ in range(subgenoArray[0].length)
            ]
            for _ in range(len(subgenoArray))
        ]

        for i, Gi in enumerate(subgenoArray):
            for j in range(Gi.length):
                if Gi.G[j] >= 0:
                    pos = 0
                    for ind in indexes:
                        if ind.index == Gi.G[j]:
                            break
                        pos += 1
                    subthetaArray[i][j][pos] = 1
                else:
                    break

        return subthetaArray

    def lightClone(self):
        obj = Board()
        obj.maxsize = self.maxsize

        for item in self:
            obj.add(item.lightClone())

        return obj

    def lightTrimCloneByBest(self, size=None):
        obj = Board()
        obj.maxsize = self.maxsize if size is None else size

        cnt = 0
        for item in self:
            obj.add(item.lightClone())
            cnt += 1
            if cnt >= obj.maxsize:
                break

        return obj

    def weightBoardItem(self):
        """weight the board items by the fitness uniqueness"""
		
		# get the max board item size
        maxboarditem = 0
        for item in self:
            if len(item) > maxboarditem:
                maxboarditem = len(item)

        inverRank = len(self)
        sumw = 0.0

        for item in self:
            item.weight = maxboarditem * inverRank / len(item)
            inverRank -= 1
            sumw += item.weight

        for item in self:
            item.weight /= sumw

    def randomShrinkItem(self, state:EvolutionState, thread:int, batchsize:int):
        for item in self:
            while len(item) > batchsize:
                item.pop(state.random[thread].randint(0, len(item) - 1))
