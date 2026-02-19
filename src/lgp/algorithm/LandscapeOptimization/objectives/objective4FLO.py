from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List
from src.ec import *
# from src.lgp.algorithm.LandscapeOptimization.indexing.indexList import IndexList
from src.lgp.algorithm.LandscapeOptimization.indexing.board import Board
from src.lgp.algorithm.LandscapeOptimization.indexing.genoVector import GenoVector

T = TypeVar("T")

class Objective4FLO(ABC, Generic[T]):
    
    def __init__(self):
        self.indexlist = None
        self.board = None
        self.usedItem: List[bool] = []
        self.privateCoef: float = 1.0  # for normalization, normalize the objective values to [0, 1]

    def evaluate(self, indexlist=None, board=None) -> float:
        """
        Calculates the objective value. If arguments are passed, it updates 
        the instance's indexlist and board before evaluating.
        """
        if indexlist is not None:
            self.indexlist = indexlist
        if board is not None:
            self.board = board
            
        return self._evaluate()

    def gradient(self, indexlist=None, board=None) -> List[float]:
        """
        Calculates the gradient. Updates attributes if provided.
        """
        if indexlist is not None:
            self.indexlist = indexlist
        if board is not None:
            self.board = board
            
        gradient_val = self._gradient()
        
        # transform into a unit vector
        # norm = 0.0
        # for n in range(len(gradient_val)):
        #     norm += gradient_val[n] ** 2
        # norm = math.sqrt(norm)
        # 
        # for n in range(len(gradient_val)):
        #     gradient_val[n] /= norm
        
        return gradient_val

    def setUsedItem(self, indexlist:'IndexList', board:Board):
        self.indexlist = indexlist
        self.board = board
        
        gv_list:list[GenoVector] = []
        
        # Assuming board acts like a 2D array or list of lists
        for b in range(len(board)):
            for bi in range(len(board[b])):
                ind = board[b][bi]
                gv:GenoVector = self.indexlist.getGenoVector(ind)
                gv_list.append(gv)
        
        self.usedItem = [False] * len(indexlist)
        for gv in gv_list:
            for k in range(gv.length):
                if gv.G[k] >= 0:
                    # find the position of item that has this index
                    pos = 0
                    for ni in self.indexlist:
                        if ni.index == gv.G[k]:
                            break
                        pos += 1
                        
                    if pos < len(self.usedItem):
                        self.usedItem[pos] = True
                else:
                    break

    def getUsedItem(self) -> List[bool]:
        return self.usedItem

    @staticmethod
    def getUsedItem_static(indexlist:'IndexList', board:Board) -> List[bool]:
        """
        Static equivalent of getUsedItem. Renamed to avoid shadowing 
        the instance method since Python doesn't support same-name overloading.
        """
        usedItem = [False] * len(indexlist)
        
        for b in range(len(board)):
            for bi in range(len(board[b])):
                ind = board[b][bi]
                gv:GenoVector = indexlist.getGenoVector(ind)
                
                for k in range(gv.length):
                    if gv.G[k] >= 0:
                        # find the position of item that has this index
                        pos = 0
                        while pos < len(indexlist):
                            if indexlist[pos].index == gv.G[k]:
                                break
                            pos += 1
                            
                        if pos < len(usedItem):
                            usedItem[pos] = True
                    else:
                        break
                        
        return usedItem

    @abstractmethod
    def _evaluate(self) -> float:
        """Protected abstract evaluate method mapped from Java."""
        pass

    @abstractmethod
    def _gradient(self) -> List[float]:
        """Protected abstract gradient method mapped from Java."""
        pass

    def setPrivateCoefficiency(self, indexlist:'IndexList', board:Board):
        self.privateCoef = 1.0
        raw_obj = self.getRawObjective(indexlist, board)
        if raw_obj != 0:
            self.privateCoef = 1.0 / raw_obj

    def getRawObjective(self, indexlist:'IndexList', board:Board) -> float:
        return self.evaluate(indexlist, board) / self.privateCoef

    @abstractmethod
    def preprocessing(self, state: EvolutionState, thread: int, indexlist, board, batchsize: int, boardsize: int = None):
        pass

    def updateNewIndexList(self, state: EvolutionState, thread: int, indexlist, board):
        self.setUsedItem(indexlist, board)