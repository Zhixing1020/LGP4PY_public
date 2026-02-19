import numpy as np
from typing import Any, List, Optional

from src.ec import *
from src.lgp.algorithm.LandscapeOptimization.indexing.indexList import IndexList
from src.lgp.algorithm.LandscapeOptimization.indexing.board import Board
from src.lgp.algorithm.LandscapeOptimization.indexing.genoVector import GenoVector

from src.lgp.algorithm.LandscapeOptimization.objectives.objective4FLO import Objective4FLO
from src.lgp.algorithm.LandscapeOptimization.objectives.norm2Q import norm2Q

class L2NORM(Objective4FLO):
    def __init__(self):
        super().__init__()
        self.theta = None
        self.genoArray:list[GenoVector] = None
        self.leadBoard:Board = None

    def _gradient(self) -> List[float]:
        # 1. Prepare numerical data as NumPy arrays
        ga_matrix = np.array([ind.G for ind in self.leadBoard.toGenoArray(self.indexlist)])
        theta_tensor = np.array(self.leadBoard.getThetaArray(self.indexlist))
        weights = np.array(self.leadBoard.getWeightArray())
        
        boardsize = len(ga_matrix)
        index_size = len(self.indexlist)
        
        # 2. Identify Active Indices (Optimization for sparse usedItem)
        active_l = [l for l, used in enumerate(self.usedItem) if used]
        
        if not active_l:
            return [0.0] * index_size
            
        pHpI = np.zeros(index_size)
        pDpH = self.privateCoef / boardsize
        
        # 3. Vectorized Gradient Calculation
        for j in range(boardsize):
            w_j = weights[j]
            g_j = ga_matrix[j]
            t_j = theta_tensor[j] # Shape: (genes, index_size)
            
            # following is equivalent to "getPL2N_Sum"
            # Mask for valid genes (G[k] >= 0)
            valid = (g_j >= 0)
            valid_g = g_j[valid]
            
            # Get only the theta values for active indices 'l'
            # shape: (valid_genes, num_active_l)
            active_theta = t_j[valid][:, active_l]
            
            # Vectorized implementation of getPL2N_Sum:
            # sum_k (2 * G[k] * theta[j][k][l])
            # res shape: (num_active_l,)
            res = np.sum(2 * valid_g[:, np.newaxis] * active_theta, axis=0)
            
            pHpI[active_l] += w_j * res
            
        return (pHpI * pDpH).tolist()

    def _evaluate(self) -> float:
        ga_matrix = np.array([ind.G for ind in self.leadBoard.toGenoArray(self.indexlist)])
        weights = np.array(self.leadBoard.getWeightArray())
        boardsize = len(ga_matrix)
        
        L2N = 0.0
        for j in range(boardsize):
            # Vectorized L2 Norm: sum(G[k]^2) for G[k] >= 0
            valid_g = ga_matrix[j][ga_matrix[j] >= 0]
            L2N += weights[j] * np.sum(valid_g**2)
            
        L2N /= boardsize
        return L2N * self.privateCoef

    def setPrivateCoefficiency(self, indexlist:IndexList, board:Board):
        self.privateCoef = 1.0
        raw_obj = self.getRawObjective(indexlist, board)
        if raw_obj != 0:
            self.privateCoef = 1.0 / raw_obj

    def preprocessing(self, state: EvolutionState, thread: int, indexlist:IndexList, board:Board, batchsize: int, boardsize: Optional[int] = None):
        board.weightBoardItem()
        self.leadBoard = board.lightTrimCloneByBest(boardsize)
        # if boardsize is None:
        #     self.leadBoard = board.lightTrimCloneByBest()
        # else:
        #     self.leadBoard = board.lightTrimCloneByBest(boardsize)
        
        self.leadBoard.randomShrinkItem(state, thread, batchsize)
        self.setUsedItem(indexlist, self.leadBoard)
        
    def updateNewIndexList(self, state: EvolutionState, thread: int, indexlist:IndexList, board:Board):
        self.setUsedItem(indexlist, self.leadBoard)