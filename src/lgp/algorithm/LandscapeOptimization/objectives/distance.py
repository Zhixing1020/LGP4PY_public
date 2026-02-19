from typing import List, Optional
import numpy as np
from src.ec import *
from src.lgp.algorithm.LandscapeOptimization.indexing.indexList import IndexList
from src.lgp.algorithm.LandscapeOptimization.indexing.board import Board
from src.lgp.algorithm.LandscapeOptimization.indexing.genoVector import GenoVector

from src.lgp.algorithm.LandscapeOptimization.objectives.objective4FLO import Objective4FLO
from src.lgp.algorithm.LandscapeOptimization.objectives.norm2Q import norm2Q

class Distance(Objective4FLO):
    
    def __init__(self):
        super().__init__()
        self.theta = None       # 3D array (list of lists of lists, or numpy array)
        self.genoArray:list[GenoVector] = None   # Array of GenoVector
        self.leadBoard:Board = None   # Board object
        
    # def _gradient(self) -> List[float]:
    #     self.genoArray = self.leadBoard.toGenoArray(self.indexlist)
    #     self.theta = self.leadBoard.getThetaArray(self.indexlist)
    #     weight = self.leadBoard.getWeightArray()
        
    #     boardsize = self.leadBoard.boardSize()
        
    #     # Guard against division by zero
    #     if boardsize == 0:
    #         return [0.0] * len(self.indexlist)
            
    #     pDpH_val = self.privateCoef / (boardsize * boardsize)
    #     index_size = len(self.indexlist)
        
    #     pHpI = [0.0] * index_size
        
    #     # OPTIMIZATION: Loop Reordering & Variable Caching
    #     # We calculate (weight[j] + weight[i]) once per (j, i) pair on the outer loops 
    #     # instead of repeating it inside the l-loop.
    #     for j in range(boardsize):
    #         w_j = weight[j]
    #         for i in range(j + 1, boardsize):
    #             w_sum = w_j + weight[i]
                
    #             for l in range(index_size):
    #                 if self.usedItem[l]:
    #                     pHpI[l] += w_sum * norm2Q.pQpI(
    #                         self.genoArray, self.theta, 
    #                         self.genoArray, self.theta, 
    #                         j, i, l
    #                     )
                        
    #     # Scale all elements by pDpH_val in a single pass
    #     pHpI = [val * pDpH_val for val in pHpI]
        
    #     return pHpI
    
    def _gradient(self) -> list:
        # 1. Prepare data
        ga_matrix = np.array([ind.G for ind in self.leadBoard.toGenoArray(self.indexlist)])
        theta_tensor = np.array(self.leadBoard.getThetaArray(self.indexlist))
        weights = np.array(self.leadBoard.getWeightArray())
        
        boardsize = len(ga_matrix)
        index_size = len(self.indexlist)
        
        # 2. Identify "Active" Indices
        # Get the integer positions where usedItem is True
        # If self.usedItem is a list/array of booleans:
        active_l = [l for l, used in enumerate(self.usedItem) if used]
        
        # If no items are used, return zeros immediately
        if not active_l:
            return [0.0] * index_size
            
        pHpI = np.zeros(index_size)
        
        # 3. Process pairs
        for j in range(boardsize):
            w_j = weights[j]
            g_j = ga_matrix[j]
            t_j = theta_tensor[j]
            
            for i in range(j + 1, boardsize):
                w_sum = w_j + weights[i]
                g_i = ga_matrix[i]
                t_i = theta_tensor[i]
                
                # Filter for valid genes (where not both are -1)
                valid = (g_j >= 0) | (g_i >= 0)
                
                # subQ vector for valid genes
                sub_vals = norm2Q.subQ_vec(g_j[valid], g_i[valid])
                
                # 4. Sparse Calculation: Only compute for active 'l'
                # diff_theta shape: (num_valid_genes, num_active_l)
                diff_theta = t_j[valid][:, active_l] - t_i[valid][:, active_l]
                
                # Sum over genes and multiply by 2 and sub_vals
                # Using np.dot or sum(sub * diff) is much faster than an l-loop
                pair_grad = np.sum(sub_vals[:, np.newaxis] * 2 * diff_theta, axis=0)
                
                # Map the results back to the original index positions
                pHpI[active_l] += w_sum * pair_grad

        pDpH_val = self.privateCoef / (boardsize * boardsize)
        return (pHpI * pDpH_val).tolist()

    def _evaluate(self) -> float:
        self.genoArray = self.leadBoard.toGenoArray(self.indexlist)
        weight = self.leadBoard.getWeightArray()
        
        DI = 0.0
        boardsize = self.leadBoard.boardSize()
        
        if boardsize == 0:
            return 0.0
            
        BB = float(boardsize * boardsize)
        
        # OPTIMIZATION: Local variable caching
        for j in range(boardsize):
            w_j = weight[j]
            geno_j = self.genoArray[j]
            for i in range(j + 1, boardsize):
                DI += (w_j + weight[i]) * norm2Q.Q(geno_j, self.genoArray[i])
                
        DI /= BB
        
        return DI * self.privateCoef
    
    # def _evaluate(self) -> float:
    #     ga_matrix = np.array([ind.G for ind in self.leadBoard.toGenoArray(self.indexlist)])
    #     weights = np.array(self.leadBoard.getWeightArray())
    #     boardsize = len(ga_matrix)
        
    #     DI = 0.0
    #     for j in range(boardsize):
    #         w_j = weights[j]
    #         g_j = ga_matrix[j]
    #         for i in range(j + 1, boardsize):
    #             # Using the vectorized Q calculation
    #             valid = ~((g_j < 0) & (ga_matrix[i] < 0))
    #             sub = norm2Q.subQ_vec(g_j[valid], ga_matrix[i][valid])
    #             DI += (w_j + weights[i]) * np.sum(sub**2)
        
    #     return (DI / (boardsize * boardsize)) * self.privateCoef

    def setPrivateCoefficiency(self, indexlist:IndexList, board:Board):
        self.privateCoef = 1.0
        raw_obj = self.getRawObjective(indexlist, board)
        if raw_obj != 0:
            self.privateCoef = 1.0 / raw_obj

    def preprocessing(self, state: EvolutionState, thread: int, indexlist:IndexList, board:Board, batchsize: int, boardsize: Optional[int] = None):
        """
        Handles both overloaded Java preprocessing methods.
        If boardsize is provided, it behaves like the 6-parameter Java method.
        """
        board.weightBoardItem()
        
        self.leadBoard = board.lightTrimCloneByBest(boardsize)
        # if boardsize is None:
        #     self.leadBoard = board.lightTrimCloneByBest()
        # else:
        #     self.leadBoard = board.lightTrimCloneByBest(boardsize)
            
        self.leadBoard.randomShrinkItem(state, thread, batchsize)
        # self.leadBoard.trim2MaxsizeByBest()
        
        self.setUsedItem(indexlist, self.leadBoard)
        
    def updateNewIndexList(self, state: EvolutionState, thread: int, indexlist:IndexList, board:Board):
        self.setUsedItem(indexlist, self.leadBoard)