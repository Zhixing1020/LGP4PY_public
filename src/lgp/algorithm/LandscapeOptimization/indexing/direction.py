from typing import List
import math

from src.ec import *
from src.lgp.algorithm.LandscapeOptimization.indexing.genoVector import GenoVector
# from zhixing.cpxInd.algorithm.LandscapeOptimization.objectives.norm2Q import norm2Q


class Direction(list):
    """
    Direction is not a specific direction from point A to point B.
    Instead, it is a template, indicating:
    - the number of elements that are to be changed,
    - the program size change,
    - and the discrepancy.
    """
    """Adding a Direction to a specific point could generate different specific movement since we need to 
    instantiate the direction (randomly completing or removing)."""

    def __init__(self):
        super().__init__()
        self.sizeDirection = 0  # >0: increase ``num'' of symbols, <0: remove ``num'' of symbols.

    def clone(self):
        obj = Direction()
        for d in self:
            obj.append(d)
        obj.sizeDirection = self.sizeDirection
        return obj

    def clear(self):
        super().clear()

    # def setDirection(self, pgDestination:GenoVector, pgSource:GenoVector, state:EvolutionState, thread:int):
    #     if pgDestination.length != pgSource.length:
    #         raise RuntimeError(
    #             f"we got GenoVectors with inconsistent length in Direction: "
    #             f"{pgDestination.length} and {pgSource.length}"
    #         )

    #     # reset
    #     self.clear()

    #     # transform GenoVector
    #     pgDes_list = []
    #     pgSrc_list = []
    #     for i in range(pgDestination.length):
    #         if pgDestination.G[i] >= 0:
    #             pgDes_list.append(pgDestination.G[i])
    #         if pgSource.G[i] >= 0:
    #             pgSrc_list.append(pgSource.G[i])

    #     # align the length
    #     len_des = len(pgDes_list)
    #     len_src = len(pgSrc_list)
    #     self.sizeDirection = len_des - len_src

    #     # align pgDes_list and pgSrc_list
    #     num_element = len_src
    #     while len(pgDes_list) > num_element:
    #         index = state.random[thread].randint(0, len(pgDes_list) - 1)
    #         pgDes_list.pop(index)

    #     while len(pgDes_list) < num_element:
    #         insert = state.random[thread].randint(0, len(pgDes_list))
    #         pgDes_list.insert(insert, -1)

    #     # difference
    #     allZero = True
    #     for k in range(num_element):
    #         if pgDes_list[k] >= 0:
    #             diff = pgDes_list[k] - pgSrc_list[k]
    #             self.append(float(diff))
    #             if diff != 0:
    #                 allZero = False
    #         else:
    #             self.append(0.0)

    #     if allZero and len(self) > 0:
    #         index = state.random[thread].randint(0, len(self) - 1)
    #         self[index] = (-1.0) ** state.random[thread].randint(0, 1)

    #     Direction.normalization(self)

    # def setDirection_symrng(self, pgDestination, pgSource, state, thread, sym_rng):
    #     if pgDestination.length != pgSource.length:
    #         raise RuntimeError(
    #             f"we got GenoVectors with inconsistent length in Direction: "
    #             f"{pgDestination.length} and {pgSource.length}"
    #         )

    #     self.clear()

    #     pgDes_list = []
    #     pgSrc_list = []
    #     for i in range(pgDestination.length):
    #         if pgDestination.G[i] >= 0:
    #             pgDes_list.append(pgDestination.G[i])
    #         if pgSource.G[i] >= 0:
    #             pgSrc_list.append(pgSource.G[i])

    #     len_des = len(pgDes_list)
    #     len_src = len(pgSrc_list)
    #     self.sizeDirection = len_des - len_src

    #     num_element = len_src
    #     while len(pgDes_list) > num_element:
    #         index = state.random[thread].randint(0, len(pgDes_list) - 1)
    #         pgDes_list.pop(index)

    #     while len(pgDes_list) < num_element:
    #         insert = state.random[thread].randint(0, len(pgDes_list))
    #         pgDes_list.insert(insert, -1)

    #     allZero = True
    #     for k in range(num_element):
    #         if pgDes_list[k] >= 0:
    #             diff = pgDes_list[k] - pgSrc_list[k]
    #             self.append(float(diff))
    #             if diff != 0:
    #                 allZero = False
    #         else:
    #             self.append(0.0)

    #     if allZero and len(self) > 0:
    #         index = state.random[thread].randint(0, len(self) - 1)
    #         self[index] = float(
    #             state.random[thread].randint(0, sym_rng - 1) - pgDes_list[index]
    #         )

    # def setDirection_mask(
    #     self, pgDestination, pgSource, state, thread, sym_rng, masksize
    # ):
    #     if pgDestination.length != pgSource.length:
    #         raise RuntimeError(
    #             f"we got GenoVectors with inconsistent length in Direction: "
    #             f"{pgDestination.length} and {pgSource.length}"
    #         )

    #     self.clear()

    #     pgDes_list = []
    #     pgSrc_list = []
    #     for i in range(pgDestination.length):
    #         if pgDestination.G[i] >= 0:
    #             pgDes_list.append(pgDestination.G[i])
    #         if pgSource.G[i] >= 0:
    #             pgSrc_list.append(pgSource.G[i])

    #     self.sizeDirection = len(pgDes_list) - len(pgSrc_list)

    #     # mask destination
    #     while len(pgDes_list) > masksize:
    #         remove = state.random[thread].randint(0, len(pgDes_list) - 1)
    #         pgDes_list.pop(remove)

    #     while len(pgDes_list) < masksize:
    #         insert = state.random[thread].randint(0, len(pgDes_list))
    #         pgDes_list.insert(insert, -1)

    #     num_element = len(pgSrc_list)
    #     while len(pgDes_list) > num_element:
    #         index = state.random[thread].randint(0, len(pgDes_list) - 1)
    #         pgDes_list.pop(index)

    #     while len(pgDes_list) < num_element:
    #         insert = state.random[thread].randint(0, len(pgDes_list))
    #         pgDes_list.insert(insert, -1)

    #     allZero = True
    #     for k in range(num_element):
    #         if pgDes_list[k] >= 0:
    #             diff = pgDes_list[k] - pgSrc_list[k]
    #             self.append(float(diff))
    #             if diff != 0:
    #                 allZero = False
    #         else:
    #             self.append(0.0)

    #     if allZero and len(self) > 0:
    #         index = state.random[thread].randint(0, len(self) - 1)
    #         self[index] = float(
    #             state.random[thread].randint(0, sym_rng - 1) - pgDes_list[index]
    #         )

    @staticmethod
    def normalization(dir_list: List[float]):
        if not dir_list:
            return

        norm = 0.0
        for d in dir_list:
            norm += d * d

        norm = math.sqrt(norm)
        if norm == 0:
            return

        for i in range(len(dir_list)):
            dir_list[i] = dir_list[i] / norm

    # @staticmethod
    # def Cosine_direction(des1, src1, des2, src2, state, thread):
    #     norm1 = math.sqrt(norm2Q.Q(des1, src1))
    #     norm2 = math.sqrt(norm2Q.Q(des2, src2))

    #     dir1 = Direction()
    #     dir2 = Direction()

    #     dir1.setDirection(des1, src1, state, thread)
    #     dir2.setDirection(des2, src2, state, thread)

    #     innerProd = 0.0
    #     for i in range(min(len(dir1), len(dir2))):
    #         innerProd += (
    #             norm2Q.subQ(des1.G[i], src1.G[i])
    #             * norm2Q.subQ(des2.G[i], src2.G[i])
    #         )

    #     return innerProd / (norm1 * norm2)
