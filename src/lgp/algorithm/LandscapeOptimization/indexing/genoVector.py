from typing import List
import math
import copy
from src.ec import *

class GenoVector:

    NoneVal = -1   # renamed internally to avoid shadowing Python None
    None_ = -1     # keep Java name accessible if needed

    def __init__(self, length: int, minlength: int):
        self.G = [0] * length
        self.length = length
        self.minlength = minlength
        self.toCheckList: List[int] = []

    def getCheckList(self):
        return self.toCheckList

    def clone(self):
        obj = GenoVector(self.length, self.minlength)
        for g in range(self.length):
            obj.G[g] = self.G[g]
        # do not clone toCheckList (temporary)
        return obj

    # # ============================================================
    # # moveOnDirection with mask
    # # ============================================================
    # def moveOnDirection(self, dir, step, mask, state, thread, sym_rng):

    #     tmpDirection = dir.clone()

    #     # get actual length
    #     origin_GVlist = []
    #     for i in range(self.length):
    #         if self.G[i] >= 0:
    #             origin_GVlist.append(self.G[i])
    #         else:
    #             break
    #     len_gv = len(origin_GVlist)

    #     # align direction
    #     n = len_gv - tmpDirection.size()
    #     for _ in range(n):
    #         insert = state.random[thread].randint(0, tmpDirection.size() - 1)
    #         tmpDirection.add(insert, 0.0)

    #     n = tmpDirection.size() - len_gv
    #     for _ in range(n):
    #         tryi = state.random[thread].randint(0, tmpDirection.size() - 2)
    #         tmpDirection.set(tryi + 1,
    #                          tmpDirection.get(tryi) + tmpDirection.get(tryi + 1))
    #         tmpDirection.remove(tryi)

    #     tmpDirection.normalization(tmpDirection)

    #     gvList = []
    #     num_instr_change = 1
    #     self.toCheckList.clear()

    #     for i in range(len(origin_GVlist)):
    #         toCheckValue = False

    #         if i in mask:
    #             accept_add_remove = state.random[thread].random() < 0.5

    #             if (tmpDirection.sizeDirection > 0 and accept_add_remove and
    #                     len(origin_GVlist) + num_instr_change <= self.length):

    #                 gvList.append(
    #                     int(math.floor(origin_GVlist[i] +
    #                                    (-step + state.random[thread].random() * 2 * step)))
    #                 )
    #                 mask.remove(i)
    #                 i -= 1
    #                 num_instr_change += 1
    #                 tmpDirection.sizeDirection -= 1
    #                 toCheckValue = True
    #                 self.toCheckList.append(len(gvList) - 1)

    #             elif (tmpDirection.sizeDirection < 0 and accept_add_remove and
    #                     len(origin_GVlist) - num_instr_change >= self.minlength):

    #                 tmpDirection.sizeDirection += 1
    #                 mask.remove(i)
    #                 num_instr_change += 1

    #             else:
    #                 gvList.append(
    #                     int(math.floor(origin_GVlist[i] +
    #                                    tmpDirection.get(i) * step))
    #                 )
    #                 mask.remove(i)
    #                 toCheckValue = True
    #                 if tmpDirection.get(i) != 0:
    #                     self.toCheckList.append(len(gvList) - 1)
    #         else:
    #             gvList.append(origin_GVlist[i])

    #         if toCheckValue:
    #             v = gvList[-1]
    #             if v < 0 or v >= sym_rng:
    #                 gvList[-1] = state.random[thread].randint(0, sym_rng - 1)

    #     # write back
    #     for i in range(self.length):
    #         if i < len(gvList):
    #             self.G[i] = gvList[i]
    #         else:
    #             self.G[i] = GenoVector.None_

    # # ============================================================
    # # moveOnDirection without mask
    # # ============================================================
    # def moveOnDirection2(self, dir, step, state, thread, sym_rng):

    #     tmpDirection = dir.clone()

    #     origin_GVlist = []
    #     for i in range(self.length):
    #         if self.G[i] >= 0:
    #             origin_GVlist.append(self.G[i])
    #         else:
    #             break

    #     len_gv = len(origin_GVlist)

    #     n = len_gv - tmpDirection.size()
    #     for _ in range(n):
    #         insert = state.random[thread].randint(0, tmpDirection.size())
    #         tmpDirection.add(insert, 0.0)

    #     n = tmpDirection.size() - len_gv
    #     for _ in range(n):
    #         if tmpDirection.size() == 1:
    #             break
    #         tryi = state.random[thread].randint(0, tmpDirection.size() - 2)
    #         if abs(tmpDirection.get(tryi)) > abs(tmpDirection.get(tryi + 1)):
    #             tmpDirection.set(tryi + 1, tmpDirection.get(tryi))
    #         tmpDirection.remove(tryi)

    #     gvList = []
    #     self.toCheckList.clear()

    #     cache = []
    #     cacheActive = False
    #     num_instr_change = 1
    #     num_instr_changed = 0

    #     for i in range(len(origin_GVlist)):
    #         if tmpDirection.sizeDirection > 0 and tmpDirection.get(i) != 0:

    #             if not cacheActive:
    #                 cacheActive = True

    #             newsym = GenoVector.moveSymbol(
    #                 state, thread,
    #                 origin_GVlist[i],
    #                 tmpDirection.get(i),
    #                 step
    #             )

    #             if newsym < 0 or newsym >= sym_rng:
    #                 newsym = state.random[thread].randint(0, sym_rng - 1)

    #             cache.append(newsym)

    #         if ((i == len(origin_GVlist) - 1 or tmpDirection.get(i) == 0)
    #                 and cache):

    #             cnt = min(len(cache), tmpDirection.sizeDirection)
    #             cnt = min(cnt, self.length - (len(origin_GVlist) + num_instr_changed))

    #             start = i - cnt
    #             end = i
    #             if i == len(origin_GVlist) - 1 and tmpDirection.get(i) != 0:
    #                 start += 1
    #                 end += 1

    #             for j in range(start, end):
    #                 cache.append(origin_GVlist[j])
    #                 num_instr_changed += 1

    #             for j, newsym in enumerate(cache):
    #                 gvList.append(newsym)
    #                 if j < len(cache) - cnt:
    #                     self.toCheckList.append(len(gvList) - 1)

    #             tmpDirection.sizeDirection -= cnt

    #             if tmpDirection.get(i) == 0:
    #                 i -= 1

    #             cache.clear()
    #             cacheActive = False
    #             continue

    #         if not cacheActive:
    #             newsym = GenoVector.moveSymbol(
    #                 state, thread,
    #                 origin_GVlist[i],
    #                 tmpDirection.get(i),
    #                 step
    #             )
    #             if newsym < 0 or newsym >= sym_rng:
    #                 newsym = state.random[thread].randint(0, sym_rng - 1)

    #             gvList.append(newsym)
    #             if tmpDirection.get(i) != 0:
    #                 self.toCheckList.append(len(gvList) - 1)

    #     # remove symbols
    #     while (tmpDirection.sizeDirection < 0 and
    #            len(origin_GVlist) - num_instr_change >= self.minlength):

    #         index = state.random[thread].randint(0, len(gvList) - 1)
    #         gvList.pop(index)

    #         tmpDirection.sizeDirection += 1
    #         num_instr_change += 1

    #         for j in range(len(self.toCheckList)):
    #             if self.toCheckList[j] >= index:
    #                 self.toCheckList[j] -= 1
    #         if index - 1 in self.toCheckList:
    #             self.toCheckList.remove(index - 1)

    #     for i in range(self.length):
    #         if i < len(gvList):
    #             self.G[i] = gvList[i]
    #         else:
    #             self.G[i] = GenoVector.None_

    # ============================================================
    # static moveSymbol
    # ============================================================
    @staticmethod
    def moveSymbol(state:EvolutionState, thread:int, oldsym:int, direction:float, confiStep:float):
        return int(math.floor(
            oldsym +
            math.copysign(
                max(1, abs(direction) * state.random[thread].uniform(0,1) * confiStep),
                direction
            )
        ))
