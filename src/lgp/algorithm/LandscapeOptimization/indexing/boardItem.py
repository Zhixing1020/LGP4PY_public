from functools import total_ordering
from src.ec import *

@total_ordering
class BoardItem(list[GPIndividual]):
    """
    an item of the leading board, including fitness and individuals
    """

    def __init__(self, ind: GPIndividual):
        super().__init__()
        # clone fitness (ECJ-style)
        self.fitness:Fitness = ind.fitness.clone()
        self.weight = 1.0
        super().append(ind)

    # ===== Comparable<BoardItem> =====
    def compareTo(self, o2:"BoardItem"):
        if self.fitness.betterThan(o2.fitness):
            return -1
        elif self.fitness.equivalentTo(o2.fitness):
            return 0
        return 1

    def __lt__(self, other):
        return self.compareTo(other) < 0

    def __eq__(self, other):
        if isinstance(other, BoardItem):
            return self.fitness.equivalentTo(other.fitness)
        return False

    # ===== override add =====
    def add(self, ind:GPIndividual)->bool:
        if ind.fitness.equivalentTo(self.fitness):
            self.append(ind)
            return True
        return False

    # ===== clone =====
    def lightClone(self):
        item = BoardItem(self[0])
        for i in range(1, len(self)):
            item.add(self[i])
        item.weight = self.weight
        return item