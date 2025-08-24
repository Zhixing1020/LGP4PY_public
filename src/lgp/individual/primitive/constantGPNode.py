
from src.ec.gp_node import GPNode
import numpy as np
class ConstantGPNode(GPNode):
    range = []  # shared across instances

    def __init__(self, val: float = 1, begin: float = 0.0, end: float = 1.0, step: float = 0.1):
        super().__init__()
        self.children = []
        self.lb = begin
        self.ub = end
        self.step = step

        if val is not None:
            self.value = val
        else:
            ConstantGPNode.range = [round(i, 10) for i in np.arange(begin, end + step / 2, step)]
            self.value = begin

    def getValue(self):
        return self.value

    def getRange(self):
        return ConstantGPNode.range

    def setValue(self, v):
        self.value = v

    def __str__(self):
        return str(self.value)

    def expectedChildren(self):
        return 0

    def eval(self, state, thread, input, individual, problem, argval: list[float] = None):
        if not input.to_vectorize:
            input.value = self.value
        else:
            input.values = self.value

    # def __hash__(self):
    #     return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, ConstantGPNode) and self.value == other.getValue()

    def resetNode(self, state, thread):
        self.value = state.random[thread].uniform(self.lb, self.ub)

    def lightClone(self):
        clone = super().lightClone()
        clone.setValue(self.value)
        clone.range = self.range
        clone.lb = self.lb
        clone.ub = self.ub
        return clone
