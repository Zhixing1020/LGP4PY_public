import sys
from src.ec import *
from src.ec.util import *
from src.lgp.individual.primitive.inputFeatureGPNode import InputFeatureGPNode
from tasks.classification.optimization.lgp_classification import LGPClassificationProblem

from tasks.classification.individual.lgpindividual4Class import LGPIndividual4Class

class InputFeature4Class(InputFeatureGPNode):

    def __init__(self, ind=None, size=None):
        if ind is not None and size is not None:
            super().__init__(ind, size)
        elif ind is not None:
            super().__init__(ind)
        else:
            super().__init__()

    def eval(self, state:EvolutionState, thread:int, input:GPData, individual: LGPIndividual4Class, problem: LGPClassificationProblem):
        if problem.getDatadim() != self.range and state is not None:
            self.setRange(problem.getDatadim())
            self.index = state.random[thread].randint(0, self.range-1)

        data = input  # DoubleData
        if self.index < len(problem.X):
            data.value = problem.X[self.index]
        else:
            sys.stderr.write("The input index exceeds the data dimension\n")
            sys.exit(1)

    def lightClone(self):
        n = super().lightClone()
        n.setIndex(self.index)
        n.setRange(self.range)
        return n