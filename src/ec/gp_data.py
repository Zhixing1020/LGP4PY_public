from dataclasses import dataclass
import numpy as np
# from copy import deepcopy

@dataclass
class GPData:
    value:float = 0.0
    values:np.array = None
    to_vectorize:bool = False

    def setup(self, state, base):
        pass

    # def __deepcopy__(self):
    #     d = self.__class__
    #     d.value = self.value

    # def __init__(self):
    #     self.value = 0.0

    # def clone(self):
    #     d = self.__class__
    #     d.value = self.value