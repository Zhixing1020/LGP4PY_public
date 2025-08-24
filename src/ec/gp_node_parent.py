from abc import ABC, abstractmethod
import copy

class GPNodeParent(ABC):
    def __init__(self):
        pass

    # def __deepcopy__(self):
    #     # self.clone()
    #     return self.__class__()

    # @abstractmethod
    def clone(self)->'GPNodeParent':
        return self.__class__()