

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import warnings

class SupervisedProblem(ABC):
    @abstractmethod
    def getDatanum(self) -> int:
        pass

    @abstractmethod
    def getDatadim(self) -> int:
        pass

    @abstractmethod
    def getOutputnum(self) -> int:
        pass

    @abstractmethod
    def getOutputdim(self) -> int:
        pass

    @abstractmethod
    def getTargets(self) -> List[int]:
        pass

    @abstractmethod
    def getTargetNum(self) -> int:
        pass

    @abstractmethod
    def getDataMax(self) -> List[float]:
        pass

    @abstractmethod
    def getDataMin(self) -> List[float]:
        pass

    @abstractmethod
    def getData(self) -> List[List[float]]:
        pass

    @abstractmethod
    def getDataOutput(self) -> List[List[float]]:
        pass

    @abstractmethod
    def getX(self) -> List[float]:
        pass

    @abstractmethod
    def getX_index(self) -> int:
        pass

    @abstractmethod
    def setX_index(self, ind: int):
        pass

    @abstractmethod
    def istraining(self) -> bool:
        pass

    def setData(self, X:np.ndarray, y:np.ndarray):
        
        self.setX(X)
        self.setY(y)

    def setX(self, X:np.ndarray):

        if X is None:
            # warnings.warn("Setting data X as None. Set it later or result may be invalid", RuntimeWarning)
            return
        
        self.data = X
        self.datanum, self.datadim = X.shape
        self.data_max = np.max(X, axis=0)
        self.data_min = np.min(X, axis=0)

    def setY(self, y:np.ndarray):
        if y is None:
            # warnings.warn("Setting data y as None. Set it later or result may be invalid", RuntimeWarning)
            return

        self.data_output = y
        self.outputnum, self.outputdim = y.shape
