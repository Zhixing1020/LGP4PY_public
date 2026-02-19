
from src.ec import *
from src.ec.util import *
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List

T = TypeVar("T")

class IndexSymbolBuilder(GPBuilder, Generic[T], ABC):
    SYMBOLBUILDER = "symbolbuilder"
    P_MAXNUMBERSYMBOLS = "maxnumsymbols"

    # static
    maxNumSymbols = 0  # the maximum number of symbols the builder can enumerate

    def __init__(self):
        super().__init__()
        self.initialized = False

    def setup(self, state:EvolutionState, base:Parameter):
        super().setup(state, base)

        default = self.defaultBase()

        IndexSymbolBuilder.maxNumSymbols = state.parameters.getInt(
            base.push(self.P_MAXNUMBERSYMBOLS),
            default.push(self.P_MAXNUMBERSYMBOLS)
        )
        if IndexSymbolBuilder.maxNumSymbols <= 0:
            state.output.fatal(
                "The maximum number of symbols for a symbol builder must be at least 1.",
                base.push(self.P_MAXNUMBERSYMBOLS),
                default.push(self.P_MAXNUMBERSYMBOLS)
            )

    def defaultBase(self):
        return Parameter(self.SYMBOLBUILDER)

    # @abstractmethod
    # def enumerateSymbols(
    #     self,
    #     state:EvolutionState,
    #     type,
    #     thread:int,
    #     argposition:int,
    #     set:GPPrimitiveSet
    # ) -> List[T]:
    #     pass