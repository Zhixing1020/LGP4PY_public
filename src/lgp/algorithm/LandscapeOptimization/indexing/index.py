from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar
import numpy as np
from src.ec.util.parameter import Parameter

from src.ec import *

T = TypeVar("T")


class Index(ABC, Generic[T]):
    """
    Index of the symbols with the same genotype or phenotype or semantics
    """

    # ---- Static / class-level constants (ECJ Parameters) ----
    INDEX = "index"

    NUM_INPUTS = "num_inputs"
    DIM_INPUTS = "dim_inputs"
    SYMBOL_PROTO = "symbol_prototype"

    TABU_THRESHOLD = 0.95

    def __init__(self):
        # ---- Instance fields ----
        self.num_inputs: int = 0
        self.dim_inputs: int = 0

        self.input_lb: float = -100.0
        self.input_ub: float = 100.0

        self.sym_prototype: T | None = None

        self.index: int = -1  # unique index in index list
        self.symbols: List[T] = []  # symbols with same genotype/phenotype/semantics
        self.symbol_names: set[str] = set() # name of symbols in the index item

        self.tabu_frequency: float = 0.0

    def setup(self, state: EvolutionState, base: Parameter):

        default = Parameter(self.INDEX)

        self.num_inputs = state.parameters.getIntWithDefault(
            base.push(self.NUM_INPUTS),
            default.push(self.NUM_INPUTS),
            1
        )
        if self.num_inputs <= 0:
            state.output.fatal(
                "Index must have a number of inputs larger than 0"
            )

        self.dim_inputs = state.parameters.getIntWithDefault(
            base.push(self.DIM_INPUTS),
            default.push(self.DIM_INPUTS),
            2
        )
        if self.dim_inputs <= 0:
            state.output.fatal(
                "Index must have a dimension of inputs larger than 0"
            )



    @abstractmethod
    def isduplicated(self, newsym: T) -> bool:
        """
        Decide whether a new symbol is duplicated
        (genotype / phenotype / semantics depending on subclass)
        """
        pass

    @abstractmethod
    def clone(self):
        """
        ECJ-style clone (explicit, not __copy__)
        """
        pass

    def addSymbol(self, sym: T):
        self.symbols.append(sym)
        self.symbol_names.add(str(sym))

    def set_tabu_frequency(self, val: float):
        self.tabu_frequency = val

    def get_tabu_frequency(self) -> float:
        return self.tabu_frequency
