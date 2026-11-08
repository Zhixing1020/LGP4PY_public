

# sub class of GPData for ARC tasks
from src.ec import *
from typing import Any
from dataclasses import dataclass


@dataclass
class TypedData(GPData):
    value: Any = None
    values: Any = None
    to_vectorize:bool = False
    type:Any = None