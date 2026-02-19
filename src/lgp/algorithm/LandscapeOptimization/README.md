# Fitness Landscape Optimization for GP search #

This package implements the papers [1], [2]

[1] Z. Huang, Y. Mei, F. Zhang, M. Zhang, and W. Banzhaf, “Fitness Landscape Optimization Makes Stochastic Symbolic Search By Genetic Programming Easier,” IEEE Trans. Evol. Computat., pp. 1–1, 2025, doi: 10.1109/TEVC.2024.3525006.

[2] Z. Huang et al. "Fitness Landscape Compression for Genetic Programming-based Symbolic Search".

P.S. This package does not implement the full experiments in [1]. It only implemented the commonly used classes in [1].

### Project Structure ###

**`subpopulationFLO.py`**

This class defines additional members and methods in the GP population of the fitness landscape optimization.

**`FLReductionLGP`**

This package implements the algorithm of fitness landscape compression [2]. 
* `indexing.indexList4LGP_FLR.py` implements the `IndexList4LGP` that can dynamically add or remove itmes.
* `individual.reproduce.neighborhoodSearchFast.py` implements the `OAM` operator in [2].

**`indexing`**

* `Board` defines the class of a list to record the competitive individuals. The competitive individuals are used to evaluate and optimize fitness landscape objectives.
* `BoardItem` defines the class of the items in `Board`.
* `Direction` defines the class specifying the modification direction of the genotype vector (i.e., movement on the fitness landscape).
* `GenoVector` defines the class of genotype vector, including how it moves along the direction.
* `Index` defines the abstract class of a "symbol" in the symbolic search problems (i.e., a symbol and its index on the fitness landscape).
* `IndexList` defines the abstract class of the list of `Index` (i.e., `IndexList` forms axes of the fitness landscape).
* `IndexSymbolBuilder` defines the abstract class of constructing (enumerating) the symbols.

**`objectives`**

* `objective4FLO.py` defines the abstract class of objective functions for the fitness landscape. 
* `distance.py` defines the objective function of the distance between good solutions.

The following objective is used in [2], working with an origin-attracting move. [2] only uses `Distance` and `L2NORM`.
* `L2NORM.py` defines an L-2 normalization on instruction indices, encouraging prioritizing effective instructions in an ascending order.

**`reproduce`**

* `neighborhoodSearch.py` defines an abstract class of moving genotype against the optimized landscape.

**`simpleLGP`**

This package implements the fitness landscape optimization algorithm for LGP. 

**`simpleLGP/indexing`**

* `index4LGP.py` implements the class of an LGP instruction (as an index item) in the symbolic search problems.
* `indexList4LGP.py` implements the class of the list of `Index4LGP`.

### Running Examples ###

Refer to [applying fitness landscape reduction to symbolic regression](../../../../tasks/symbreg/algorithm/LandscapeOptimization/).
