### Project Structure ###

* `ec` defines the basic elements for implementing LGP evolutionary framework.
  - `evolve.py` defines the main entry of running LGP algorithms. It has a main function.
  - `statistics` defines the statistic classes, recording the information during the evolutionary process. `simple_statistics.py` and `simple_short_statistics.py` implement the functionality of the same classes in [Linear-Genetic-Programming-LGP-and-Applications](https://github.com/Zhixing1020/Linear-Genetic-Programming-LGP-and-Applications).
  - `util` defines the utilization classes, such as parameter management (`parameter.py` and `parameter_database.py`) and output (`output.py`).
 
* `lgp`
  - `algorithm` defines the abstract classes of each algorithm.
      - `LandscapeOptimization` implements [fitness landscape compression](./lgp/algorithm/LandscapeOptimization).
      - `typed_lgp` implements [typed linear genetic programming](./lgp/algorithm/typed_lgp).

  - `individual`
      - `primitive` defines the commonly used and problem-independent primitives, such as read- and write-registers.
      - `reproduce` defines the basic genetic operators of LGP, including linear crossover, macro- and micro-mutation. It also defines the `multi_breeding_pipeline.py` and `lgp_node_selector.py`.
      - `gp_tree_struct.py` defines the LGP instruction class.
      - `lgp_individual.py` defines the LGP individual class. It also defines two sub classes: `AtomicInteger` and `LGPDefaults`.

  - `species` defines the LGP species

  - `statistics` defines the statistical classes for LGP

* `lgp.py` a simple demo with a main function. P.S. we would use ```Evolve.main()``` to run LGP with more detailed parameter settings.


## How to develop a new algorithm ##

* If this algorithm is LGP-related (e.g., using LGP's individual or using LGP's evolutionary framework):
  1. Set up an independent folder "algorithm_1" in the `src.lgp.algorithm` folder.
  2. Because the new algorithm is LGP-related, you are supposed to reuse a lot of existing LGP-related code, including primitives, individual, genetic operators, and evolutionary framework. You only need to define the new component (e.g., a new genetic operator).
  3. If you are going to develop a new genetic operator, you can 1) set up a new folder "reproduce" in the `src.lgp.algorithm.algorithm_1` folder. 2) develop the new genetic operator, extended from `src.ec.BreedingPipeline`. You can refer to [`neighborhoodSearch.py`](./lgp/algorithm/LandscapeOptimization/reproduce/neighborhoodSearch.py) for an example.
  4. If you are going to use an evolutionary framework different from the original one, you should develop a `src.lgp.algorithm.algorithm_1.ec` folder and develop "evolution_state_alg1.py" in the folder. All other evolutionary-framework-related code should be defined in the `ec` folder. You can refer to [typed_lgp's evolutionary framework](./lgp/algorithm/typed_lgp/ec) for an example.
  5. If you are going to define a new individual representation (but mostly inherit from LGP), you should develop a `src.lgp.algorithm.algorithm_1.individual` folder and develop "alg1_individual.py" in the folder. All other individual-related code (e.g., gp_tree_structure, primitives, reproduce(i.e., genetic operators)) should be defined in the `individual` folder. You can refer to [typed_lgp's individual](./lgp/algorithm/typed_lgp/individual) for an example.
  6. Please develop other new algorithm-specific components in the `src.lgp.algorithm.algorithm_1` folder as well.
  7. Note that **DO NOT** use code in tasks, for the sake of class decoupling.
 
* If this algorithm is not LGP-related but GP-related:
  1. set up an independent folder "new_GP" in the `src` folder. The design of other folders refers to those of LGP. In this case, you will have to reimplement many necessary components of a GP algorithm.
