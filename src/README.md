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
