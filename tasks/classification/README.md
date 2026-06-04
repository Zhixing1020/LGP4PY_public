This folder applies LGP to solve classification problems.

### Project Structure ###

* `ec` defines the classification evolutionary framework elements.

* `algorithm` defines the detailed implementations of LGP algorithms.

* `DSL` defines the domain-specific language for classification tasks.

* `individual` defines the basic LGP individual for solving classification problems.

* `optimization` defines the class of classification problems by `lgp_classification.py`.

* `parameters` includes the parameter files.

* `ruleanalysis` defines the test procedure of basic LGP on classification.

* `util` defines the parsing functions for basic LGP classification helpers.

* `lgp_classifier.py` wraps the LGP training process as a single model based on the LGP template.

* `lgp4Class_main.py` implements the main function, inherited from `MainRunner`.


### Running Examples ###

**Example 1 - Applying basic LGP to classification by `LinearGP_Classifier`**

1. Locate the parameter file [LGP_classification_test.params](./parameters/LGP_classification_test.params). Note that please set `pop.subpop.0.species.fitness.maximize` in the parameter file. If the fitness function is a minimization objective like `ERR`, set `False`. Otherwise, like `ACC`, set `True`.
2. Prepare a dataset directory that contains the classification training and test data files.
3. Locate the main function [lgp4Class_main.py](./lgp4Class_main.py).
4. The "configurations" in `launch.json` can be:
```
{
     "name": "run lgp4Class by main and params",
     "type": "python",
     "request": "launch",
     "program": "${workspaceFolder}/tasks/classification/lgp4Class_main.py",
     "console": "integratedTerminal",
     "args": [
         "${workspaceFolder}",
         "-dir",
         "${workspaceFolder}/tasks/classification/dataset/",
         "-data",
         "YourClassificationTaskName",
         "-seed",
         "7",
         "-results",
         "${workspaceFolder}/",
         "-log",
         "${workspaceFolder}/",
         "-params",
         "${workspaceFolder}/tasks/classification/parameters/LGP_classification_test.params"
     ],
     "noDebug": true,
     "env": {
         "PYTHONOPTIMIZE": "1",
         "PYTHONHASHSEED": "1000117"
     }
 }
```
5. Run the configuration "run lgp4Class by main and params" in VSCode.
6. Finally, you will get three result files (if you only run one time with a random seed 7) `job.0.out.stat`, `job.0.outtabular.stat`, and `YourClassificationTaskName_7.json` in the project home directory (i.e., specified by the two `${workspaceFolder}/` paths). `job.0.out.stat` and `job.0.outtabular.stat` are logging files, and `YourClassificationTaskName_7.json` is the result file.
The format of `outtabular.stat` is
``[Generation index] [Population mean fitness]\t[Best fitness per generation]\t[Best fitness so far]\t[Population mean absolutate program length]\t[Population mean effective program length]\t[Population average effective rate]\t[Absolute program length of the best individual]\t[Effective program length of the best individual]\t[Effective rate of the best individual]\t[running time so far in seconds]``.

**Example 2 - Applying basic LGP to classification in the BaseEstimator style of sklearn**

Here is a simple [example](../../src/lgp.py). We have to prepare the `X_train`, `X_test`, `y_train`, and `y_test`, and use `fit(X, y)` and `predict(X)` to train and test the LGP model.
```
import sys
import os
from src.ec import *
from src.ec.util import *
from tasks.classification.lgp_classifier import LinearGP_Classifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    # Example data: three-class problem in 2D
    np.random.seed(42)
    n = 120
    X = np.random.rand(n, 2) * 4 - 2
    score = X[:, 0] + X[:, 1]
    y = np.zeros(n, dtype=float)
    y[score > -0.5] = 1.0
    y[score > 0.8] = 2.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    cwd = os.getcwd()
    param_file = f"{cwd}/tasks/classification/parameters/LGP_classification_test.params"

    lgp = LinearGP_Classifier(param_file=param_file)
    lgp.fit(X_train, y_train)

    pred = lgp.predict(X_test)

    print(f"test accuracy: {accuracy_score(y_test, pred):.4f}")
```

A more visual example is also provided in [src/lgp.py](../../src/lgp.py) by setting `test_Classification(dataset_name="gaussianq3")`. It runs a 2D classification task and plots the training data, test data, and misclassified test instances on a plane.

**Example 3 - Testing basic LGP in classification**

This procedure mainly tests the best-of-generation individuals based on the log files. The implementation is mainly in [ruleanalysis](./ruleanalysis). The main idea of this procedure is to read the LGP programs from the `.txt` log files and evaluate them on test data, i.e., get the test performance.

To run this procedure, we can:
1. There should already be log files (`job.x.out.stat` and `job.x.outtabular.stat`) in the working directory.
2. Locate the parameter file [LGP_classification_test.params](./parameters/LGP_classification_test.params).
3. Prepare a dataset directory that contains the classification training and test data files.
4. Locate the main function [ruletest4LGP_class.py](./ruleanalysis/ruletest4LGP_class.py).
5. The "configurations" in `launch.json` can be:
     ```
     {
            "name": "Run ruletest for lgp classification",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/tasks/classification/ruleanalysis/ruletest4LGP_class.py",
            "console": "integratedTerminal",
            "args": [
                "${workspaceFolder}/",
                "${workspaceFolder}/tasks/classification/dataset/",
                "YourClassificationTaskName",
                "1",
                "100",
                "100",
                "1",
                "ERR",
                "${workspaceFolder}/tasks/classification/parameters/LGP_classification_test.params",
                "-p eval.problem.Kfold_index=0"
            ],
            "noDebug": true,
            "env": {
                "PYTHONOPTIMIZE": "1",
                "PYTHONHASHSEED": "1000117"
            }
        }
     ```
     where `${workspaceFolder}/` is the log directory, `${workspaceFolder}/tasks/classification/dataset/` is the path of the dataset, `YourClassificationTaskName` is the name of the classification task, `1` is the number of runs, `100` is the number of registers, `100` is the maximum number of iterations, `1` and `ERR` are the number of objectives (i.e., fitness) and the fitness function, `${workspaceFolder}/tasks/classification/parameters/LGP_classification_test.params` is the path of the parameters, and `-p eval.problem.Kfold_index=0` is the k-fold index.
   If there are multiple runs, set the number of runs (e.g., `n`). The program will automatically read the log files from `job.0` to `job.n-1`.
6. Now, we can run the `ruletest4LGP_class.py` in VSCode.
