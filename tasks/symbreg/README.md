This folder applies LGP to solve symbolic regression problems.

### Project Structure ###

* `ec` defines the symbolic regression evolutionary framework elements.

* `algorithm` defines the detailed implementations of LGP algorithms.

* `individual` defines the basic LGP individual for solving SR problems.

* `dataset` includes the training and test data of SR benchmarks.

* `optimization` defines the class of SR problems by `gp_symbolic_regression.py`.

* `parameters` includes the parameter files.

* `ruleanalysis` defines the test procedure of basic LGP on SR.

*  `statistics` defines the logging functions of LGP for SR. (empty right now)

*  `util` defines the parsing functions for basic LGP (i.e., draw the DAG of LGP programs).

*  `lgp_regressor.py` wraps the LGP training process as a single model based on the LGP template.

*  `lgp4SR_main.py` implements the main function, inherited from `MainRunner`


### Running Examples ###

**Example 1 - Applying basic LGP to symbolic regression by `LinearGP_Regressor`**

1. Locate the parameter file [LGP_test.params](./parameters/LGP_test.params).
2. Locate the [training and test dataset](./dataset/).
3. Locate the main function [lgp4SR_main.py](./lgp4SR_main.py)
4. The "configurations" in `launch.json` can be:
```
{
     "name": "run lgp4SR by main and params",
     "type": "python",
     "request": "launch",
     "program": "${workspaceFolder}/tasks/symbreg/lgp4SR_main.py",
     "console": "integratedTerminal",
     "args": [
         "${workspaceFolder}",
         "-dir",
         "${workspaceFolder}/tasks/symbreg/dataset/",
         "-data",
         "Nguyen4",
         "-seed",
         "7",
         "-results",
         "${workspaceFolder}/",
         "-log",
         "${workspaceFolder}/",
         "-param", 
         "${workspaceFolder}/tasks/symbreg/parameters/LGP_test.params",
     ],
     "noDebug": true,
     "env": {
         "PYTHONOPTIMIZE": "1",
         "PYTHONHASHSEED": "1000117"
     }
 }
```
   
6. Finally, you will get three result files (if you only run one time with a random seed 7) `job.0.out. stat`, `job.0.outtabular.stat`, and `Nguyen4_7.json` in the project home directory (i.e., specified by the two `"${workspaceFolder}/",` respectively). `job.0.out.stat` and `job.0.outtabular.stat` are logging files, and `Nguyen4_7.json` is the result file.
The format of `outtabular.stat` is
``[Generation index] [Population mean fitness]\t[Best fitness per generation]\t[Best fitness so far]\t[Population mean absolutate program length]\t[Population mean effective program length]\t[Population average effective rate]\t[Absolute program length of the best individual]\t[Effective program length of the best individual]\t[Effective rate of the best individual]\t[running time so far in seconds]``.

**Example 2 - Applying basic LGP to symbolic regression in the BaseEstimator style of sklearn**

Here is a simple [example](../../src/lgp.py). We have to prepare the `X_train`, `X_test`, `y_train`, and `y_test`, and use `fit(X, y)` and `predict(X)` to train and test the LGP model.
```
import sys
import os
from src.ec import *
from src.ec.util import *
from tasks.symbreg.lgp_regressor import LinearGP_Regressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

if __name__ == "__main__":

    # Example data
    X = np.random.rand(100, 1)*2-1
    y = X**6+X**5+X**4+X**3+X**2+X

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    cwd = os.getcwd()
    param_file = f"{cwd}/tasks/Symbreg/parameters/LGP_test.params"

    lgp = LinearGP_Regressor(param_file=param_file)
    lgp.fit(X_train, y_train)

    pred = lgp.predict(X_test)

    print(f"test R2: {r2_score(y_test, pred)}")
```

**Example 3 - Testing basic LGP in symbolic regression**

This procedure mainly tests the best-of-generation individuals based on the log files. The implementation is mainly in [ruleanalysis](./ruleanalysis). The main idea of this procedure is to read the LGP programs from the .txt log files and to evaluate them on test data, i.e., get the test performance.

To run this procedure, we can:
1. There should already be log files (`job.x.out.stat` and `job.x.outtabular.stat`) in the working directory.
2. Locate the parameter file [LGP_test.params](./parameters/LGP_test.params).
3. Locate the [training and test dataset](./dataset/).
4. Locate the main function [ruletest4lgp_sr.py](./ruleanalysis/ruletest4lgp_sr.py)
5. The "configurations" in `launch.json` can be:
     ```
     {
            "name": "Run ruletest for lgp.py",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/tasks/symbreg/ruleanalysis/ruletest4lgp_sr.py",
            "console": "integratedTerminal",
            "args": [
                "${workspaceFolder}/",
                "${workspaceFolder}/tasks/symbreg/dataset/",
                "Nguyen4",
                "1",
                "100",
                "100",
                "1",
                "RSE",
                "${workspaceFolder}/tasks/symbreg/parameters/LGP_test.params",
                "-p eval.problem.Kfold_index=0"
            ],
            "noDebug": true,
            "env": {
                "PYTHONOPTIMIZE": "1",
                "PYTHONHASHSEED": "1000117"
            }
        }
     ```
     where `${workspaceFolder}/` is log directory, `${workspaceFolder}/tasks/symbreg/dataset/` is the path of data set, `Nguyen4` is the name of SR data set, `1` is the number of runs, `100` is the number of registers, `100` is the maximum number of iterations, `1` and `RSE` are the number of objectives (i.e., fitness) and the fitness function, `${workspaceFolder}/tasks/symbreg/parameters/LGP_test.params` is the path of the parameters, `-p eval.problem.Kfold_index=0` is the k-fold index.
   If there are multiple runs, set the number of runs (e.g., ``n``). The program will automatically read the log files from `job.0` to `job.n-1`.
6. Now, we can run the `ruletest4lgp_sr.py` in VSCode.
