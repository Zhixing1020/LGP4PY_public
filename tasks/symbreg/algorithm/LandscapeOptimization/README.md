
### Running Examples ###

**Example 1 - Applying LGP with fitness landscape reduction to symbolic regression by `LinearGP_Regressor`**

1. Locate the parameter file [FLR-LGP-SR.params](./FLReduction/parameters/FLR-LGP-SR.params).
2. Locate the [training and test dataset](./dataset/).
3. Locate the main function [lgp4SR_main.py](../../lgp4SR_main.py)
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
         "${workspaceFolder}/tasks/symbreg/algorithm/LandscapeOptimization/FLReduction/parameters/FLR-LGP-SR.params",
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

**Example 2 - Applying LGP with fitness landscape reduction to symbolic regression in the BaseEstimator style of sklearn**

Here is a simple example. We have to prepare the `X_train`, `X_test`, `y_train`, and `y_test`, and use `fit(X, y)` and `predict(X)` to train and test the LGP model.
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
    param_file = f"{cwd}/tasks/symbreg/algorithm/LandscapeOptimization/FLReduction/parameters/FLR-LGP-SR.params"

    lgp = LinearGP_Regressor(param_file=param_file)
    lgp.fit(X_train, y_train)

    pred = lgp.predict(X_test)

    print(f"test R2: {r2_score(y_test, pred)}")
```
