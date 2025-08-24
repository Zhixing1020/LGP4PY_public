"""The :mod:`lgp` module provides the necessary methods and classes to perform
Linear Genetic Programming (LGP). It provides necessary functions run LGP.
The behaviors of LGP is defined by a parameter file.

This file is equivalent to "Evolve.java" in ECJ
"""



# import classes and files
import sys
import os

# Get current working directory
cwd = os.getcwd()

# Insert at the front of sys.path (highest priority)
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from src.ec import *
from src.ec.util import *
from tasks.symbreg.lgp_regressor import LinearGP_Regressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# set up LGP behaviors
'''read parameters from the parameter file and set it into pset and toolbox'''

# define main() and run main()  

if __name__ == "__main__":

    # Example data
    X = np.random.rand(100, 1)*2-1
    y = X**6+X**5+X**4+X**3+X**2+X

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


    lgp = LinearGP_Regressor()
    lgp.fit(X_train, y_train)

    pred = lgp.predict(X_test)

    print(f"test R2: {r2_score(y_test, pred)}")
    # Evolve.main()