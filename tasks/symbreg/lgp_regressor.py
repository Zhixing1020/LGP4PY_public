import os
import sys
from src.ec import *
from src.ec.util import *
# from src.lgp.individual import LGPIndividual
# from sklearn.base import BaseEstimator, RegressorMixin
from tasks.symbreg.optimization.gp_Xy_symbolic_regression import XySymbolicRegression
from tasks.lgp_template import LinearGP_Template
import numpy as np

class LinearGP_Regressor(LinearGP_Template):

    def __init__(self, param_file:str=None):

        if not param_file:
            cwd = os.getcwd()
            param_file = f"{cwd}/tasks/Symbreg/parameters/LGP_regressor_Xy.params"

        super().__init__(param_file)

    def fit(self, X, y):
        """
        Fit the model to data.
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        """
        """this function will set up the evaluation problem in GP"""

        X = np.asarray(X)
        y = np.asarray(y)

        if isinstance(self.state.evaluator.p_problem, XySymbolicRegression):
            self.state.evaluator.p_problem.setData(X, y)
        else:
            raise ValueError(f"the optimization of LinearGP_Regressor must be type of {XySymbolicRegression.__name__}")
    
        self.state.run()  # C_STARTED_FRESH

        self.output_ind = self.state.statistics.best_of_run[0]  # Assuming single subpopulation

        # If you have more complex algorithm, do training here
        return self  # fit must return self
    
    def predict(self, X)->np.ndarray:
        """
        Predict using the trained model.
        X: array-like of shape (n_samples, n_features)
        Returns: predictions
        """
        """this function will use quickevaluate(output_ind) to output the predict output"""

        X = np.asarray(X)
        
        if isinstance(self.state.evaluator.p_problem, XySymbolicRegression):
            self.state.evaluator.p_problem.setX(X)
        else:
            raise ValueError(f"the optimization of LinearGP_Regressor must be type of {XySymbolicRegression.__name__}")
        
        if not self.output_ind:
            print("the linear genetic programming has not been trained. I found no output individual")
            sys.exit(1)

        #by default, there is only one output. We take the output from the first register
        predict = self.state.evaluator.p_problem.quickevaluate(self.output_ind, X)[:, 0]

        return predict
    
    def score(self, X, y):
        """Return R^2 score, just like scikit-learn"""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)