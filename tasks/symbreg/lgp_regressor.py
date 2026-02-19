import os
import sys
from src.ec import *
from src.ec.util import *
# from src.lgp.individual import LGPIndividual
# from sklearn.base import BaseEstimator, RegressorMixin
# from tasks.symbreg.optimization.gp_Xy_symbolic_regression import XySymbolicRegression
from tasks.symbreg.optimization.gp_symbolic_regression import GPSymbolicRegression
from tasks.lgp_template import LGP_Model_Template
import numpy as np
import time

class LinearGP_Regressor(LGP_Model_Template):

    def __init__(self, param_file:str, seed:int=7, setup_problem_script=False,
                 output_file1:str=None,
                 output_file2:str=None):
        '''
        when the output_file1 & 2 are none, the output file paths will follow the parameter file settings.
        '''

        # if not param_file:
        #     cwd = os.getcwd()
        #     param_file = f"{cwd}/tasks/Symbreg/parameters/LGP_regressor_Xy.params"

        super().__init__(param_file, seed, setup_problem_script=setup_problem_script,
                         output_file1=output_file1,
                         output_file2=output_file2)

    def fit(self, X, y):
        """
        Fit the model to data.
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        """
        """this function will set up the evaluation problem in GP"""
        """this function is reserved for implementing the BaseEstimator in sklearn.base"""

        X = np.asarray(X)
        y = np.asarray(y)

        self.state.startFresh()

        if isinstance(self.state.evaluator.p_problem, GPSymbolicRegression):
            self.state.evaluator.p_problem.setData(X, y)
        else:
            raise ValueError(f"the optimization of LinearGP_Regressor must be type of {GPSymbolicRegression.__name__}")
    
        self.state.run() 

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
        """this function is reserved for implementing the BaseEstimator in sklearn.base"""

        X = np.asarray(X)
        
        if isinstance(self.state.evaluator.p_problem, GPSymbolicRegression):
            self.state.evaluator.p_problem.setX(X)
        else:
            raise ValueError(f"the optimization of LinearGP_Regressor must be type of {GPSymbolicRegression.__name__}")
        
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
    
    def fit_params(self, loca:str, datan:str):
        """fit the model based on given parameters
        loca: the file location of SR dataset.
        datan: the name of the SR task.
        """
        self.state.startFresh()

        if isinstance(self.state.evaluator.p_problem, GPSymbolicRegression):
            self.state.evaluator.p_problem.load_data(self.state, loca, datan, istraining=True)
        else:
            raise ValueError(f"the optimization of LinearGP_Regressor must be type of {GPSymbolicRegression.__name__}")
    
        t0t = time.time()
        self.state.run()  
        self.train_time = time.time() - t0t
        print('Training time measure:', self.train_time)

        # self.output_ind = self.state.statistics.best_of_run[0]  # Assuming single subpopulation
        self.output_ind = self.state.statistics.best_i[0]
        self.train_fitness = self.output_ind.fitness.fitness()

        # If you have more complex algorithm, do training here
        return self  # fit must return self

    def predict_params(self, loca:str, datan:str):
        """
        Predict using the trained model.
        loca: the file location of SR dataset.
        datan: the name of the SR task.
        """
        """this function will use simplevaluate(output_ind) to get the predict performance based on the given fitness function in parameters"""

        if isinstance(self.state.evaluator.p_problem, GPSymbolicRegression):
            self.state.evaluator.p_problem.load_data(self.state, loca, datan, istraining=False)
        else:
            raise ValueError(f"the optimization of LinearGP_Regressor must be type of {GPSymbolicRegression.__name__}")
        
        if not self.output_ind:
            print("the linear genetic programming has not been trained. I found no output individual")
            sys.exit(1)

        self.state.evaluator.p_problem.simpleevaluate(self.output_ind)
        self.test_fitness = self.output_ind.fitness.fitness()

    def play_quiz(self, loca:str, datan:str):
        self.fit_params(loca, datan)

        self.predict_params(loca, datan)
