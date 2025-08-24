import os, sys
from src.ec import *
from src.ec.util import *
from src.lgp.individual import LGPIndividual
from sklearn.base import BaseEstimator
import numpy as np
from abc import ABC, abstractmethod


class LinearGP_Template(BaseEstimator, ABC):

    def __init__(self, param_file:str=None):
        cwd = os.getcwd()
        self.default_paramfile:str = f"{cwd}/tasks/Symbreg/parameters/LGP_regressor_Xy.params"
        
        if not param_file:
            param_file = self.default_paramfile

        args = [
            "-file", 
            param_file,
            "-p seed.0=7"
        ]

        self.output_ind:LGPIndividual = None # the output individual after training

        # running the main()
        job = 0
        numJobs = 1
        # Load parameters
        parameters = Evolve.loadParameterDatabase(args)

        self.state:EvolutionState = Evolve.initialize(parameters, job)
        self.state.output.message(f"Job: {job}")
        self.state.job = [job]
        self.state.runtimeArguments = args

        self.state.output.message("Setting up")
        self.state.setup()  # garbage Parameter equivalent

    def model(self)->LGPIndividual:
        if not self.output_ind:
            print("the linear genetic programming has not been trained. I found no output individual")
            sys.exit(1)
        
        return self.output_ind