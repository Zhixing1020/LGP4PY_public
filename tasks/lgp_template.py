import os, sys
from src.ec import *
from src.ec.util import *
from src.lgp.individual import LGPIndividual
from sklearn.base import BaseEstimator
import numpy as np
from abc import ABC, abstractmethod


class LGP_Model_Template(BaseEstimator, ABC):
    '''
    This template defines the behaviors when initializing an LGP model.
    All necessary parameters should be given when initializing the LGP model.
    To enforce other wrapper to define their parameters, there are no default values for these parameters.
    '''
    def __init__(self, param_file:str, 
                 seed:int, 
                 setup_problem_script:bool,
                 output_file1:str,
                 output_file2:str):
        '''
        param_file: the .param file recording all parameters
        seed: the random seed of the GP run
        setup_problem_now: to indicate if the state.setup() function sets up the problem details based on parameter script or not.
        output_file1 & 2: the output files to record the evolution process
        '''
        # cwd = os.getcwd()
        # self.default_paramfile:str = f"{cwd}/tasks/Symbreg/parameters/LGP_regressor_Xy.params"
        
        # if not param_file:
        #     param_file = self.default_paramfile

        self.SETUP_PROBLEM_SCRIPT = "true" if setup_problem_script else "false"

        args = [
            "-file", 
            param_file,
            f"-p seed.0={seed}",
            f"-p setup_problem_script={self.SETUP_PROBLEM_SCRIPT}",
            f"-p stat.file={output_file1}" if output_file1 else "",
            f"-p stat.child.0.file={output_file2}" if output_file2 else ""
        ]

        self.output_ind:LGPIndividual = None # the output individual after training

        # running the main()
        job = 0
        
        # Load parameters
        self.parameters = Evolve.loadParameterDatabase(args)

        self.state:EvolutionState = Evolve.initialize(self.parameters, job)
        self.state.output.message(f"Job: {job}")
        self.state.job = [job]
        self.state.runtimeArguments = args
        self.state.setup_problem_script = setup_problem_script

        # self.state.output.message("Setting up by lgp_template")
        # self.state.setup()  # garbage Parameter equivalent

    def model(self)->LGPIndividual:
        if not self.output_ind:
            print("the linear genetic programming has not been trained. I found no output individual")
            sys.exit(1)
        
        return self.output_ind