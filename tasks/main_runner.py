
#This file defines the commonly used functions for the main function in running LGP

import sys
import os
import argparse
import json
from abc import ABC, abstractmethod
from src.ec.util.parameter_database import ParameterDatabase
from tasks.utils import jsonify
from tasks.lgp_template import LGP_Model_Template

class MainRunner(ABC):
    """
    MainRunner is a template for implementing the main entrance of a certain problem.
    """

    def preprarePaths(self, parser):
        """
        set the necessary file paths
        """
        self.args = parser.parse_args()

        self.loca = self.args.DIR
        self.datan = self.args.DATA
        self.seed = self.args.SEED
        self.results_path = self.args.RDIR #TODO  check the proper setting of the output directory
        self.param_file = self.args.PARAMS
        self.run_index = self.args.run_index
        self.output_file1 = os.sep.join([self.args.LOGDIR, f"job.{self.run_index}.out.stat"])
        self.output_file2 = os.sep.join([self.args.LOGDIR, f"job.{self.run_index}.outtabular.stat"])

        print(f"playing the task {self.datan} with a seed {self.seed} and the param file {self.param_file}")
        print(f"results will be logged to {self.results_path}")
        print(f"output files: {self.output_file1}, {self.output_file2}")

    @abstractmethod
    def fit_model(self)->LGP_Model_Template:
        """training and test of a model"""
        pass

    def recordResult(self, lgp_model:LGP_Model_Template):
        """collect the result from the current run"""
        self.results = {
            'dataset':self.datan,
            'params':jsonify(lgp_model.parameters.params),
            'random_seed':self.seed,
            'time_time': lgp_model.train_time, 
        }

        output_program:str = lgp_model.output_ind.printTrees(None)
        print(f"the output program: {output_program}")

        self.results['output_program'] = output_program
        self.results['train_fitness'] = lgp_model.train_fitness

        self.results['test_fitness'] = lgp_model.test_fitness

    def write2File(self):
        """write the result to a file"""
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        save_file = os.path.join(
            self.results_path,
            '_'.join([self.datan, str(self.seed)])
        )

        print('save_file:',save_file, '.json')

        with open(save_file + '.json', 'w') as out:
            json.dump(jsonify(self.results), out, indent=4)

        print(f"finish {self.datan} run {self.run_index}. Results are logged to {self.results_path}")

    
