# import classes and files
import sys
import os
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="An independent run for LGP solving classification problems.",
        add_help=False,
    )
    parser.add_argument('WORK_DIR', action='store', type=str, help='the working directory')
    parser.add_argument('-dir', action='store', dest='DIR', default=None, type=str, help='Dataset directory')
    parser.add_argument('-data', action='store', dest='DATA', default=None, type=str, help='name of the classification task')
    parser.add_argument('-seed', action='store', dest='SEED', default=None, type=int, help='A specific random seed')
    parser.add_argument('-results', action='store', dest='RDIR', default='results', type=str, help='Results directory')
    parser.add_argument('-log', action='store', dest='LOGDIR', default='output', type=str, help='log directory')
    parser.add_argument('-params', action='store', dest='PARAMS', default='LGP_classification.params', type=str, help='the parameter file for LGP solving classification')
    parser.add_argument('-run_index', action='store', default=0, type=int, help='index of independent runs')

    args = parser.parse_args()

    if args.WORK_DIR not in sys.path:
        sys.path.insert(0, args.WORK_DIR)

    from tasks.main_runner import MainRunner
    from tasks.lgp_template import LGP_Model_Template
    from tasks.classification.lgp_classifier import LinearGP_Classifier


    class ClassifierRunner(MainRunner):
        def fit_model(self) -> LGP_Model_Template:
            lgp = LinearGP_Classifier(
                param_file=self.param_file,
                seed=self.seed,
                setup_problem_script=False,
                output_file1=self.output_file1,
                output_file2=self.output_file2,
            )

            lgp.play_quiz(self.loca, self.datan)
            return lgp


    runner = ClassifierRunner()
    runner.preprarePaths(parser)

    lgp = runner.fit_model()

    runner.recordResult(lgp)
    runner.write2File()