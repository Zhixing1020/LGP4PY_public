

import sys
import os
import argparse
import glob
import subprocess
from joblib import Parallel, delayed
import shutil

class BatchRunner():

    def __init__(self):
        self.all_commands = []
        self.job_info=[]
        self.args = None

    def clear_folder(self, folder_path):
        '''clear all files and subfolder in a folder'''
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)       # delete file or symlink
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)   # delete subfolder
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

    def collect_parameters(self):
        '''collect settings from the command line'''
        parser = argparse.ArgumentParser(
            description="Batch runner is launching.", add_help=False)
        parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
        parser.add_argument('WORK_DIR', action='store',
                type=str, help='the working directory')
        parser.add_argument('-dir', action='store', dest='DIR',  default=None,
                type=str, help='Dataset directory like (pmlb/datasets)')
        parser.add_argument('-data', action='store', dest='DATA',default=None,
                type=str, help='name of the ARC task')
        parser.add_argument('--local', action='store_true', dest='LOCAL', default=False, 
                help='Run locally as opposed to on LPC')
        parser.add_argument('--local_parallel', action='store_true', dest='LO_PARALLEL', default=False, 
            help='Run locally and parallelly')
        parser.add_argument('--slurm', action='store_true', dest='SLURM', default=False, 
                help='Run on a SLURM scheduler as opposed to on LPC')
        parser.add_argument('-n_jobs',action='store',dest='N_JOBS',default=1,type=int,
                help='Number of parallel jobs')
        parser.add_argument('-time_limit',action='store',dest='TIME',default='48:00',
                type=str, help='Time limit (hr:min) e.g. 24:00')
        parser.add_argument('-seed',action='store',dest='SEED',default=7,
                type=int, help='A specific random seed')
        parser.add_argument('-n_trials',action='store',dest='N_TRIALS',default=1,
                type=int, help='Number of parallel jobs for each task')
        parser.add_argument('-results',action='store',dest='RDIR',default='results',
                type=str,help='Results directory')
        parser.add_argument('-script',action='store',dest='SCRIPT',
                            default='lgp_main.py',
                            type=str,help='Python script to run')
        parser.add_argument('-job_limit',action='store',dest='JOB_LIMIT',
                            default=1000, type=int, 
                            help='Limit number of jobs submitted at once')
        parser.add_argument('-params',action='store',dest='PARAMS',
                            default='LGP_arc_try.params',
                            type=str,help='the paramater file for ARC')
        parser.add_argument('-A', action='store', dest='A', default='plgsrbench', 
                help='SLURM account')
        # parser.add_argument('-clear_data', action='store_true', default=False, 
        #         help='clear the training and test data for ARC')
        parser.add_argument('-m',action='store',dest='M',default=16384,type=int,
                help='LSF memory request and limit (MB)')
        parser.add_argument('-q',action='store',dest='QUEUE',
                            default='epistasis_long',
                            type=str,help='LSF queue')
        
        self.args = parser.parse_args()

        # Insert at the front of sys.path (highest priority)
        if self.args.WORK_DIR not in sys.path:
            sys.path.insert(0, self.args.WORK_DIR)

    def validate_parameters(self):
        '''validate the values of the parameters'''
        if (self.args.DIR == None or not os.path.exists(self.args.DIR)) and self.args.DATA == None:
            raise Exception("both data directory and data name are not given or do not exist")

        print('dataset directory:', self.args.DIR)

    def build_commands(self):
        '''build all the commands for a batch of runs'''
        if self.args.DATA == None:
            tasknames = [ os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{self.args.DIR}*.json")]
        else:
            tasknames = [self.args.DATA]

        # for each task, run with different random seeds, record the best
        for dataname in tasknames:
            for t in range(self.args.SEED, self.args.SEED+self.args.N_TRIALS):
                
                results_path = os.path.abspath( os.sep.join([self.args.RDIR, dataname])) + os.sep
                self.clear_folder(results_path)
                if not os.path.exists(results_path):
                    os.makedirs(results_path)

                # save_file = (results_path + '/' + dataname + '_' 
                #                 + str(t))
                log_path = os.path.abspath( os.sep.join([results_path, 'log'])) + os.sep
                self.clear_folder(log_path)                
                if not os.path.exists(log_path):
                    os.mkdir(log_path)
                

                self.all_commands.append(
                    '{PYTHON_CMD} {SCRIPT} {WORK_DIR}'
                                        ' -dir {DATASET}'
                                        ' -data {TASKNAME}'
                                        ' -seed {RS} '
                                        ' -results {RDIR}'
                                        ' -log {LOGDIR}'
                                        ' -run_index {RUN_INDEX} '
                                        ' -params {PARAM_FILE} '.format(
                                            PYTHON_CMD = "py" if os.name == "nt" else "python3",
                                            SCRIPT=self.args.SCRIPT if self.args.SCRIPT.endswith('.py') else self.args.SCRIPT+".py",
                                            WORK_DIR=self.args.WORK_DIR,
                                            DATASET=self.args.DIR,
                                            TASKNAME=dataname,
                                            RDIR=results_path,
                                            LOGDIR=log_path,
                                            RS=t,
                                            RUN_INDEX=t-self.args.SEED,
                                            PARAM_FILE=self.args.PARAMS,
                                            )
                                        
                )
                self.job_info.append({'dataset':dataname,
                                    'seed':str(t),
                                    'results_path':results_path,
                                    'log_path':log_path
                                    })
                
        if len(self.all_commands) > self.args.JOB_LIMIT:
            print('shaving jobs down to job limit ({})'.format(self.args.JOB_LIMIT))
            self.all_commands = self.all_commands[:self.args.JOB_LIMIT]

    def run(self):
        '''run a batch of commands'''
        if self.args.LOCAL:
        # run locally  
        
            for run_cmd in self.all_commands:
                print(run_cmd)
                subprocess.run(run_cmd)
        elif self.args.LO_PARALLEL:
            results = Parallel(n_jobs=self.args.N_JOBS)(delayed(subprocess.run)(run_cmd, shell=True, capture_output=True, text=True, )
                                    for run_cmd in self.all_commands)
            for i, r in enumerate(results):
                print(f"run {i} return code:{r.returncode}")
                print(f"run {i} stdout:\n{r.stdout}")
                print(f"run {i} stderr:\n{r.stderr}")
        else:
            # sbatch
            for i,run_cmd in enumerate(self.all_commands):
                job_name = '_'.join([
                                    self.job_info[i]['dataset'],
                                    self.job_info[i]['seed'],
                                    os.path.splitext(os.path.basename(self.args.SCRIPT))[0]
                                    ])
                
                # slurm_output_path = os.path.join(self.args.RDIR, 'rap_output', "")
                slurm_output_path = os.path.join(self.job_info[i]['results_path'], 'output', "")
                if os.path.exists(slurm_output_path):
                    # shutil.rmtree(log_path)
                    for filename in os.listdir(slurm_output_path):
                        file_path = os.path.join(slurm_output_path, filename)
                        try:
                            os.unlink(file_path)
                        except Exception as e:
                            print(f'Failed to delete {file_path}. Reason: {e}')
                
                if not os.path.exists(slurm_output_path):
                    os.mkdir(slurm_output_path)

                out_file = os.path.join(slurm_output_path, '%x-%A-%a.o.txt')
                error_file = out_file[:-5] + 'e.txt'

                if self.args.SLURM:
                    batch_script = \
"""#!/bin/bash 
#SBATCH -N 1 
#SBATCH -n {N_CORES} 
#SBATCH -J {JOB_NAME} 
#SBATCH --partition={QUEUE} 
#SBATCH --cpus-per-task=1
#SBATCH --time={TIME}:00 
#SBATCH --mem={M} 
#SBATCH -o {OUT_FILE} 
#SBATCH -e {ERR_FILE} 

module load Python/3.11.5

{cmd}
    """.format(
            OUT_FILE=out_file,
            ERR_FILE=error_file,
            JOB_NAME=job_name,
            QUEUE=self.args.QUEUE,
            A=self.args.A,
            N_CORES=self.args.N_JOBS,
            M=self.args.M,
            cmd=run_cmd,
            TIME=self.args.TIME
            )
                    with open('tmp_script','w') as f:
                        f.write(batch_script)

                    # print(batch_script)
                    print(job_name)
                    sbatch_response = subprocess.check_output(['sbatch tmp_script'],
                                                                shell=True).decode()     # submit jobs 
                    print(sbatch_response)

                else: # LPC
                    # activate srbench env, load modules
                    # pre_run_cmds = ["conda activate srbench",
                    #                 "source lpc_modules.sh"]
                    # run_cmd = '; '.join(pre_run_cmds + [run_cmd])
                    bsub_cmd = ('bsub -o {OUT_FILE} '
                                '-e {ERR_FILE} '
                                '-n {N_CORES} '
                                '-J {JOB_NAME} '
                                '-q {QUEUE} '
                                '-R "span[hosts=1] rusage[mem={M}]" '
                                '-W {TIME} '
                                '-M {M} ').format(
                                        OUT_FILE=out_file,
                                        ERR_FILE=error_file,
                                        JOB_NAME=job_name,
                                        QUEUE=self.args.QUEUE,
                                        N_CORES=self.args.N_JOBS,
                                        M=self.args.M,
                                        TIME=self.args.TIME
                                        )
                    
                    bsub_cmd +=  '"' + run_cmd + '"'
                    print(bsub_cmd)
                    os.system(bsub_cmd)     # submit jobs 
            
        print('Finished submitting',len(self.all_commands),'jobs.')

if __name__ == "__main__":
    batch_runner = BatchRunner()
    batch_runner.collect_parameters()
    batch_runner.build_commands()
    batch_runner.run()