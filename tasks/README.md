
This package defines the problem-specific algorithm implementation.

### Project Structure ###

* We have implemented the following tasks:
  - `ARC`: using LGP to solve Abstract and Reasoning Corpus (ARC)
  - `classification`: using LGP to solve [multi-class classification problems](./classification/).
  - `symbreg` using LGP to solve [symbolic regression problems](./symbreg/).

* `batch_runner.py`: The main idea is to create a series of commands and submit them to a server. It has a main function. Here are two running examples for `batch_runner.py`:
  - **Example 1: Running a single ARC task (007bbfb7) twice locally via `launch.json`, each run with different random seeds (7 & 8)**
    1. Locate the `workspaceFolder`.
    2. Locate the main Python file (e.g., [`lgp_main.py`](./ARC/lgp_main.py) in this example)
    3. Locate the `datasetDirectory`.
    4. Locate the `resultDirectory`.
    5. Locate the [parameterFile](./ARC/parameters/LGP_arc_try.params)
    6. The "configurations" in `launch.json` can be:
       ```
       {
            "name": "run batch lgp4arc",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/tasks/batch_runner.py",
            "console": "integratedTerminal",
            "noDebug": true,
            "args": [
                "${workspaceFolder}",
                "-dir",
                "[datasetDirectory]\\",
                "--local",
                "-n_jobs",
                "1",
                "-data",
                "007bbfb7",
                "-n_trials",
                "2",
                "-script",
                "${workspaceFolder}/tasks/ARC/lgp_main.py",
                "-seed",
                "7",
                "-results",
                "[resultDirectory]/",
                "-params", 
                "[parameterFile]",
            ],
            "env": {
                "PYTHONHASHSEED": "1000117"
            }
        }
       ```
   
  - **Example 2: Running all ARC tasks, each with 5 independent runs, on a Slurm server**
    1. Locate the `workspaceFolder`.
    2. Locate the main Python file (e.g., [`lgp_main.py`](./ARC/lgp_main.py) in this example)
    3. Locate the `datasetDirectory`.
    4. Locate the `resultDirectory`.
    5. Locate the [parameterFile](./ARC/parameters/LGP_arc_try.params)
    6. The command on the server can be: ```python3 batch_runner.py [workspaceFolder] -dir [datasetDirectory] --slurm -n_trials 5 -script [workspaceFolder]/tasks/ARC/lgp_main.py -seed 7 -results [resultDirectory] -params [parameterFile] -time_limit 5:00  -q parallel  -job_limit 3000```. The folder, directory, and file should be in an absolute path on the server. For more details, please refer to `batch_runner.py`.

* `lgp_template.py` defines the template of an LGP model. The LGP model is trained to produce an output individual.

* `problem.py` defines an abstract Problem class.

* `supervisedproblem.py` defines an abstract supervised problem class, working with the Problem class to complete the necessary abstract methods of a problem.

* `utils.py` defines utilization functions.


### How to apply LGP to your own applications ###

For example, you have an application of "Application1". Then you can extend a basic LGP to this application by following these steps:

1. Set up an independent folder "Application1" in the `tasks` folder.
2. Set up folders of `individual`, `optimization`, `parameters`. Other folders like `ec`, `ruleanalysis`, `algorithm`, `DSL` and `util`, are required if you want to have more application-specific designs in LGP.
3. Build up `lgp_application1.py` in the folder of `optimization`. `lgp_application1.py` defines the necessary members and methods for this application.
4. Build up `lgpindividual4App1.py` in the folder of `individual`. `lgpindividual4App1.py` defines the LGP individual for solving the "Application1". If you have specific GP primitives for solving this application, you would set up a sub-folder `primitives` in `individual` and define the new primitive classes in the `primitives` sub-folder.
5. Build up `lgp_application1_solver.py` in the folder of `Application1`. This .py file defines the interface of an LGP model, including kicking off the training process, testing the output individual, etc.
6. Build up `lgp4App1_main.py` in the folder of `Application1`. This .py file defines the main entrance of this application. The `lgp4App1_main.py` can be used in `batch_runner.py`.
7. Write a parameter file `LGP_application1_test.params` in the folder of `paramters`.  This .params file defines all the necessary parameter settings for LGP and the problem itself.
