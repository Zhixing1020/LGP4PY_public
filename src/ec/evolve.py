import sys
import os
import time
import random
from typing import List, Optional, Dict, Any, Tuple
from multiprocessing import cpu_count
from src.ec import *
from src.ec.util import *

class Evolve:
    """Python implementation maintaining original ECJ names"""
    
    # The argument indicating that we're starting fresh from a new parameter file.
    A_FILE = "-file"
    
    # The argument indicating that we're starting fresh from a parameter file stored in a jar file or as some resource.
    A_FROM = "-from"

    # The argument indicating the class where the resource is relative to.
    A_AT = "-at"

    # The argument indicating a request to print out the help message.
    A_HELP = "-help"

    # The argument indicating we're restoring from checkpoint
    A_CHECKPOINT = "-checkpoint"

    # evalthreads parameter
    P_EVALTHREADS = "evalthreads"

    # breedthreads parameter
    P_BREEDTHREADS = "breedthreads"

    # seed parameter
    P_SEED = "seed"

    # 'time' seed parameter value
    V_SEED_TIME = "time"

    # state parameter
    P_STATE = "state"
    
    # 'auto' thread parameter value
    V_THREADS_AUTO = "auto"
    
    # Should we muzzle stdout and stderr?
    P_SILENT = "silent"

    # Should we muzzle stdout and stderr? [deprecated]
    P_MUZZLE = "muzzle"

    @staticmethod
    def checkForHelp(args: List[str]) -> None:
        """Optionally prints the help message."""
        for arg in args:
            if arg == Evolve.A_HELP:
                print("""
Evolutionary Computation Framework

Format:

    python evolve.py -file FILE [-p PARAM=VALUE] [-p PARAM=VALUE] ...
    python evolve.py -from FILE [-p PARAM=VALUE] [-p PARAM=VALUE] ... 
    python evolve.py -from FILE -at CLASS [-p PARAM=VALUE] [-p PARAM=VALUE] ...
    python evolve.py -checkpoint CHECKPOINT
    python evolve.py -help

-help                   Shows this message and exits.

-file FILE              Launches using the provided parameter FILE.

-from FILE              Launches using the provided parameter FILE
                        which is defined relative to the directory
                        holding the evolve.py file.

-from FILE -at CLASS    Launches using the provided parameter FILE
                        which is defined relative to the directory
                        holding the CLASS file.

-p PARAM=VALUE          Overrides the parameter PARAM in the parameter
                        file, setting it to VALUE instead.

-checkpoint CHECKPOINT  Launches from the provided CHECKPOINT file.
                """)
                sys.exit(1)

    # @staticmethod
    # def possiblyRestoreFromCheckpoint(args: List[str]) -> Optional[EvolutionState]:
    #     """Restores an EvolutionState from checkpoint if "-checkpoint FILENAME" is in the command-line arguments."""
    #     for i in range(len(args)-1):
    #         if args[i] == Evolve.A_CHECKPOINT:
    #             checkpointFile = args[i+1]
    #             print(f"Restoring from Checkpoint {checkpointFile}")
    #             try:
    #                 # TODO: Implement actual checkpoint restore
    #                 state = Evolve.EvolutionState()
    #                 return state
    #             except Exception as e:
    #                 print(f"An exception was generated upon starting up from a checkpoint.\nFor help, try: python evolve.py -help\n\n{e}")
    #                 sys.exit(1)
    #     return None

    @staticmethod
    def loadParameterDatabase(args: List[str]) -> ParameterDatabase:
        """Loads a ParameterDatabase from checkpoint if "-params" is in the command-line arguments."""
        # parameters = {}
        
        # Simplified parameter loading - would need actual implementation
        if Evolve.A_FILE in args:
            idx = args.index(Evolve.A_FILE)
            if idx + 1 < len(args):
                paramFile = args[idx+1]
                parameters = ParameterDatabase(paramFile)
                # try:
                #     with open(paramFile) as f:
                #         for line in f:
                #             if '=' in line:
                #                 key, val = line.replace(" ", "").split('=', 1)
                #                 parameters[key] = val
                # except Exception as e:
                #     print(f"An exception was generated upon reading the parameter file \"{paramFile}\".\nHere it is:\n{e}")
                #     sys.exit(1)
        
        # Handle parameter overrides (-p)
        for arg in args:
            if arg.startswith("-p") and '=' in arg:
                param, value = arg[3:].split('=', 1)
                parameters.params[param] = value
        
        if not parameters:
            print("No parameter or checkpoint file was specified.\nFor help, try: python evolve.py -help")
            sys.exit(1)
            
        return parameters

    @staticmethod
    def determineThreads(output: Output, parameters: ParameterDatabase, threadParameter: str) -> int:
        """Loads the number of threads."""
        tmp_s = parameters.getString(Parameter(threadParameter))
        if tmp_s is None:
            output.fatal("Threads number must exist.", threadParameter, None)
        elif tmp_s.lower() == Evolve.V_THREADS_AUTO:
            return cpu_count()
        else:
            try:
                return int(tmp_s)
            except ValueError:
                output.fatal(f"Invalid, non-integer threads value ({tmp_s})", threadParameter, None)
        return 1  # unreachable

    @staticmethod
    def primeGenerator(generator: random.Random) -> random.Random:
        """Primes the generator by calling random() several times."""
        for _ in range(624 * 2 + 1):  # Same as Java version
            generator.random()
        return generator

    @staticmethod
    def determineSeed(output: Output, parameters: ParameterDatabase, seedParameter: str, 
                     currentTime: int, offset: int, auto: bool) -> int:
        """Loads a random generator seed."""
        tmp_s = parameters.getString(Parameter(seedParameter))
        if tmp_s is None and not auto:
            output.fatal("Seed must exist.", seedParameter, None)
        elif tmp_s == Evolve.V_SEED_TIME or (tmp_s is None and auto):
            if tmp_s is None and auto:
                output.warning("Using automatic determination number of threads, but not all seeds are defined.\nThe rest will be defined using the wall clock time.")
            seed = currentTime
            if seed == 0:
                output.fatal(f"Whoa! This Python version is returning 0 for time.time(), which ain't right. This means you can't use '{Evolve.V_SEED_TIME}' as a seed", seedParameter, None)
        else:
            try:
                seed = int(tmp_s)
            except ValueError:
                output.fatal(f"Invalid, non-integer seed value ({tmp_s})", seedParameter, None)
        return seed + offset

    # @staticmethod
    def buildOutput() -> Output:
        """Constructs and sets up an Output object."""
        output = Output()
        return output

    # @staticmethod
    def initialize(parameters: ParameterDatabase, randomSeedOffset: int) -> EvolutionState:
        """Initializes an evolutionary run."""
        return Evolve.initializeWithOutput(parameters, randomSeedOffset, Evolve.buildOutput())

    @staticmethod
    def initializeWithOutput(parameters: ParameterDatabase, randomSeedOffset: int, output: Output) -> EvolutionState:
        """Initializes with pre-constructed Output."""
        state:EvolutionState = parameters.getInstanceForParameter(Parameter(Evolve.P_STATE), None, EvolutionState)
        seeds = []
        
        # Handle silent/muzzle
        if parameters.exists(Evolve.P_MUZZLE) and parameters.getString(Evolve.P_MUZZLE, "").lower() == "true":
            output.warning(f"{Evolve.P_MUZZLE} has been deprecated. We suggest you use {Evolve.P_SILENT} or similar newer options.")
        
        if (parameters.exists(Evolve.P_SILENT) and parameters.getString(Evolve.P_SILENT, "").lower() == "true" 
            or
            parameters.exists(Evolve.P_MUZZLE) and parameters.getString(Evolve.P_MUZZLE, "").lower() == "true"):
            output.silent = True

        output.message("Linear Genetic Programming on Evolutionary Computation Framework (Python)")
                
        # Determine threads
        breedthreads = Evolve.determineThreads(output, parameters, Evolve.P_BREEDTHREADS)
        evalthreads = Evolve.determineThreads(output, parameters, Evolve.P_EVALTHREADS)
        auto = (parameters.getString(Evolve.P_BREEDTHREADS, "").lower() == Evolve.V_THREADS_AUTO or
               parameters.getString(Evolve.P_EVALTHREADS, "").lower() == Evolve.V_THREADS_AUTO)

        # Create random number generators
        numRNGs = max(breedthreads, evalthreads)
        rngs = []
        
        currentTime = int(time.time() * 1000)
        
        for x in range(numRNGs):
            seed = Evolve.determineSeed(output, parameters, f"{Evolve.P_SEED}.{x}",
                                       currentTime + x, numRNGs * randomSeedOffset, auto)
            # Check for duplicate seeds
            for y in range(x):
                if seed == seeds[y]:
                    output.fatal(f"{Evolve.P_SEED}.{x} ({seed}) and {Evolve.P_SEED}.{y} ({seeds[y]}) ought not be the same seed.", None, None)
            
            seeds.append(seed)
            rng = random.Random(seed)
            rng = Evolve.primeGenerator(rng)
            rngs.append(rng)

        # Setup evolution state
        state.parameters = parameters
        state.random = rngs
        state.output = output
        state.evalthreads = evalthreads
        state.breedthreads = breedthreads
        state.randomSeedOffset = randomSeedOffset

        output.message(f"Threads: breed/{breedthreads} eval/{evalthreads}")
        output.message("Seed: " + " ".join(map(str, seeds)))
                
        return state

    # @staticmethod
    # def cleanup(state: EvolutionState) -> None:
    #     """Cleanup resources"""
    #     pass  # Would implement actual cleanup

    @staticmethod
    def main() -> None:
        """Top-level evolutionary loop."""
        args = sys.argv[1:]
        
        # Check for help
        Evolve.checkForHelp(args)
                
        # Check for checkpoint restore
        # state = Evolve.possiblyRestoreFromCheckpoint(args)
        currentJob = 0

        # if state is not None:
        #     # Restored from checkpoint
        #     try:
        #         if state.runtimeArguments is None:
        #             print("Checkpoint completed from job started by foreign program. Exiting...")
        #             sys.exit(1)
        #         args = state.runtimeArguments
        #         currentJob = state.job[0] + 1
        #         state.run(0)  # C_STARTED_FROM_CHECKPOINT
        #         Evolve.cleanup(state)
        #     except Exception as e:
        #         print("EvolutionState's jobs variable is not set up properly. Exiting...")
        #         sys.exit(1)

        # Load parameters
        parameters = Evolve.loadParameterDatabase(args)
        if currentJob == 0:
            currentJob = parameters.getIntWithDefault("current-job", None, 0)
        if currentJob < 0:
            print("The 'current-job' parameter must be >= 0 (or not exist, which defaults to 0)")
            sys.exit(1)
            
        numJobs = int(parameters.getParamValue("jobs", 1))
        if numJobs < 1:
            print("The 'jobs' parameter must be >= 1 (or not exist, which defaults to 1)")
            sys.exit(1)
                
        # Run jobs
        for job in range(currentJob, numJobs):
            # try:
            if job > currentJob:
                parameters = Evolve.loadParameterDatabase(args)
                        
            state:EvolutionState = Evolve.initialize(parameters, job)
            state.output.message(f"Job: {job}")
            state.job = [job]
            state.runtimeArguments = args
            
            # if numJobs > 1:
            #     jobFilePrefix = f"job.{job}."
            #     state.checkpointPrefix = jobFilePrefix + state.checkpointPrefix
                                
            state.run()  # C_STARTED_FRESH
            # Evolve.cleanup(state)
            parameters = None  # Force reload next time
            # except Exception as e:
            #     print(f"Error in job {job}: {e}")
                # Continue to next job

if __name__ == "__main__":
    Evolve.main()