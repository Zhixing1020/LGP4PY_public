
from src.ec.util.parameter_database import ParameterDatabase
from src.ec.util.parameter import Parameter
from src.ec.util.output import Output

from random import Random

class EvolutionState:

    # Run status codes
    R_SUCCESS: int = 0
    R_FAILURE: int = 1
    R_NOTDONE: int = 2

    UNDEFINED: int = -1

    # Parameter keys
    P_INITIALIZER: str = "init"
    P_FINISHER: str = "finish"
    P_BREEDER: str = "breed"
    P_EVALUATOR: str = "eval"
    P_POP: str = "pop"
    P_STATISTICS: str = "stat"
    P_EXCHANGER: str = "exch"
    P_GENERATIONS: str = "generations"
    P_NODEEVALUATIONS: str = "nodeevaluations"
    P_EVALUATIONS: str = "evaluations"
    P_QUITONRUNCOMPLETE: str = "quit-on-run-complete" 
    P_BUILDER: str = "gp.tc.0.init"  # only for the initializer in the first tree constraint
    P_PRIMSET: str = "gp.fs"
    P_SETUP_PROBLEM_NOW: str = "setup_problem_now" # set up the problem when we initialize the search space

    def __init__(self):
        from src.ec import Statistics
        from src.ec.breeder import Breeder

        # ParameterDatabase
        self.parameters:ParameterDatabase = None

        self.output = Output()

        self.random : list[Random] = [None] * 1

        self.breedthreads:int = 0
        self.evalthreads:int = 1
        self.randomSeedOffset:int = 0

        self.generation = self.__class__.UNDEFINED
        self.numGenerations = 200
        
        self.nodeEvaluation = self.__class__.UNDEFINED
        self.numNodeEva = 1e7
        self.numEvaluations = self.__class__.UNDEFINED

        self.population = None  # will be a Population instance
        self.evaluator = None  # Evaluator instance
        # self.initializer = None  # Initializer instance
        self.breeder:Breeder = None  # Breeder instance
        self.statistics:Statistics = None  # Statistics instance

        self.builder = None
        self.primitive_sets = None

        self.quitOnRunComplete = False
        self.setup_problem_script = True

        self.job: list[None] = None
        self.runtimeArguments: list[str] = None

        self.num_gen_trap_fit = 0
        self.previous_best_ind = None

    def setup(self, base:str=""):

        # p = Parameter(base)

        # self.data = [{} for _ in self.random]  # per-thread data

        # self.checkpoint = parameters.getBoolean("checkpoint", False)
        # self.checkpointPrefix = parameters.getString("checkpoint-prefix")

        # if self.checkpointPrefix is None:
        #     old_prefix = parameters.getString("prefix")
        #     if old_prefix is None:
        #         self.fatal("No checkpoint prefix specified.")
        #     else:
        #         self.output.warning("The parameter 'prefix' is deprecated. Please use 'checkpoint-prefix'.")
        #         self.checkpointPrefix = old_prefix
        # elif parameters.getString("prefix") is not None:
        #     self.output.warning("You have BOTH 'prefix' and 'checkpoint-prefix' defined. Using 'checkpoint-prefix'.")

        # self.checkpointModulo = parameters.getInt("checkpoint-modulo", 1)
        # if self.checkpointModulo <= 0:
        #     self.fatal("The checkpoint modulo must be an integer > 0.")

        # if parameters.exists("checkpoint-directory"):
        #     self.checkpointDirectory = parameters.getFile("checkpoint-directory")
        #     if self.checkpointDirectory is None:
        #         self.fatal("The checkpoint directory name is invalid.")
        #     if not self.checkpointDirectory.is_dir():
        #         self.fatal("The checkpoint directory is not a directory.")

        if self.parameters.exists(self.P_EVALUATIONS):
            self.numEvaluations = self.parameters.getInt(self.P_EVALUATIONS, None)
            if self.numEvaluations <= 0:
                self.output.fatal("Evaluations must be >= 1 if defined.")

        if self.parameters.exists(self.P_GENERATIONS):
            self.numGenerations = self.parameters.getInt(self.P_GENERATIONS, None)
            if self.numGenerations <= 0:
                self.output.fatal("Generations must be >= 1 if defined.")
            if self.numEvaluations != self.__class__.UNDEFINED:
                self.output.warning("Both generations and evaluations defined. Generations will be ignored.")
                self.numGenerations = self.__class__.UNDEFINED
        elif self.numEvaluations == self.__class__.UNDEFINED:
            self.output.fatal("Either evaluations or generations must be defined.")

        if self.parameters.exists(self.P_NODEEVALUATIONS):
            self.numNodeEva = self.parameters.getDoubleWithDefault(self.P_NODEEVALUATIONS, None, -1)
            if self.numNodeEva <= 0:
                self.output.fatal("Node evaluations must be >= 1 if defined.")

        self.quitOnRunComplete = self.parameters.getBoolean(self.P_QUITONRUNCOMPLETE, None, False)

        # self.initializer = self.parameters.getInstanceForParameter(self.P_INITIALIZER, None, Initializer)
        # self.initializer.setup(self, self.P_INITIALIZER)

        # self.finisher = self.parameters.getInstanceForParameter(self.P_FINISHER, None, Finisher)
        # self.finisher.setup(self, self.P_FINISHER)

        from src.ec.breeder import Breeder
        self.breeder = self.parameters.getInstanceForParameter(self.P_BREEDER, None, Breeder)
        self.breeder.setup(self, Parameter(self.P_BREEDER))

        from src.ec.evaluator import Evaluator
        # Ensure the Evaluator is set up correctly
        if not self.parameters.exists(self.P_EVALUATOR):
            self.output.fatal(f"Evaluator not defined in parameters: {self.P_EVALUATOR}")
        self.evaluator:Evaluator = self.parameters.getInstanceForParameter(self.P_EVALUATOR, None, Evaluator)
        self.evaluator.setup(self, Parameter(self.P_EVALUATOR))

        from src.ec import Statistics
        self.statistics = self.parameters.getInstanceForParameter(self.P_STATISTICS, None, Statistics)
        self.statistics.setup(self, Parameter(self.P_STATISTICS))

        # self.exchanger = self.parameters.getInstanceForParameter(self.P_EXCHANGER, None, Exchanger)
        # self.exchanger.setup(self, self.P_EXCHANGER)

        from src.ec.gp_builder import GPBuilder
        if self.parameters.exists(self.P_BUILDER):
            self.builder:GPBuilder = self.parameters.getInstanceForParameter(self.P_BUILDER, None, GPBuilder)
            self.builder.setup(self, Parameter(self.P_BUILDER))

        from src.ec.gp_primitive_set import GPPrimitiveSet
        # if self.parameters.exists(self.P_PRIMSET):
            # self.primitive_set = self.parameters.getInstanceForParameter(self.P_PRIMSET, None, GPPrimitiveSet)
        # self.primitive_set = GPPrimitiveSet()
        # self.primitive_set.setup(self, Parameter(self.P_PRIMSET))

        num_prim_set = self.parameters.getInt(Parameter(self.P_PRIMSET).push('size'), None)
        if num_prim_set < 1:
            self.output.fatal("there is less than one primitive set.", Parameter(self.P_PRIMSET).push('size'))
        self.primitive_sets = []
        for pi in range(0, num_prim_set):
            pbase = Parameter(self.P_PRIMSET).push(str(pi))
            ps = GPPrimitiveSet()
            ps.setup(self, pbase)
            self.primitive_sets.append(ps)

        self.generation = 0

        from src.ec.population import Population
        self.population = Population()
        self.population.setup(self, Parameter(self.P_POP))

    def finish(self, result: int):
        self.statistics.finalStatistics(self, result)
        # self.finisher.finishPopulation(self, result)
        # self.exchanger.closeContacts(self, result)
        # self.evaluator.closeContacts(self, result)

    def startFresh(self):
        self.output.message("Setting up a new run")
        self.setup()  # garbage Parameter equivalent
        self.num_gen_trap_fit = 0
        self.previous_best_ind = None

        # POPULATION INITIALIZATION
        self.output.message("Initializing Generation 0")
        self.statistics.preInitializationStatistics(self)
        # self.population = self.initializer.initialPopulation(self, 0)
        self.population.populate(self, 0)
        self.statistics.postInitializationStatistics(self)

        # Compute generations from evaluations if necessary
        if self.numEvaluations > self.UNDEFINED:
            generationSize = sum(
                len(subpop.individuals) for subpop in self.population.subpops
            )

            if self.numEvaluations < generationSize:
                self.numEvaluations = generationSize
                self.numGenerations = 1
                self.output.warning(f"Using evaluations, but evaluations is less than the initial total population size ({generationSize}).  Setting to the population size.")
            else:
                if self.numEvaluations % generationSize != 0:
                    new_evals = (self.numEvaluations // generationSize) * generationSize
                    self.output.warning(f"Using evaluations, but initial total population size does not divide evenly into it.  Modifying evaluations to a smaller value ({new_evals}) which divides evenly.")
                self.numGenerations = self.numEvaluations // generationSize
                self.numEvaluations = self.numGenerations * generationSize

            self.output.message(f"Generations will be {self.numGenerations}")

        # self.exchanger.initializeContacts(self)
        # self.evaluator.initializeContacts(self)

    def evolve(self):
        if self.generation > 0:
            self.output.message(f"Generation {self.generation}")

        # EVALUATION
        self.statistics.preEvaluationStatistics(self)
        self.evaluator.evaluatePopulation(self)
        self.statistics.postEvaluationStatistics(self)

        # record best fitness
        tmp_best_of_run = None
        for x in range(len(self.population.subpops)):
            for y in range(0, len(self.population.subpops[x].individuals)):
                individual = self.population.subpops[x].individuals[y]
                            
                # Update best_of_run if better individual found
                if (tmp_best_of_run is None or 
                    individual.fitness.betterThan(tmp_best_of_run.fitness)):
                    tmp_best_of_run = individual.clone()
        if self.previous_best_ind == None or tmp_best_of_run.fitness.betterThan(self.previous_best_ind.fitness):
            self.previous_best_ind = tmp_best_of_run
            self.num_gen_trap_fit = 0
        else:
            self.num_gen_trap_fit = self.num_gen_trap_fit + 1

        # SHOULD WE QUIT?
        if self.evaluator.runComplete(self) and self.quitOnRunComplete:
            self.output.message("Found Ideal Individual")
            return self.R_SUCCESS

        if self.generation == self.numGenerations - 1:
            return self.R_FAILURE

        # PRE-BREEDING EXCHANGING
        # self.statistics.prePreBreedingExchangeStatistics(self)
        # self.population = self.exchanger.preBreedingExchangePopulation(self)
        # self.statistics.postPreBreedingExchangeStatistics(self)

        # exchanger_msg = self.exchanger.runComplete(self)
        # if exchanger_msg is not None:
        #     self.output.message(exchanger_msg)
        #     return self.R_SUCCESS

        # BREEDING
        self.statistics.preBreedingStatistics(self)
        self.population = self.breeder.breedPopulation(self)
        self.statistics.postBreedingStatistics(self)

        # POST-BREEDING EXCHANGING
        # self.statistics.prePostBreedingExchangeStatistics(self)
        # self.population = self.exchanger.postBreedingExchangePopulation(self)
        # self.statistics.postPostBreedingExchangeStatistics(self)

        # INCREMENT GENERATION AND CHECKPOINT
        self.generation += 1
        # if self.checkpoint and self.generation % self.checkpointModulo == 0:
        #     self.output.message("Checkpointing")
        #     self.statistics.preCheckpointStatistics(self)
        #     Checkpoint.setCheckpoint(self)
        #     self.statistics.postCheckpointStatistics(self)

        return self.R_NOTDONE

    def run(self):
        
        # self.startFresh() # in python implementation, I separate startFresh() with run() to avoid
        # duplicated setup

        result = self.R_NOTDONE
        while result == self.R_NOTDONE:
            result = self.evolve()

        self.finish(result)