
from src.ec.util.parameter_database import ParameterDatabase
from src.ec.util.parameter import Parameter
# from src.ec.util.output import Output
from src.ec.evolution_state import EvolutionState
from src.lgp.algorithm.typed_lgp.dsl.agent import Agent
from typing import override


class TypedEvolutionState(EvolutionState):

    P_TYPE_AGENT = "typeagent"

    def __init__(self):
        super().__init__()
        self.typeagent:Agent = None

    @override
    def setup(self, base:str=""):

        if self.parameters.exists(self.P_TYPE_AGENT):
            self.typeagent = self.parameters.getInstanceForParameter(self.P_TYPE_AGENT, None, Agent)
            self.typeagent.setup(self, Parameter(self.P_TYPE_AGENT) )
        else:
            self.output.fatal(f"type agent not defined in parameters: {self.P_TYPE_AGENT}")

        super().setup(base)

    @override
    def lightClone_w_pickable(self):
        """Create a shallow picklable clone of this TypedEvolutionState.

        Extends the parent's lightClone_w_pickable() to handle the non-picklable
        typeagent which contains module variables and other unpicklable objects.
        Uses Agent.lightClone() to preserve computation data while removing module references.
        """
        new_state = super().lightClone_w_pickable()

        
        # Light-clone the typeagent to make it picklable
        # Preserves all function/type/feature data but removes module objects
        if self.typeagent is not None:
            new_state.typeagent = self.typeagent.lightClone_w_pickable()        

        # import pickle

        # def _get_all_attrs(obj):
        #     attrs = {}

        #     # __dict__
        #     if hasattr(obj, "__dict__"):
        #         attrs.update(vars(obj))

        #     # __slots__
        #     for cls in obj.__class__.__mro__:
        #         if hasattr(cls, "__slots__"):
        #             slots = cls.__slots__
        #             if isinstance(slots, str):
        #                 slots = [slots]
        #             for slot in slots:
        #                 if hasattr(obj, slot):
        #                     attrs[slot] = getattr(obj, slot)

        #     return attrs

        # def _find_unpicklable_attrs(obj):
        #     bad = {}
            
        #     # if items is None:
        #     try:
        #         pickle.dumps(obj)
        #     except Exception as e:
        #         bad[repr(obj)] = (type(obj), repr(e))   
            
        #     items = _get_all_attrs(obj)
        #     if items is not None:            
        #         for _, v in items.items():
        #             # try:
        #             #     pickle.dumps(v)
        #             # except Exception as e:
        #             #     bad[k] = (type(v), repr(e))
        #             bad.update(_find_unpicklable_attrs(v))

        #     return bad
        
        # bad = _find_unpicklable_attrs(new_state)

        # for k, (t, err) in bad.items():
        #     print(f"{k}: {t} -> {err}")

        return new_state
        
        # if self.parameters.exists(self.P_EVALUATIONS):
        #     self.numEvaluations = self.parameters.getInt(self.P_EVALUATIONS, None)
        #     if self.numEvaluations <= 0:
        #         self.output.fatal("Evaluations must be >= 1 if defined.")

        # if self.parameters.exists(self.P_GENERATIONS):
        #     self.numGenerations = self.parameters.getInt(self.P_GENERATIONS, None)
        #     if self.numGenerations <= 0:
        #         self.output.fatal("Generations must be >= 1 if defined.")
        #     if self.numEvaluations != self.__class__.UNDEFINED:
        #         self.output.warning("Both generations and evaluations defined. Generations will be ignored.")
        #         self.numGenerations = self.__class__.UNDEFINED
        # elif self.numEvaluations == self.__class__.UNDEFINED:
        #     self.output.fatal("Either evaluations or generations must be defined.")

        # if self.parameters.exists(self.P_NODEEVALUATIONS):
        #     self.numNodeEva = self.parameters.getDoubleWithDefault(self.P_NODEEVALUATIONS, None, -1)
        #     if self.numNodeEva <= 0:
        #         self.output.fatal("Node evaluations must be >= 1 if defined.")

        # self.quitOnRunComplete = self.parameters.getBoolean(self.P_QUITONRUNCOMPLETE, None, False)

        # self.initializer = self.parameters.getInstanceForParameter(self.P_INITIALIZER, None, Initializer)
        # self.initializer.setup(self, self.P_INITIALIZER)

        # self.finisher = self.parameters.getInstanceForParameter(self.P_FINISHER, None, Finisher)
        # self.finisher.setup(self, self.P_FINISHER)

        # from src.ec.breeder import Breeder
        # self.breeder = self.parameters.getInstanceForParameter(self.P_BREEDER, None, Breeder)
        # self.breeder.setup(self, Parameter(self.P_BREEDER))

        # from src.ec.evaluator import Evaluator
        # # Ensure the Evaluator is set up correctly
        # if not self.parameters.exists(self.P_EVALUATOR):
        #     self.output.fatal(f"Evaluator not defined in parameters: {self.P_EVALUATOR}")
        # self.evaluator:Evaluator = self.parameters.getInstanceForParameter(self.P_EVALUATOR, None, Evaluator)
        # self.evaluator.setup(self, Parameter(self.P_EVALUATOR))

        # from src.ec import Statistics
        # self.statistics = self.parameters.getInstanceForParameter(self.P_STATISTICS, None, Statistics)
        # self.statistics.setup(self, Parameter(self.P_STATISTICS))

        # # self.exchanger = self.parameters.getInstanceForParameter(self.P_EXCHANGER, None, Exchanger)
        # # self.exchanger.setup(self, self.P_EXCHANGER)

        # from src.ec.gp_builder import GPBuilder
        # if self.parameters.exists(self.P_BUILDER):
        #     self.builder:GPBuilder = self.parameters.getInstanceForParameter(self.P_BUILDER, None, GPBuilder)
        #     self.builder.setup(self, Parameter(self.P_BUILDER))

        # from src.ec.gp_primitive_set import GPPrimitiveSet
        # num_prim_set = self.parameters.getInt(Parameter(self.P_PRIMSET).push('size'), None)
        # if num_prim_set < 1:
        #     self.output.fatal("there is less than one primitive set.", Parameter(self.P_PRIMSET).push('size'))
        # self.primitive_sets = []
        # for pi in range(0, num_prim_set):
        #     pbase = Parameter(self.P_PRIMSET).push(str(pi))
        #     ps = GPPrimitiveSet()
        #     ps.setup(self, pbase)
        #     self.primitive_sets.append(ps)

        # # In ARC problem, we dynamically set up the primitive set based on a given DSL python file.

        # self.generation = 0

        # from src.ec.population import Population
        # self.population = Population()
        # self.population.setup(self, Parameter(self.P_POP))

    # @override
    # def startFresh(self):

    #     self.output.message("Setting up a new run")
    #     self.setup()  # garbage Parameter equivalent
    #     self.num_gen_trap_fit = 0
    #     self.previous_best_ind = None

    #     # POPULATION INITIALIZATION
    #     self.output.message("Initializing Generation 0")
    #     self.statistics.preInitializationStatistics(self)
    #     # self.population = self.initializer.initialPopulation(self, 0)
    #     self.population.populate(self, 0)
    #     self.statistics.postInitializationStatistics(self)

    #     # Compute generations from evaluations if necessary
    #     if self.numEvaluations > self.UNDEFINED:
    #         generationSize = sum(
    #             len(subpop.individuals) for subpop in self.population.subpops
    #         )

    #         if self.numEvaluations < generationSize:
    #             self.numEvaluations = generationSize
    #             self.numGenerations = 1
    #             self.output.warning(f"Using evaluations, but evaluations is less than the initial total population size ({generationSize}).  Setting to the population size.")
    #         else:
    #             if self.numEvaluations % generationSize != 0:
    #                 new_evals = (self.numEvaluations // generationSize) * generationSize
    #                 self.output.warning(f"Using evaluations, but initial total population size does not divide evenly into it.  Modifying evaluations to a smaller value ({new_evals}) which divides evenly.")
    #             self.numGenerations = self.numEvaluations // generationSize
    #             self.numEvaluations = self.numGenerations * generationSize

    #         self.output.message(f"Generations will be {self.numGenerations}")

    #     # self.exchanger.initializeContacts(self)
    #     # self.evaluator.initializeContacts(self)

    