from src.ec.util.parameter_database import ParameterDatabase
from src.ec.util.parameter import Parameter
from src.ec.util.output import Output
from src.ec.evolution_state import EvolutionState
from random import Random
from typing import override

class Xy_EvolutionState(EvolutionState):

    @override
    def startFresh(self):

        # the original setup function in EvolutionState is moved to the init function of LGP models when given X and y.

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