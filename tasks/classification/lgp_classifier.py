import sys
import time
import numpy as np

from tasks.classification.optimization.lgp_classification import LGPClassificationProblem
from tasks.lgp_template import LGP_Model_Template


class LinearGP_Classifier(LGP_Model_Template):

	def __init__(
		self,
		param_file: str,
		seed: int = 4,
		setup_problem_script=False,
		output_file1: str = None,
		output_file2: str = None,
		dsl_output_path: str = None,
		module_path: str = None,
		output_file3: str = None,
		output_file4: str = None,
	):
		"""
		When output_file1 and output_file2 are None, output paths follow
		parameter file settings.
		"""
		super().__init__(
			param_file,
			seed,
			setup_problem_script=setup_problem_script,
			output_file1=output_file1,
			output_file2=output_file2,
			dsl_output_path=dsl_output_path,
			module_path=module_path,
			output_file3=output_file3,
			output_file4=output_file4,
		)

		self.train_time = None
		self.train_fitness = None
		self.test_fitness = None

	def fit(self, X, y):
		"""
		Fit the model to data.
		X: array-like of shape (n_samples, n_features)
		y: array-like of shape (n_samples,) or (n_samples, 1)
		"""
		X = np.asarray(X)
		y = np.asarray(y)

		if y.ndim == 1:
			y = y.reshape(-1, 1)

		self.state.startFresh()

		if isinstance(self.state.evaluator.p_problem, LGPClassificationProblem):
			self.state.evaluator.p_problem.setData(X, y)
		else:
			raise ValueError(
				f"the optimization of LinearGP_Classifier must be type of {LGPClassificationProblem.__name__}"
			)

		self.state.run()

		self.output_ind = self.state.statistics.best_of_run[0]
		return self

	def predict(self, X) -> np.ndarray:
		"""
		Predict class labels using the trained model.
		X: array-like of shape (n_samples, n_features)
		Returns: ndarray of shape (n_samples,)
		"""
		X = np.asarray(X)

		if isinstance(self.state.evaluator.p_problem, LGPClassificationProblem):
			self.state.evaluator.p_problem.setX(X)
		else:
			raise ValueError(
				f"the optimization of LinearGP_Classifier must be type of {LGPClassificationProblem.__name__}"
			)

		if not self.output_ind:
			print("the linear genetic programming has not been trained. I found no output individual")
			sys.exit(1)

		predict = self.state.evaluator.p_problem.quickevaluate(self.output_ind)
		predict = np.asarray(predict, dtype=float)

		if predict.ndim == 2 and predict.shape[1] >= 1:
			return predict[:, 0]
		return predict.reshape(-1)

	def score(self, X, y):
		"""Return accuracy score, aligned with scikit-learn classifiers."""
		from sklearn.metrics import accuracy_score

		y_pred = self.predict(X)
		y_true = np.asarray(y)
		if y_true.ndim > 1 and y_true.shape[1] >= 1:
			y_true = y_true[:, 0]
		return accuracy_score(y_true, y_pred)

	def fit_params(self, loca: str, datan: str):
		"""
		Fit model based on file parameters.
		loca: file location of classification dataset.
		datan: name of the classification task.
		"""
		self.state.startFresh()

		if isinstance(self.state.evaluator.p_problem, LGPClassificationProblem):
			# fitn = self.state.evaluator.p_problem.fitness
			# self.state.evaluator.p_problem.setProblem(self.state, loca, datan, fitn, True)
			self.state.evaluator.p_problem.load_data(self.state, loca, datan, istraining=True)
		else:
			raise ValueError(
				f"the optimization of LinearGP_Classifier must be type of {LGPClassificationProblem.__name__}"
			)

		t0t = time.time()
		self.state.run()
		self.train_time = time.time() - t0t
		print("Training time measure:", self.train_time)

		self.output_ind = self.state.statistics.best_i[0]
		self.train_fitness = self.output_ind.fitness.fitness()

		return self

	def predict_params(self, loca: str, datan: str):
		"""
		Evaluate the trained individual on test data from parameter-based files.
		loca: file location of classification dataset.
		datan: name of the classification task.
		"""
		if isinstance(self.state.evaluator.p_problem, LGPClassificationProblem):
			# fitn = self.state.evaluator.p_problem.fitness
			# self.state.evaluator.p_problem.setProblem(None, loca, datan, fitn, False)
			self.state.evaluator.p_problem.load_data(self.state, loca, datan, istraining=False)
		else:
			raise ValueError(
				f"the optimization of LinearGP_Classifier must be type of {LGPClassificationProblem.__name__}"
			)

		if not self.output_ind:
			print("the linear genetic programming has not been trained. I found no output individual")
			sys.exit(1)

		self.output_ind.evaluated = False
		self.state.evaluator.p_problem.simpleevaluate(self.output_ind)
		self.test_fitness = self.output_ind.fitness.fitness()

	def play_quiz(self, loca: str, datan: str):
		self.fit_params(loca, datan)
		self.predict_params(loca, datan)

