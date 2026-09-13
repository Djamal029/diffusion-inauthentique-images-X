from skopt import gp_minimize
from skopt.plots import plot_objective, plot_convergence


class BayesianOptimization:
    def __init__(
        self,
        objective_func,
        space,
        n_calls=50,
        n_initial_points=10,
        acq_func="EI",
        random_state=42,
    ):
        self.objective_func = objective_func
        self.space = space
        self.n_calls = n_calls
        self._n_initial_points = n_initial_points
        self.acq_func = acq_func
        self.random_state = random_state
        self.results = None

    def run_optimization(self):
        self.results = gp_minimize(
            func=self.objective_func,
            dimensions=self.space,
            n_calls=self.n_calls,
            n_initial_points=self._n_initial_points,
            acq_func=self.acq_func,
            random_state=self.random_state,
        )
        return self

    # ---------- Résultats ----------

    @property
    def best_params(self):
        if self.results is None:
            raise RuntimeError("Optimization has not been run yet.")
        return dict(zip([dim.name for dim in self.space], self.results.x))

    @property
    def best_score(self):
        if self.results is None:
            raise RuntimeError("Optimization has not been run yet.")
        return -self.results.fun

    @property
    def history(self):
        if self.results is None:
            raise RuntimeError("Optimization has not been run yet.")
        return {
            "params": self.results.x_iters,
            "scores": -self.results.func_vals,
        }

    # ---------- Visualisation ----------

    def plot_objective_func(self):
        if self.results is None:
            raise RuntimeError("Optimization has not been run yet.")
        plot_objective(self.results)

    def plot_convergence(self):
        if self.results is None:
            raise RuntimeError("Optimization has not been run yet.")
        plot_convergence(self.results)
