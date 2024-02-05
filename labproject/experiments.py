import torch
from metrics import sliced_wasserstein_distance, gaussian_kl_divergence
from plotting import plot_scaling_metric_dimensionality
from metrics.gaussian_squared_wasserstein import gaussian_squared_w2_distance
import pickle


class Experiment:
    def __init__(self):
        pass

    def run_experiment(self, metric, dataset1, dataset2):
        raise NotImplementedError("Subclasses must implement this method")

    def plot_experiment(self):
        raise NotImplementedError("Subclasses must implement this method")

    def log_results(self, results, log_path):
        raise NotImplementedError("Subclasses must implement this method")


class ScaleDim(Experiment):
    def __init__(self, metric_name, metric_fn, min_dim=1, max_dim=1000, step=100):
        self.metric_name = metric_name
        self.metric_fn = metric_fn
        self.dimensionality = list(range(min_dim, max_dim, step))
        super().__init__()

    def run_experiment(self, dataset1, dataset2):
        distances = []
        for d in self.dimensionality:
            distances.append(self.metric_fn(dataset1[:, :d], dataset2[:, :d]))
        return self.dimensionality, distances

    def plot_experiment(self, dimensionality, distances, dataset_name):
        plot_scaling_metric_dimensionality(
            dimensionality, distances, self.metric_name, dataset_name
        )

    def log_results(self, results, log_path):
        """
        Save the results to a file.
        """
        with open(log_path, "wb") as f:
            pickle.dump(results, f)


class ScaleDimKL(ScaleDim):
    def __init__(self, min_dim=2, **kwargs):
        super().__init__("KL", gaussian_kl_divergence, min_dim=min_dim, **kwargs)


class ScaleDimSW(ScaleDim):
    def __init__(self, min_dim=2, **kwargs):
        super().__init__("Sliced Wasserstein", sliced_wasserstein_distance, **kwargs)
