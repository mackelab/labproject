import torch
from labproject.metrics import sliced_wasserstein_distance, gaussian_kl_divergence
from labproject.plotting import plot_scaling_metric_dimensionality, plot_scaling_metric_sample_size
from labproject.metrics.gaussian_squared_wasserstein import gaussian_squared_w2_distance
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

    def plot_experiment(self, dimensionality, distances, dataset_name, ax=None):
        plot_scaling_metric_dimensionality(
            dimensionality, distances, self.metric_name, dataset_name, ax=ax
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


class ScaleSampleSize(Experiment):

    def __init__(
        self, metric_name, metric_fn, min_samples=3, max_samples=2000, step=100, sample_sizes=None
    ):
        assert min_samples > 2, "min_samples must be greater than 2 to compute covariance for KL"
        self.metric_name = metric_name
        self.metric_fn = metric_fn
        # TODO: add logarithmic scale or only keep pass in run experiment
        if sample_sizes is not None:
            self.sample_sizes = sample_sizes
        else:
            self.sample_sizes = list(range(min_samples, max_samples, step))
        print(self.sample_sizes)
        super().__init__()

    def run_experiment(self, dataset1, dataset2, nb_runs=5, sample_sizes=None):
        """
        Computes for each subset 5 different random subsets and averages performance across the subsets.
        """
        final_distances = []
        final_errors = []
        if sample_sizes is None:
            sample_sizes = self.sample_sizes
        for idx in range(nb_runs):
            distances = []
            for n in sample_sizes:
                data1 = dataset1[torch.randperm(dataset1.size(0))[:n], :]
                data2 = dataset2[torch.randperm(dataset2.size(0))[:n], :]
                distances.append(self.metric_fn(data1, data2))
            final_distances.append(distances)
        final_distances = torch.transpose(torch.tensor(final_distances), 0, 1)
        final_errors = (
            torch.tensor([torch.std(d) for d in final_distances])
            if nb_runs > 1
            else torch.zeros_like(torch.tensor(sample_sizes))
        )
        final_distances = torch.tensor([torch.mean(d) for d in final_distances])

        return sample_sizes, final_distances, final_errors

    def plot_experiment(
        self,
        sample_sizes,
        distances,
        errors,
        dataset_name,
        ax=None,
        color=None,
        label=None,
        linestyle="-",
        **kwargs
    ):
        plot_scaling_metric_sample_size(
            sample_sizes,
            distances,
            errors,
            self.metric_name,
            dataset_name,
            ax=ax,
            color=color,
            label=label,
            linestyle=linestyle,
            **kwargs
        )

    def log_results(self, results, log_path):
        """
        Save the results to a file.
        """
        with open(log_path, "wb") as f:
            pickle.dump(results, f)


class ScaleSampleSizeKL(ScaleSampleSize):
    def __init__(self, min_samples=3, sample_sizes=None, **kwargs):
        super().__init__(
            "KL",
            gaussian_kl_divergence,
            min_samples=min_samples,
            sample_sizes=sample_sizes,
            **kwargs
        )


class ScaleSampleSizeSW(ScaleSampleSize):
    def __init__(self, min_samples=3, sample_sizes=None, **kwargs):
        super().__init__(
            "Sliced Wasserstein",
            sliced_wasserstein_distance,
            min_samples=min_samples,
            sample_sizes=sample_sizes,
            **kwargs
        )


class CIFAR10_FID_Train_Test(Experiment):
    def __init__(self):
        super().__init__()

    def run_experiment(self, dataset1, dataset2):
        fid_metric = gaussian_squared_w2_distance(dataset1, dataset2)
        return fid_metric

    def log_results(self, fid_metric, log_path):
        with open(log_path, "wb") as f:
            pickle.dump(fid_metric, f)

    def plot_experiment(self, fid_metric, dataset_name):
        pass