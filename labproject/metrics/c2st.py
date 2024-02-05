# from sbi: https://github.com/sbi-dev/sbi/blob/main/sbi/utils/metrics.py

from typing import Any, Dict, Optional, Literal

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from torch import Tensor


def c2st_nn(
    X: Tensor,
    Y: Tensor,
    seed: int = 1,
    n_folds: int = 5,
    metric: str = "accuracy",
    activation: Literal["identity", "logistic", "tanh", "relu"] = "relu",
    clf_kwargs: dict[str, Any] = {},
) -> Tensor:
    r"""
    Return accuracy of MLP classifier trained to distinguish samples from supposedly
    two distributions <X> and <Y>. For details on the method, see [1,2].
    If the returned accuracy is 0.5, <X> and <Y> are considered to be from the
    same generating PDF, i.e. they can not be differentiated.
    If the returned accuracy is around 1., <X> and <Y> are considered to be from
    two different generating PDFs.

    Training of the `MLPClassifier` from `sklearn.neural_network` is performed with
    N-fold cross-validation [3]. Before both samples are ingested, they are normalized
    (z scored) under the assumption that each dimension in X follows a normal distribution,
    i.e. the mean(X) is subtracted from X and this difference is divided by std(X)
    for every dimension.

    If you need a more flexible interface which is able to take a sklearn
    compatible classifier and more, see the `c2st_score` method in this module.

    Args:
        X: Samples from one distribution.
        Y: Samples from another distribution.
        seed: Seed for the sklearn classifier and the KFold cross-validation
        n_folds: Number of folds to use
        metric: sklearn compliant metric to use for the scoring parameter of cross_val_score
        activation: activation function for the hidden layer
        clf_kwargs: additional kwargs for `MLPClassifier`

    Return:
        torch.tensor containing the mean accuracy score over the test sets
        from cross-validation

    Example:
    ``` py
    > c2st_nn(X,Y)
    [0.51904464] #X and Y likely come from the same PDF or ensemble
    > c2st_nn(P,Q)
    [0.998456] #P and Q likely come from two different PDFs or ensembles
    ```

    References:
        [1]: http://arxiv.org/abs/1610.06545
        [2]: https://www.osti.gov/biblio/826696/
        [3]: https://scikit-learn.org/stable/modules/cross_validation.html
        [4]: https://github.com/psteinb/c2st/
    """

    # the default configuration
    clf_class = MLPClassifier
    ndim = X.shape[-1]
    defaults = {
        "activation": activation,
        "hidden_layer_sizes": (10 * ndim, 10 * ndim),
        "max_iter": 1000,
        "solver": "adam",
        "early_stopping": True,
        "n_iter_no_change": 50,
    }
    defaults.update(clf_kwargs)

    scores_ = c2st_scores(
        X,
        Y,
        seed=seed,
        n_folds=n_folds,
        metric=metric,
        z_score=True,
        noise_scale=None,
        verbosity=0,
        clf_class=clf_class,
        clf_kwargs=defaults,
    )

    scores = np.mean(scores_).astype(np.float32)
    value = torch.from_numpy(np.atleast_1d(scores))
    return value


def c2st_rf(
    X: Tensor,
    Y: Tensor,
    seed: int = 1,
    n_folds: int = 5,
    metric: str = "accuracy",
    n_estimators: int = 100,
    clf_kwargs: dict[str, Any] = {},
) -> Tensor:
    r"""
    Return accuracy of random forest classifier trained to distinguish samples from supposedly
    two distributions <X> and <Y>. For details on the method, see [1,2].
    If the returned accuracy is 0.5, <X> and <Y> are considered to be from the
    same generating PDF, i.e. they can not be differentiated.
    If the returned accuracy is around 1., <X> and <Y> are considered to be from
    two different generating PDFs.

    Training of the `RandomForestClassifier` from `sklearn.ensemble` is performed with
    N-fold cross-validation [3]. Before both samples are ingested, they are normalized
    (z scored) under the assumption that each dimension in X follows a normal distribution,
    i.e. the mean(X) is subtracted from X and this difference is divided by std(X)
    for every dimension.

    If you need a more flexible interface which is able to take a sklearn
    compatible classifier and more, see the `c2st_score` method in this module.

    Args:
        X: Samples from one distribution.
        Y: Samples from another distribution.
        seed: Seed for the sklearn classifier and the KFold cross-validation
        n_folds: Number of folds to use
        metric: sklearn compliant metric to use for the scoring parameter of cross_val_score
        n_estimators: the number of trees in the forest
        clf_kwargs: additional kwargs for `RandomForestClassifier`

    Return:
        torch.tensor containing the mean accuracy score over the test sets
        from cross-validation

    Example:
    ``` py
    > c2st_rf(X,Y)
    [0.51904464] #X and Y likely come from the same PDF or ensemble
    > c2st_rf(P,Q)
    [0.998456] #P and Q likely come from two different PDFs or ensembles
    ```

    References:
        [1]: http://arxiv.org/abs/1610.06545
        [2]: https://www.osti.gov/biblio/826696/
        [3]: https://scikit-learn.org/stable/modules/cross_validation.html
        [4]: https://github.com/psteinb/c2st/
    """

    # the default configuration
    clf_class = RandomForestClassifier
    clf_kwargs["n_estimators"] = n_estimators

    scores_ = c2st_scores(
        X,
        Y,
        seed=seed,
        n_folds=n_folds,
        metric=metric,
        z_score=True,
        noise_scale=None,
        verbosity=0,
        clf_class=clf_class,
        clf_kwargs=clf_kwargs,
    )

    scores = np.mean(scores_).astype(np.float32)
    value = torch.from_numpy(np.atleast_1d(scores))
    return value


def c2st_knn(
    X: Tensor,
    Y: Tensor,
    seed: int = 1,
    n_folds: int = 5,
    metric: str = "accuracy",
    n_neighbors: int = 5,
    clf_kwargs: dict = {},
) -> Tensor:
    r"""
    Return accuracy of K-nearest neighbors classifier trained to distinguish samples from supposedly
    two distributions <X> and <Y>. For details on the method, see [1,2].
    If the returned accuracy is 0.5, <X> and <Y> are considered to be from the
    same generating PDF, i.e. they can not be differentiated.
    If the returned accuracy is around 1., <X> and <Y> are considered to be from
    two different generating PDFs.

    Training of the `KNeighborsClassifier` from `sklearn.neighbors` is performed with
    N-fold cross-validation [3]. Before both samples are ingested, they are normalized
    (z scored) under the assumption that each dimension in X follows a normal distribution,
    i.e. the mean(X) is subtracted from X and this difference is divided by std(X)
    for every dimension.

    If you need a more flexible interface which is able to take a sklearn
    compatible classifier and more, see the `c2st_score` method in this module.

    Args:
        X: Samples from one distribution.
        Y: Samples from another distribution.
        seed: Seed for the sklearn classifier and the KFold cross-validation
        n_folds: Number of folds to use
        metric: sklearn compliant metric to use for the scoring parameter of cross_val_score
        n_neighbors: the number of neighbors to use by default for `kneighbors` queries
        clf_kwargs: additional kwargs for `KNeighborsClassifier`

    Return:
        torch.tensor containing the mean accuracy score over the test sets
        from cross-validation

    Example:
    ``` py
    > c2st_knn(X,Y)
    [0.51904464] #X and Y likely come from the same PDF or ensemble
    > c2st_knn(P,Q)
    [0.998456] #P and Q likely come from two different PDFs or ensembles
    ```

    References:
        [1]: http://arxiv.org/abs/1610.06545
        [2]: https://www.osti.gov/biblio/826696/
        [3]: https://scikit-learn.org/stable/modules/cross_validation.html
        [4]: https://github.com/psteinb/c2st/
    """

    # the default configuration
    clf_class = KNeighborsClassifier
    clf_kwargs["n_neighbors"] = n_neighbors

    scores_ = c2st_scores(
        X,
        Y,
        seed=seed,
        n_folds=n_folds,
        metric=metric,
        z_score=True,
        noise_scale=None,
        verbosity=0,
        clf_class=clf_class,
        clf_kwargs=clf_kwargs,
    )

    scores = np.mean(scores_).astype(np.float32)
    value = torch.from_numpy(np.atleast_1d(scores))
    return value


def c2st_scores(
    X: Tensor,
    Y: Tensor,
    seed: int = 1,
    n_folds: int = 5,
    metric: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    verbosity: int = 0,
    clf_class: Any = RandomForestClassifier,
    clf_kwargs: Dict[str, Any] = {},
) -> Tensor:
    r"""
    Return accuracy of classifier trained to distinguish samples from supposedly
    two distributions <X> and <Y>. For details on the method, see [1,2].
    If the returned accuracy is 0.5, <X> and <Y> are considered to be from the
    same generating PDF, i.e. they can not be differentiated.
    If the returned accuracy is around 1., <X> and <Y> are considered to be from
    two different generating PDFs.

    This function performs training of the classifier with N-fold cross-validation [3] using sklearn.
    By default, a `RandomForestClassifier` by from `sklearn.ensemble` is used which
    is recommended based on the study performed in [4].
    This can be changed using <clf_class>. This class is used in the following
    fashion:

    ``` py
    clf = clf_class(random_state=seed, **clf_kwargs)
    #...
    scores = cross_val_score(
        clf, data, target, cv=shuffle, scoring=scoring, verbose=verbosity
    )
    ```
    Further configuration of the classifier can be performed using <clf_kwargs>.
    If you like to provide a custom class for training, it has to satisfy the
    internal requirements of `sklearn.model_selection.cross_val_score`.

    Args:
        X: Samples from one distribution.
        Y: Samples from another distribution.
        seed: Seed for the sklearn classifier and the KFold cross validation
        n_folds: Number of folds to use for cross validation
        metric: sklearn compliant metric to use for the scoring parameter of cross_val_score
        z_score: Z-scoring using X, i.e. mean and std deviation of X is used to normalize Y, i.e. Y=(Y - mean)/std
        noise_scale: If passed, will add Gaussian noise with standard deviation <noise_scale> to samples of X and of Y
        verbosity: control the verbosity of sklearn.model_selection.cross_val_score
        clf_class: a scikit-learn classifier class
        clf_kwargs: key-value arguments dictionary to the class specified by clf_class, e.g. sklearn.ensemble.RandomForestClassifier

    Return:
        np.ndarray containing the calculated <metric> scores over the test set
        folds from cross-validation

    Example:
    ``` py
    > c2st_scores(X,Y)
    [0.51904464,0.5309201,0.4959452,0.5487709,0.50682926]
    #X and Y likely come from the same PDF or ensemble
    > c2st_scores(P,Q)
    [0.998456,0.9982912,0.9980476,0.9980488,0.99805826]
    #P and Q likely come from two different PDFs or ensembles
    ```

    References:
        [1]: http://arxiv.org/abs/1610.06545
        [2]: https://www.osti.gov/biblio/826696/
        [3]: https://scikit-learn.org/stable/modules/cross_validation.html
        [4]: https://github.com/psteinb/c2st/
    """
    if z_score:
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * torch.randn(X.shape)
        Y += noise_scale * torch.randn(Y.shape)

    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    clf = clf_class(random_state=seed, **clf_kwargs)

    # prepare data
    data = np.concatenate((X, Y))
    # labels
    target = np.concatenate((np.zeros((X.shape[0],)), np.ones((Y.shape[0],))))

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=metric, verbose=verbosity)

    return scores
