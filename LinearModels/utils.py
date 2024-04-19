import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from model import LinearModel


def load_data(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Create toy regression and classification datasets.

    Args:
        dataset: name of dataset

    Returns:
        Data, target
    '''
    if dataset == 'toy_reg':
        X = np.linspace(-2, 1, 200).reshape(-1, 1)
        y = - 2 * X * X + np.random.RandomState(2023).randn(*X.shape) * 0.33
    elif dataset == 'toy_clf':
        X = np.zeros((200, 2))
        X[:100] = np.random.RandomState(2023).randn(100, 2) + np.array([3, 3])
        X[100:] = np.random.RandomState(2023).randn(100, 2) + np.array([-3, -3])
        y = np.array([0] * 100 + [1] * 100).reshape(-1, 1)
    else:
        raise ValueError('Unknown dataset: %s' % dataset)
    return X, y


def split_data(
        X: np.ndarray,
        y: np.ndarray,
        split_ratio: list = [0.7, 0.1, 0.2],
        random_state: int = 2023,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Split data into train, validation, and test sets. (default: 70%, 10%, 20%)
    (An alternative is to use sklearn.model_selection.train_test_split)

    Args:
        X: input data
        y: target data
        split_ratio: list of split ratios
        random_state: random seed

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    '''

    np.random.RandomState(random_state).shuffle(X)
    np.random.RandomState(random_state).shuffle(y)

    X_train = X[:int(X.shape[0] * split_ratio[0])]
    y_train = y[:int(y.shape[0] * split_ratio[0])]
    X_val = X[
        int(X.shape[0] * split_ratio[0]):int(X.shape[0] * (
                split_ratio[0] + split_ratio[1]))]
    y_val = y[
        int(y.shape[0] * split_ratio[0]):int(y.shape[0] * (
                split_ratio[0] + split_ratio[1]))]
    X_test = X[int(X.shape[0] * split_ratio[0] + split_ratio[1]):]
    y_test = y[int(y.shape[0] * split_ratio[0] + split_ratio[1]):]
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_performance(
        model: LinearModel,
        X: np.ndarray,
        y: np.ndarray,
        loss: str,
        metric: str,
        batch_size=32,
    ) -> Tuple[float, float]:
    '''
    Get performance of model on data according to specified loss and metric.

    Args:
        model: model
        X: input data
        y: target data
        loss: loss function
        metric: metric function
        batch_size: batch size

    Returns:
        loss, metric
    '''
    num_batches = X.shape[0] // batch_size + (X.shape[0] % batch_size > 0)
    sigmoid_fn = lambda x: 1 / (1 + np.exp(-x))

    if loss == 'mse':
        loss_fn = lambda y, y_pred: np.mean((y - y_pred) ** 2)
    elif loss == 'mae':
        loss_fn = lambda y, y_pred: np.mean(np.abs(y - y_pred))
    elif loss == 'bce':
        loss_fn = \
            lambda y, y_pred: -np.mean(
                y * np.log(sigmoid_fn(y_pred)) + (1 - y) * np.log(1 - sigmoid_fn(y_pred)))
    else:
        raise ValueError('Unknown loss function: %s' % loss)

    if metric == 'mse':
        metric_fn = lambda y, y_pred: np.mean((y - y_pred) ** 2)
    elif metric == 'mae':
        metric_fn = lambda y, y_pred: np.mean(np.abs(y - y_pred))
    elif metric == 'acc':
        metric_fn = lambda y, y_pred:\
            np.mean(y.reshape(-1,1) == (sigmoid_fn(y_pred) > 0.5).reshape(-1,1))
    else:
        raise ValueError('Unknown metric function: %s' % metric)
    
    loss = 0
    metric = 0
    total_samples = 0
    for i in range(num_batches):
        X_batch = X[i * batch_size:(i + 1) * batch_size]
        y_batch = y[i * batch_size:(i + 1) * batch_size]

        y_pred = model(X_batch)
        loss += loss_fn(y_batch, y_pred) * X_batch.shape[0]
        metric += metric_fn(y_batch, y_pred) * X_batch.shape[0]
        total_samples += X_batch.shape[0]
    loss /= total_samples
    metric /= total_samples

    return loss, metric


def plot_metrics(
        train_metric: np.ndarray,
        val_metric: np.ndarray,
        metric_name: str, ) -> None:
    '''
    Plot training and validation metrics.

    Args:
        train_metric: training metric
        val_metric: validation metric
        metric_name: name of metric

    Returns:
        None
    '''
    plt.plot(train_metric, label='train')
    plt.plot(val_metric, label='val')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()


def plot_fit(
        X: np.ndarray,
        y: np.ndarray,
        model: LinearModel,
        partition: str = 'train',
        poly_degree: int = 1,
    ) -> None:
    '''
    Plot data and model fit.

    Args:
        X: input data
        y: target data
        model: model
        partition: partition of data
        poly_degree: degree of polynomial

    Returns:
        None
    '''
    plt.scatter(X[:, 0], y, label='data')
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    Xs = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    Xs = np.concatenate([Xs ** i for i in range(1, poly_degree + 1)], axis=1)
    Xs = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
    plt.plot(Xs[:, 0], model(Xs), label='fit', color='red')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('%s %s' % (partition, 'fit'))
    plt.legend()
    plt.show()


def plot_decision_boundary(clf, X, Y, partition, cmap='Paired_r'):
    '''
    Plot decision boundary of model.

    Args:
        clf: model
        X: input data
        Y: target data
        partition: partition of data
        cmap: colormap

    Returns:
        None
    '''
    h = 0.02
    x_min, x_max = X[:, 0].min() - 10 * h, X[:, 0].max() + 10 * h
    y_min, y_max = X[:, 1].min() - 10 * h, X[:, 1].max() + 10 * h
    xs, ys = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = []
    for x, y in zip(xs.ravel(), ys.ravel()):
        Z.append(clf([x, y, 1]) > 0.5)
    Z = np.array(Z).reshape(xs.shape)

    plt.figure(figsize=(5, 5))
    plt.contourf(xs, ys, Z, cmap=cmap, alpha=0.25)
    plt.contour(xs, ys, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap, edgecolors='k')
    plt.title('%s decision boundary' % partition)
    plt.show()
