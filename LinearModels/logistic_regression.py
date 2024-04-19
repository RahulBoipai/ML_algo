import numpy as np
from typing import Tuple

from utils import *
from model import LinearModel


def train(
        model: LinearModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
        num_epochs: int = 1000,
        learning_rate: float = 1e-1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a linear model for classification using SGD.

    Args:
        model: The model to train.
        X_train: The training data.
        y_train: The training labels.
        X_val: The validation data.
        y_val: The validation labels.
        batch_size: The batch size.
        num_epochs: The number of epochs.
        learning_rate: The learning rate.

    Returns:
        train_loss: The training loss at each epoch.
        train_metric: The training metric at each epoch.
        val_loss: The validation loss at each epoch.
        val_metric: The validation metric at each epoch.
    '''
    loss_fn = lambda y, y_pred: np.mean(
        -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))
    metric_fn = lambda y, y_pred: np.mean(y == (y_pred > 0.5))
    sigmoid_fn = lambda x: 1 / (1 + np.exp(-x))

    train_loss, train_metric = [], []
    val_loss, val_metric = [], []

    num_train_bathces =\
        X_train.shape[0] // batch_size + (X_train.shape[0] % batch_size > 0)
    num_val_batches =\
        X_val.shape[0] // batch_size + (X_val.shape[0] % batch_size > 0)

    for epoch in range(num_epochs):
        # Train
        train_loss_epoch, train_metric_epoch = 0, 0
        total_train_samples = 0
        shuffled_indices = np.random.permutation(X_train.shape[0])
        for i in range(num_train_bathces):
            batch_idxs = shuffled_indices[i * batch_size:(i + 1) * batch_size]
            X_batch = X_train[batch_idxs]
            y_batch = y_train[batch_idxs]

            outs = model(X_batch)
            y_pred = sigmoid_fn(outs)
            loss = loss_fn(y_batch, y_pred)

            grad_W = (
                (y_pred - y_batch).reshape(-1, 1) * X_batch
            ).mean(axis=0).reshape(-1, 1)
            model.W -= learning_rate * grad_W

            train_loss_epoch += loss * X_batch.shape[0]
            train_metric_epoch += metric_fn(y_batch, y_pred) * X_batch.shape[0]
            total_train_samples += X_batch.shape[0]
        train_loss_epoch /= total_train_samples
        train_metric_epoch /= total_train_samples
        train_loss.append(train_loss_epoch)
        train_metric.append(train_metric_epoch)

        # Validation
        val_loss_epoch, val_metric_epoch = 0, 0
        total_val_samples = 0
        for i in range(num_val_batches):
            X_batch = X_val[i * batch_size:(i + 1) * batch_size]
            y_batch = y_val[i * batch_size:(i + 1) * batch_size]

            y_pred = sigmoid_fn(model(X_batch))
            loss = loss_fn(y_batch, y_pred)

            val_loss_epoch += loss * X_batch.shape[0]
            val_metric_epoch += metric_fn(y_batch, y_pred) * X_batch.shape[0]
            total_val_samples += X_batch.shape[0]
        val_loss_epoch /= total_val_samples
        val_metric_epoch /= total_val_samples
        val_loss.append(val_loss_epoch)
        val_metric.append(val_metric_epoch)

        print(f'Epoch {epoch + 1}: train loss {train_loss_epoch:.4f}, '
                + f'val loss {val_loss_epoch:.4f}'
                + f', train acc {train_metric_epoch:.4f}, '
                + f'val acc {val_metric_epoch:.4f}')

    return train_loss, train_metric, val_loss, val_metric


def main(normalize: bool = False) -> None:
    '''
    Main function for logistic regression.

    Args:
        normalize: Whether to normalize the data.

    Returns:
        None
    '''
    # Load data
    X, y = load_data('toy_clf')

    # Normalize data
    if normalize:
        # X = (X - X.mean(axis=0)) / X.std(axis=0)
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    X = np.hstack([X, np.ones((X.shape[0], 1))])
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # Initialize model
    model = LinearModel(X_train.shape[1])

    # Train model
    train_loss, train_metric, val_loss, val_metric = \
        train(model, X_train, y_train, X_val, y_val)
    
    # Test model
    train_set_loss, train_set_metric =\
        get_performance(model, X_train, y_train, 'bce', 'acc', 32)
    print(f'Train set performance: BCE {train_set_loss:.4f}, ',
            f'Acc {train_set_metric:.4f}')
    test_set_loss, test_set_metric =\
        get_performance(model, X_test, y_test, 'bce', 'acc', 32)
    print(f'Test set performance: BCE {test_set_loss:.4f}, ',
            f'Acc {test_set_metric:.4f}')

    # Plot metrics
    plot_metrics(train_loss, val_loss, 'bce')
    plot_metrics(train_metric, val_metric, 'acc')

    # Show decision boundary
    plot_decision_boundary(model, X_train, y_train, 'train')
    plot_decision_boundary(model, X_val, y_val, 'val')
    plot_decision_boundary(model, X_test, y_test, 'test')
