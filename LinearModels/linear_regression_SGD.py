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
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        L1_reg: float = 0,
        L2_reg: float = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a linear model using SGD.

    Args:
        model: The model to train.
        X_train: The training data.
        y_train: The training labels.
        X_val: The validation data.
        y_val: The validation labels.
        batch_size: The batch size.
        num_epochs: The number of epochs.
        learning_rate: The learning rate.
        L1_reg: The L1 regularization strength.
        L2_reg: The L2 regularization strength.

    Returns:
        train_loss: The training loss at each epoch.
        train_metric: The training metric at each epoch.
        val_loss: The validation loss at each epoch.
        val_metric: The validation metric at each epoch.
    '''
    loss_fn = lambda y, y_pred: np.mean((y - y_pred) ** 2)
    metric_fn = lambda y, y_pred: np.mean(np.abs(y - y_pred))

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

            y_pred = model(X_batch)
            loss = loss_fn(y_batch, y_pred)

            grad_W = 2 * (X_batch.T @ X_batch @ model.W - X_batch.T @ y_batch)
            grad_W += L1_reg * np.sum(np.sign(model.W))  # L1 regularization (if L1_reg > 0)
            grad_W += L2_reg * 2 * np.sum(model.W)  # L2 regularization (if L2_reg > 0)
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

            y_pred = model(X_batch)
            loss = loss_fn(y_batch, y_pred)

            val_loss_epoch += loss * X_batch.shape[0]
            val_metric_epoch += metric_fn(y_batch, y_pred) * X_batch.shape[0]
            total_val_samples += X_batch.shape[0]
        val_loss_epoch /= total_val_samples
        val_metric_epoch /= total_val_samples
        val_loss.append(val_loss_epoch)
        val_metric.append(val_metric_epoch)

        # # Early stopping (based on validation loss; uncomment to use)
        # if epoch > 0 and val_loss_epoch > val_loss[-2]:
        #     print('Early stopping at epoch', epoch + 1)
        #     break

    return train_loss, train_metric, val_loss, val_metric



def main(
        poly_degree: int = 1,
        L1_reg: float = 0.0,
        L2_reg: float = 0.0,
    ) -> None:
    '''
    Main function for linear regression with SGD.

    Args:
        poly_degree: The degree of the polynomial to fit.
        L1_reg: The L1 regularization coefficient.
        L2_reg: The L2 regularization coefficient.
    
    Returns:
        None
    '''
    # Load data
    X, y = load_data('toy_reg')
    X = np.concatenate([X**i for i in range(1, poly_degree + 1)], axis=1)
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # Initialize model
    model = LinearModel(X_train.shape[1])

    # Train model
    train_loss, train_metric, val_loss, val_metric = train(
        model, X_train, y_train, X_val, y_val, L1_reg=L1_reg, L2_reg=L2_reg)
    for i in range(0, len(train_loss), 10):
        print(f'Epoch {i + 1}: train loss {train_loss[i]:.4f}, '
              f'val loss {val_loss[i]:.4f}')
    print("Parameters:", model.W)

    # Test model
    train_set_loss, train_set_metric =\
        get_performance(model, X_train, y_train, 'mse', 'mae', 32)
    print(f'Train set performance: MSE {train_set_loss:.4f}, ',
            f'MAE {train_set_metric:.4f}')
    test_set_loss, test_set_metric =\
        get_performance(model, X_test, y_test, 'mse', 'mae', 32)
    print(f'Test set performance: MSE {test_set_loss:.4f}, ',
            f'MAE {test_set_metric:.4f}')

    # Plot fit
    plot_fit(
        X_train, y_train, model, partition='train', poly_degree=poly_degree)
    plot_fit(X_val, y_val, model, partition='val', poly_degree=poly_degree)
    plot_fit(X_test, y_test, model, partition='test', poly_degree=poly_degree)

    # # Plot metrics
    plot_metrics(train_loss, val_loss, 'mse')
    plot_metrics(train_metric, val_metric, 'mae')
