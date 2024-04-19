import numpy as np

from utils import *
from model import LinearModel


def train(model: LinearModel, X_train: np.ndarray, y_train: np.ndarray) -> None:
    '''
    Train a linear model using the exact OLS solution.

    Args:
        model: The model to train.
        X_train: The training data.
        y_train: The training labels.
    
    Returns:
        None
    '''
    model.W = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train


def main(poly_degree: int = 1) -> None:
    '''
    Main function for linear regression with exact solution.

    Args:
        poly_degree: The degree of the polynomial kernel to use.
    
    Returns:
        None
    '''
    # Load data
    X, y = load_data('toy_reg')
    X = np.concatenate([X**i for i in range(1, poly_degree+1)], axis=1)
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    y = y.reshape(-1, 1)
    X_train, y_train, _, _, X_test, y_test = split_data(X, y)

    # Initialize model
    model = LinearModel(X_train.shape[1])
    print(model)

    # Train model
    train(model, X_train, y_train)

    # Get performance
    train_loss, train_metric = get_performance(
        model, X_train, y_train, 'mse', 'mae')
    test_loss, test_metric = get_performance(
        model, X_test, y_test, 'mse', 'mae')
    print("Linear regression results:")
    print('Train loss: %.4f, metric: %.4f' % (train_loss, train_metric))
    print('Test loss: %.4f, metric: %.4f' % (test_loss, test_metric))

    # Plot fit
    plot_fit(
        X_train, y_train, model, partition='train', poly_degree=poly_degree)
    plot_fit(
        X_test, y_test, model, partition='test', poly_degree=poly_degree)
