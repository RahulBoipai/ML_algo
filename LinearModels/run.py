from linear_regression_exact import main as exact_linear
from linear_regression_SGD import main as SGD_linear
from logistic_regression import main as logistic


if __name__ == '__main__':
    
    ############################################################################################################

    # Linear regression (OLS)

    # exact_linear()  # Exact linear regression (Ordinary Least Squares)
    # exact_linear(poly_degree=2)  # Ordinary Least Squares with polynomial kernel
    # exact_linear(poly_degree=20)  # OLS with polynomial kernel (overfitting case)

    ############################################################################################################

    # Linear regression (SGD)

    # SGD_linear()  # SGD linear regression
    # SGD_linear(poly_degree=2)  # SGD linear regression with polynomial kernel

    # L2 regularization (the model parameters are penalized for being large)
    # As the regularization coefficient increases, the model parameters are penalized more and more
    SGD_linear(poly_degree=2, L2_reg=0.1)  # SGD linear regression with polynomial kernel and L2 regularization
    # SGD_linear(poly_degree=2, L2_reg=1.0)  # SGD linear regression with polynomial kernel and L2 regularization

    # L1 regularization (the model parameters are penalized for being large in absolute value)
    # As the regularization coefficient increases, the model parameters become more sparse
    # i.e. the 3rd and 4th model parameters are go close to 0 since they are not important (data is quadratic)
    # SGD_linear(poly_degree=4, L1_reg=0.1)  # SGD linear regression with polynomial kernel and L1 regularization
    # SGD_linear(poly_degree=4, L1_reg=1.0)  # SGD linear regression with polynomial kernel and L1 regularization

    ############################################################################################################

    # Logistic regression

    # logistic()  # Logistic regression without normalization
    #logistic(normalize=True)  # Logistic regression with normalization

    # Hint: To notice the effect of normalization, notice the range of values
    # at the x and y axes in the classification plot

    ############################################################################################################
