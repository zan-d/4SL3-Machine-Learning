import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
np.random.seed(8647)  # using last 4 digits of student number

# code given by assignment file
X_train = np.linspace(0, 1, 10)  # training set
X_valid = np.linspace(0, 1, 100)  # validation set
t_train = np.sin(4 * np.pi * X_train) + 0.3 * np.random.randn(10)
t_valid = np.sin(4 * np.pi * X_valid) + 0.3 * np.random.randn(100)
# get t_true for graphing error
t_true = np.sin(4 * np.pi * X_valid)

# number of training and validation examples
N_t = len(X_train)
N_v = len(X_valid)


def main():
    # m_err_t, m_err_v, m_err used to plot training error/validation error vs M
    m_err_t = []
    m_err_v = []
    m_err = []
    m_err_true = []

    # linReg(M)
    # change M to see the model at different capacities
    # each graph is printed separately at the end of the function
    # if running lin_reg multiple times, to see the next graph, you have to exit the current graph
    output = lin_reg(0)
    m_err.append(0)
    m_err_t.append(output[0])
    m_err_v.append(output[1])
    m_err_true.append(output[2])

    output = lin_reg(1)
    m_err.append(1)
    m_err_t.append(output[0])
    m_err_v.append(output[1])
    m_err_true.append(output[2])

    output = lin_reg(2)
    m_err.append(2)
    m_err_t.append(output[0])
    m_err_v.append(output[1])
    m_err_true.append(output[2])

    output = lin_reg(3)
    m_err.append(3)
    m_err_t.append(output[0])
    m_err_v.append(output[1])
    m_err_true.append(output[2])

    output = lin_reg(4)
    m_err.append(4)
    m_err_t.append(output[0])
    m_err_v.append(output[1])
    m_err_true.append(output[2])

    output = lin_reg(5)
    m_err.append(5)
    m_err_t.append(output[0])
    m_err_v.append(output[1])
    m_err_true.append(output[2])

    output = lin_reg(6)
    m_err.append(6)
    m_err_t.append(output[0])
    m_err_v.append(output[1])
    m_err_true.append(output[2])

    output = lin_reg(7)
    m_err.append(7)
    m_err_t.append(output[0])
    m_err_v.append(output[1])
    m_err_true.append(output[2])

    output = lin_reg(8)
    m_err.append(8)
    m_err_t.append(output[0])
    m_err_v.append(output[1])
    m_err_true.append(output[2])

    output = lin_reg(9)
    m_err.append(9)
    m_err_t.append(output[0])
    m_err_v.append(output[1])
    m_err_true.append(output[2])

    # regularization graph
    m_err_reg = []
    m_err_t_reg = []
    m_err_v_reg = []
    m_err_reg1 = []
    m_err_t_reg1 = []
    m_err_v_reg1 = []
    # lin_reg_regular(M, lambda)
    # regularization
    output = lin_reg_regular(9, 9e-11)
    m_err_reg.append(9)
    m_err_t_reg.append(output[0])
    m_err_v_reg.append(output[1])

    output = lin_reg_regular(9, 9e-3)
    m_err_reg1.append(9)
    m_err_t_reg1.append(output[0])
    m_err_v_reg1.append(output[1])

    # graph
    plt.scatter(m_err, m_err_t, color='magenta', label='training error')
    plt.scatter(m_err, m_err_v, color='blue', label='validation error')
    plt.plot(m_err, m_err_true, color='purple', label='validation/true error')
    plt.scatter(m_err_reg, m_err_t_reg, color='green', label='training error regularized')
    plt.scatter(m_err_reg, m_err_v_reg, color='red', label='validation error regularized')
    plt.scatter(m_err_reg1, m_err_t_reg1, color='orange', label='underfit train err regularized')
    plt.scatter(m_err_reg1, m_err_v_reg1, color='yellow', label='underfit valid err regularized')
    plt.legend(loc='best')
    plt.title('errors at each M (ZD, 400148647)')
    plt.xlabel('M')
    plt.ylabel('error')
    plt.show()


########################################
# function for error calculation
########################################
def err(y_vector, t_vector, N, err_type):
    diff_t = np.subtract(t_vector, y_vector)
    err_t = np.dot(diff_t.T, diff_t) / N
    RMSE_t = np.sqrt(err_t)
    print("{0} error: {1}".format(err_type, err_t))
    print("root mean square {0} error: {1}".format(err_type, RMSE_t))
    return err_t


########################################
# train model without regularization
########################################
def lin_reg(M):
    ##### training #####
    # For M = 0, X will equal a column vector of 1, so X.T*X = N
    # therefore, for M = 0, w = X.T*t/N

    # get X
    if M == 0:
        X_t = np.ones(N_t).T
    else:
        ones_col = np.ones(N_t)  # get a row of ones
        X_t = np.array([ones_col, X_train]).T

    if M > 1:
        for i in range(2, M + 1):
            X_t = np.c_[X_t, X_train ** i]

    # get parameters
    if M == 0:
        w_t = np.dot(X_t.T, t_train) / N_t
    else:
        w_t = np.dot(np.linalg.inv(np.dot(X_t.T, X_t)), np.dot(X_t.T, t_train))

    # prediction
    y_t = np.dot(X_t, w_t)

    # error
    err_train = err(y_t, t_train, N_t, 'training')

    ###### validation #####
    # get X
    if M == 0:
        X_v = np.ones(N_v).T
    else:
        ones_col = np.ones(N_v)  # get a row of ones
        X_v = np.array([ones_col, X_valid]).T

    if M > 1:
        for i in range(2, M + 1):
            X_v = np.c_[X_v, X_valid ** i]

    # prediction
    y_v = np.dot(X_v, w_t)

    # error
    err_valid = err(y_v, t_valid, N_v, 'validation')

    # error between validation and f_true
    err_true = err(t_valid, t_true, N_v, 'validation/true')

    # graphs
    plt.scatter(X_train, t_train, color='magenta', label='training set')
    plt.plot(X_train, y_t, color='red', label='model')
    plt.scatter(X_valid, t_valid, color='blue', label='validation set')

    # graph f_true
    x = np.linspace(0, 1, 100)
    f_true = np.sin(4 * np.pi * x)
    plt.plot(x, f_true, color='green', label='f_true')

    # print graph
    plt.legend(loc='upper right')
    plt.title('M = {0} (ZD, 400148647)'.format(M))
    plt.xlabel('x')
    plt.ylabel('f_M(x)')
    plt.show()

    return err_train, err_valid, err_true


########################################
# train model with regularization
########################################
def lin_reg_regular(M, l):
    # B is diagonal matrix of 2*lambda
    B = [2 * l] * N_t
    B = np.diag(B)

    # training
    # get X
    ones_col = np.ones(N_t)  # get a row of ones
    X_t = X_train.T

    if M > 1:
        for i in range(2, M + 1):
            X_t = np.c_[X_t, X_train ** i]

    # standardization
    X_t = sc.fit_transform(X_t)
    X_t = np.c_[ones_col, X_t]

    # get parameters
    w_t = np.dot(np.linalg.inv(np.dot(X_t.T, X_t) + N_t / 2 * B), np.dot(X_t.T, t_train))

    # prediction
    y_t = np.dot(X_t, w_t)

    # error
    err_train_regular = err(y_t, t_train, N_t, 'training')

    # validation
    # get X
    ones_col = np.ones(N_v)  # get a row of ones
    X_v = X_valid.T

    if M > 1:
        for i in range(2, M + 1):
            X_v = np.c_[X_v, X_valid ** i]

    # standardization
    X_v = sc.transform(X_v)
    X_v = np.c_[ones_col, X_v]

    # prediction
    y_v = np.dot(X_v, w_t)

    # error
    err_valid_regular = err(y_v, t_valid, N_v, 'validation')

    # graphs
    plt.scatter(X_train, t_train, color='magenta', label='training set')
    plt.plot(X_train, y_t, color='red', label='model')
    plt.scatter(X_valid, t_valid, color='blue', label='validation set')

    # graph f_true
    x = np.linspace(0, 1, 100)
    f_true = np.sin(4 * np.pi * x)
    plt.plot(x, f_true, color='green', label='f_true')

    # print graph
    plt.legend(loc='upper right')
    plt.title('M = {0}, lambda = {1} (ZD, 400148647)'.format(M, l))
    plt.xlabel('x')
    plt.ylabel('f_M(x)')
    plt.show()

    return err_train_regular, err_valid_regular


if __name__ == '__main__':
    main()
