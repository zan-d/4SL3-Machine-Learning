import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

rand_seed = 8647  # using last 4 digits of student number

X_data, train = load_boston(return_X_y=True)


def main():
    X_train, X_valid, t_train, t_valid = train_test_split(X_data, train, test_size=0.2, random_state=rand_seed)

    N_orig_t = len(X_train)
    N_orig_v = len(X_valid)

    K_max = 13  # max number of steps

    # list of errors to be graphed
    errors_graph = []  # format k, cross valid, valid, basis cross valid, basis valid

    X_v_subset = np.ones(N_orig_v).T  # subset of X_valid for testing
    S = np.ones(N_orig_t).T  # chosen features
    for k in range(1, K_max + 1):
        temp_err_graph = [k]
        print("K: {0}".format(k))
        f_error = [0] * (K_max + 1 - k)  # average error for each feature
        for f in range(K_max + 1 - k):  # feature number
            S_f = X_train[:, f]  # get f-th column
            S_f = np.c_[S, S_f]  # append it to the current S
            f_error[f] = cross_valid(5, S_f, t_train)  # get cross-validation error

        print("average cross-validation error for each fold:")
        print(f_error)
        # get feature with the smallest error
        f_min_error_idx = np.argmin(f_error)
        f_min_error = f_error[f_min_error_idx]  # error to be graphed
        temp_err_graph.append(f_min_error)
        print("location of feature with smallest error: {0}".format(f_min_error_idx))
        S = np.c_[S, X_train[:, f_min_error_idx]]  # add feature with lowest error to S
        print("shape of implemented feature: {0}".format(S.shape))
        X_train = np.delete(X_train, f_min_error_idx, 1)
        print("shape of not implemented feature: {0}".format(X_train.shape))

        # find error using X_valid
        X_v_subset = np.c_[X_v_subset, X_valid[:, f_min_error_idx]]
        X_valid = np.delete(X_valid, f_min_error_idx, 1)
        y_train, y_valid, w = lin_reg(S, t_train, X_v_subset)
        print("parameters w for subset: {0}".format(w))

        # calculate error
        S_err = err(y_valid, t_valid, N_orig_v, 'validation')  # error to be graphed
        temp_err_graph.append(S_err)
        basis_cross_valid_err, basis_valid_err, w_basis = lin_reg_basis(S_err, S, t_train, X_v_subset,
                                                                        t_valid)  # both errors need to be graphed
        print("parameters w for basis subset: {0}".format(w_basis))
        temp_err_graph.append(basis_cross_valid_err)
        temp_err_graph.append(basis_valid_err)
        print("basis expansion cross validation error: {0}".format(basis_cross_valid_err))
        print("basis expansion test error: {0}".format(basis_valid_err))
        errors_graph.append(temp_err_graph)
        print("\n")
    errors_graph = np.stack(errors_graph, axis=0)
    print("final errors: {0}".format(errors_graph))  # format k, cross valid, valid, basis cross valid, basis valid

    # graph
    plt.scatter(errors_graph[:, 0], errors_graph[:, 1], color='magenta', label='cross valid')
    plt.scatter(errors_graph[:, 0], errors_graph[:, 2], color='green', label='valid')
    plt.scatter(errors_graph[:, 0], errors_graph[:, 3], color='blue', label='basis cross valid')
    plt.scatter(errors_graph[:, 0], errors_graph[:, 4], color='red', label='basis valid')

    # print graph
    plt.legend(loc='best')
    plt.title('Errors (ZD, 400148647)')
    plt.xlabel('k')
    plt.ylabel('error')
    plt.show()
    print("\n")


def lin_reg_basis(S_err, S, t, X_v_subset, t_valid):
    print("input dimensions: S = {0}, X = {1}".format(S.shape, X_v_subset.shape))
    # two functions for basis expansion
    X_a = np.c_[S, S[:, 1:] ** 2]
    X_a_valid = np.c_[X_v_subset, X_v_subset[:, 1:] ** 2]

    X_b = np.concatenate(
        (S, S[:, 1:] ** 2, S[:, 1:] ** 3, S[:, 1:] ** 4, S[:, 1:] ** 5, S[:, 1:] ** 6, S[:, 1:] ** 8, S[:, 1:] ** 10),
        axis=1)
    X_b_valid = np.concatenate((X_v_subset, X_v_subset[:, 1:] ** 2, X_v_subset[:, 1:] ** 3, X_v_subset[:, 1:] ** 4,
                                X_v_subset[:, 1:] ** 5, X_v_subset[:, 1:] ** 6, X_v_subset[:, 1:] ** 8,
                                X_v_subset[:, 1:] ** 10),
                               axis=1)

    # cross validation and find validation error
    a_err = cross_valid(6, X_a, t)
    try:
        b_err = cross_valid(6, X_b, t)
    except:
        print("\nERROR: singular matrix, use different basis expansion")
        X_b = np.concatenate((S, S[:, 1:] ** 6), axis=1)
        X_b_valid = np.concatenate((X_v_subset, X_v_subset[:, 1:] ** 6),
                                   axis=1)
        b_err = cross_valid(6, X_b, t)

    print("a_err: {0}".format(a_err))
    print("b_err: {0}".format(b_err))

    # check if errors from basis expansion is smaller than S_err
    if a_err < S_err:
        # linear regression with a
        y_train, y_valid, w = lin_reg(X_a, t, X_a_valid)
        return a_err, err(y_valid, t_valid, len(X_a_valid), 'validation'), w
    elif b_err < S_err:
        # linear regression with b
        y_train, y_valid, w = lin_reg(X_b, t, X_b_valid)
        return b_err, err(y_valid, t_valid, len(X_b_valid), 'validation'), w
    else:
        print("WARNING: neither basis expansion functions produce an error smaller than S_err\n")
        if a_err < b_err:
            y_train, y_valid, w = lin_reg(X_a, t, X_a_valid)
            return a_err, err(y_valid, t_valid, len(X_a_valid), 'validation'), w
        else:
            y_train, y_valid, w = lin_reg(X_b, t, X_b_valid)
            return b_err, err(y_valid, t_valid, len(X_b_valid), 'validation'), w


def cross_valid(split, S_f, t):
    fold_err = [0] * split  # error for each fold

    kf = KFold(n_splits=split, random_state=rand_seed, shuffle=True)
    i = 0

    for train_index, test_index in kf.split(S_f):
        X_train1, X_valid1 = S_f[train_index], S_f[test_index]
        t_train1, t_valid1 = t[train_index], t[test_index]

        # get predictor
        N_v = len(X_valid1)
        y_train, y_valid, w = lin_reg(X_train1, t_train1, X_valid1)

        # calculate error, need validation error for cross-validation error calculation
        fold_err[i] = err(y_valid, t_valid1, N_v, 'validation')
        i += 1

    # get average error for each fold
    return sum(fold_err) / len(fold_err)


def err(y_vector, t_vector, N, err_type):
    diff_t = np.subtract(t_vector, y_vector)
    err_t = np.dot(diff_t.T, diff_t) / N
    # RMSE_t = np.sqrt(err_t)
    # print("{0} error: {1}".format(err_type, err_t))
    # print("root mean square {0} error: {1}".format(err_type, RMSE_t))
    return err_t


def lin_reg(X_train_in, t_train_in, X_valid_in):
    # training #####
    # get X
    X_t = X_train_in

    # get parameters
    w_t = np.dot(np.linalg.inv(np.dot(X_t.T, X_t)), np.dot(X_t.T, t_train_in))

    # prediction
    y_t = np.dot(X_t, w_t)

    # validation #####
    # get X
    X_v = X_valid_in

    # prediction
    y_v = np.dot(X_v, w_t)

    return y_t, y_v, w_t


if __name__ == '__main__':
    main()
