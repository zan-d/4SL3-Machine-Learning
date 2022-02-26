import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sc = StandardScaler()

dataset = pd.read_csv(
    "C:/Users/School/Documents/University/2021-2022/First Semester/4SL3 - Machine Learning/data_banknote_authentication.txt",
    header=None)
X_data = dataset.iloc[:, :-1].values
train = dataset.iloc[:, -1].values

rand_seed = 8647  # using last 4 digits of student number

# split into train, test and validation set and standardize
X_train, X_test, t_train, t_test = train_test_split(X_data, train, test_size=0.2, random_state=rand_seed)
X_train, X_valid, t_train, t_valid = train_test_split(X_train, t_train, test_size=0.25, random_state=rand_seed)
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)
X_test = sc.transform(X_test)


def main():
    # TEST TO VERIFY neural_net
    # x = np.array([1, -1, 2])
    # t = 1
    # W1 = np.array([[0.5, -1, -0.5, 1.5], [1, 0.2, -0.4, -1]])
    # W2 = np.array([[-1.5, 1, 0.4], [-1, -2, 0.5]])
    # W3 = np.array([-2, 1, -1])
    # y, W1, W2, W3 = neural_net(x, t, W1, W2, W3)

    max1 = 5
    max2 = 5
    best_n1 = 0
    best_n2 = 0
    best_n_ce = 100000  # initializa as a large number
    for n1 in range(2, max1):
        for n2 in range(2, max2):
            W1_best = np.zeros((n1, 5))
            W2_best = np.zeros((n2, n1 + 1))
            W3_best = np.zeros((1, n2 + 1))
            best_ce = 100000  # initializa as a large number
            for w in range(1, 6):  # 5 different starting weights
                print("n1: {0}  n2: {1}  start w: {2}".format(n1, n2, w))
                epoch = []
                err_valid = []
                err_train = []

                # initialize weights
                W1 = np.full((n1, 5), w / 5)  # 4 features so num of columns is 5
                W2 = np.full((n2, n1 + 1), w / 5)
                W3 = np.full((1, n2 + 1), w / 5)

                for i in range(100):  # epoch
                    epoch.append(i)

                    # shuffle rows each time
                    temp_arr = np.c_[X_train, t_train]
                    np.random.shuffle(temp_arr)
                    X = temp_arr[:, :-1]
                    t = temp_arr[:, -1]
                    y_pred = []
                    for j in range(len(X_train)):  # iterate through every sample
                        # update the weights each iteration
                        y, W1, W2, W3 = neural_net(X[j, :], t[j], W1, W2, W3)
                        y_pred.append(y)
                    y_pred = np.array(y_pred)
                    # training cross entropy
                    ce_loss_train = l_ce(t, y_pred)
                    err_train.append(ce_loss_train)

                    # validation cross entropy
                    y_valid = []
                    for j in range(len(X_valid)):  # iterate through every sample
                        # use trained weights each iteration
                        y, W1_temp, W2_temp, W3_temp = neural_net(X_valid[j, :], t_valid[j], W1, W2, W3)
                        y_valid.append(y)
                    y_valid = np.array(y_valid)
                    ce_loss_valid = l_ce(t_valid, y_valid)
                    err_valid.append(ce_loss_valid)
                    y_class = np.where(y_valid > 0.5, 1, 0)
                    misclass_rate = 1 - accuracy_score(t_valid, y_class)

                    # check if this is the smallest validation cross-entropy
                    if ce_loss_valid < best_ce:
                        best_ce = ce_loss_valid
                        best_misclass_rate = misclass_rate
                        W1_best = W1
                        W2_best = W2
                        W3_best = W3

                # calc validation, training error
                # print("err_train: {0}".format(err_train))
                # print("err_valid: {0}".format(err_valid))
                print("best validation cross-entropy: {0}".format(best_ce))
                print("misclassification rate: {0}".format(best_misclass_rate))
                print("best W1: {0}".format(W1_best))
                print("best W2: {0}".format(W2_best))
                print("best W3: {0}".format(W3_best))

                if w == 5:  # graph one of the weight assignments
                    # plot learning curve
                    plt.plot(epoch, err_train, color='red', label='training')
                    plt.plot(epoch, err_valid, color='green', label='validation')
                    plt.title('learning curve n1: {0} n2: {1} (ZD, 400148647)'.format(n1, n2))
                    plt.legend(loc='upper right')
                    plt.xlabel('epoch')
                    plt.ylabel('cross entropy loss')
                    plt.show()
                print("\n")
            # check if this is the smallest validation cross-entropy for all n1, n2 so far
            if best_ce < best_n_ce:
                best_n1 = n1
                best_n2 = n2
                best_n_ce = best_ce
                W1_n_best = W1_best
                W2_n_best = W2_best
                W3_n_best = W3_best
    print("best n1: {0}".format(best_n1))
    print("best n2: {0}".format(best_n2))
    print("best overall cross-entropy: {0}".format(best_n_ce))
    print("W1 overall best: {0}".format(W1_n_best))
    print("W2 overall best: {0}".format(W2_n_best))
    print("W3 overall best: {0}".format(W3_n_best))

    # for the chosen n1, n2
    # misclassification on test set
    y_test = []
    for j in range(len(X_test)):  # iterate through every sample
        # use trained weights each iteration
        y, W1, W2, W3 = neural_net(X_test[j, :], t_test[j], W1_n_best, W2_n_best, W3_n_best)
        y_test.append(y)
    y_test = np.array(y_test)
    y_class = np.where(y_test > 0.5, 1, 0)
    misclass_rate = 1 - accuracy_score(t_test, y_class)
    print("misclassification rate: {0}".format(misclass_rate))


def l_ce(t, y_pred):
    y_pred[y_pred == 1] = 0.99999999999999999
    y_pred[y_pred == 0] = 0.00000000000000001
    return np.dot(-1 * t, np.log(y_pred)) - np.dot((1 - t), np.log(1 - y_pred))


def relu(h):
    h[h < 0] = 0
    return h


def d_relu(g):
    # relu derivative
    g[g >= 0] = 1
    g[g < 0] = 0
    return g


def new_w(w, w_J):
    alpha = 0.05
    w_new = w - np.dot(alpha, w_J)
    return w_new


def neural_net(x, t, W1, W2, W3):
    # fwd
    z1 = np.dot(W1, np.r_[1, x.T][:, None])
    # print("z1: {0}\n".format(z1))
    h1 = relu(z1.copy())
    # print("h1: {0}\n".format(h1))
    z2 = np.dot(W2, np.c_[1, h1.T].T)
    # print("z2: {0}\n".format(z2))
    h2 = relu(z2.copy())
    # print("h2: {0}\n".format(h2))
    z3 = np.dot(W3, np.c_[1, h2.T].T)
    # print("z3: {0}\n".format(z3))
    y = 1 / (1 + np.exp(-z3))
    # print("y: {0}\n".format(y))

    # bwd
    dj_dz3 = -t + y  # cross-entropy loss
    # print("dj_dz3: {0}\n".format(dj_dz3))
    w3_J = dj_dz3 * np.c_[1, h2.T]
    # print("w3_J: {0}\n".format(w3_J))
    z2_J = np.multiply(d_relu(z2), W3[:, 1:].T * dj_dz3)  # element wise multiplication
    # print("z2_J: {0}\n".format(z2_J))
    w2_J = np.dot(z2_J, np.c_[1, h1.T])
    # print("w2_J: {0}\n".format(w2_J))
    z1_J = np.multiply(d_relu(z1), np.dot(W2[:, 1:].T, z2_J))
    # print("z1_J: {0}\n".format(z1_J))
    w1_J = np.dot(z1_J, np.r_[1, x.T][None, :])
    # print("w1_J: {0}\n".format(w1_J))

    # new weights
    W1_new = new_w(W1, w1_J)
    W2_new = new_w(W2, w2_J)
    W3_new = new_w(W3, w3_J)

    return y.item(0), W1_new, W2_new, W3_new


if __name__ == '__main__':
    main()
