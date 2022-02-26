import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

rand_seed = 8647  # using last 4 digits of student number


def main():
    X_data, train = load_breast_cancer(return_X_y=True)
    X_train, X_valid, t_train, t_valid = train_test_split(X_data, train, test_size=0.2, random_state=rand_seed)
    X_train = sc.fit_transform(X_train)
    X_valid = sc.transform(X_valid)
    N_t = len(X_train)
    N_v = len(X_valid)

    # logistic regression
    logistic_reg(X_train, X_valid, t_train, t_valid, N_t, N_v)
    logistic_reg_sklearn(X_train, t_train, X_valid, t_valid)

    # get classifier for k-nearest neighbours
    knn_c, knn_sk_c = cross_valid(8, X_train, t_train)

    # k-nearest neighbours
    classifier, err, err_all = k_nn(0, knn_c, X_train, t_train, X_valid, t_valid)
    classifier_sk, err_sk, err_all_sk = k_nn_sklearn(0, knn_sk_c, X_train, t_train, X_valid, t_valid)
    print("k_nn error: {0} classifier: {1}".format(err, classifier))
    print("k_nn_sklearn error: {0} classifier: {1}".format(err_sk, classifier_sk))


def logistic_reg(X_t, X_v, t_t, t_v, N_t, N_v):
    new_col = np.ones(N_t)
    X_t = np.insert(X_t, 0, new_col, axis=1)  # add the dummy column
    new_col = np.ones(N_v)
    X_v = np.insert(X_v, 0, new_col, axis=1)  # add the dummy column
    alpha = 0.5
    w = np.ones(31)  # initial w
    z_t = np.zeros(N_t)
    z_v = np.zeros(N_v)
    y_t = np.zeros(N_t)
    y_v = np.zeros(N_v)
    IT = 200  # number of iterations of GD
    cost_t = np.zeros(IT)
    cost_v = np.zeros(IT)
    for n in range(IT):
        # compute gradient
        z_t = np.dot(X_t, w)
        z_v = np.dot(X_v, w)
        y_t = 1 / (1 + np.exp(-z_t))
        diff = y_t - t_t
        gr = np.dot(X_t.T, diff) / N_t

        # update w
        w = w - alpha * gr

    # compute classification error
    for i in range(N_v):
        if z_v[i] >= 0:
            y_v[i] = 1
    u = y_v - t_v
    err = np.count_nonzero(u) / N_v  # misclassification rate
    f1 = f1_score(t_v, y_v, average=None)

    print("logistic_reg misclassification error: {0} f1 score: {1}".format(err, f1))
    print("logistic_reg vector of parameters: {0}".format(w))

    # PR curve
    theta = np.sort(z_v)
    precision = np.zeros(N_v)
    recall = np.zeros(N_v)
    idx = 0
    for t in theta:
        y_pr = np.zeros(N_v)
        for i in range(N_v):
            if z_v[i] >= t:
                y_pr[i] = 1
        precision[idx] = precision_score(t_v, y_pr, average='binary', zero_division=0)
        recall[idx] = recall_score(t_v, y_pr, average='binary')
        idx += 1

    plt.plot(recall, precision)
    plt.title('PR Curve (ZD, 400148647)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.show()


def logistic_reg_sklearn(X_t, t_t, X_v, t_v):
    clf = LogisticRegression(random_state=rand_seed).fit(X_t, t_t)  # fit on train
    y_pred = clf.predict(X_v)  # predict on verification
    score = clf.score(X_v, t_v)  # correct classification rate
    err = 1 - score
    f1 = f1_score(t_v, y_pred, average=None)
    print("logistic_reg_sklearn misclassification error: {0} f1 score: {1}".format(err, f1))
    # print("logistic_reg_sklearn vector of parameters: {0}".format(w))

    # PR curve
    y_scores = clf.decision_function(X_v)
    precision, recall, thresholds = precision_recall_curve(t_v, y_scores)

    plt.plot(recall, precision)
    plt.title('PR Curve sklearn (ZD, 400148647)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.show()


def k_nn(r, K, X_t, t_t, X_v, t_v):
    N_t = len(X_t)
    N_v = len(X_v)

    # compute distances between each valid point and each training point
    dist = np.zeros((N_v, N_t))
    for i in range(N_v):
        for j in range(N_t):
            diff = X_v[i] - X_t[j]
            dist[i, j] = np.dot(diff, diff.T)  # diff * diff.T  = this is squared euclidean distance

    ind = np.argsort(dist, axis=1)
    y = np.zeros((K, N_v))
    err = np.zeros(K)
    for k in range(K):  # k+1 nearest neighbour
        for i in range(N_v):
            # compute prediction for x_i
            for s in range(k):
                y[k, i] += t_t[ind[i, s]]
            y[k, i] /= (k + 1)
        # compute error for this k
        z = np.subtract(t_v, y[k, :])
        err[k] = np.dot(z, z) / N_v

    k_best = np.argmin(err)

    if r == 0:  # only get data from the best classifier determined by cross-validation
        k_best = K - 1

    # print("k_nn misclassification error: {0} classifier: {1}".format(err[k_best], k_best+1))
    print("k_nn err: {0}".format(err))
    return k_best+1, err[k_best], err


def k_nn_sklearn(r, K, X_t, t_t, X_v, t_v):
    err = np.zeros(K)
    for k in range(1, K+1):
        neigh = knn(n_neighbors=k).fit(X_t, t_t)
        y = neigh.predict(X_v)
        score = neigh.score(X_v, t_v)
        err[k-1] = 1 - score

    k_best = np.argmin(err)

    if r == 0:  # only get data from the best classifier determined by cross-validation
        k_best = K - 1

    # print("k_nn_sklearn misclassification error: {0} classifier: {1}".format(err[k_best], k_best+1))
    print("k_nn_sklearn err: {0}".format(err))
    return k_best+1, err[k_best], err


def cross_valid(split, X, t):
    fold_err = [0] * split  # error for each fold
    fold_err_sk = [0] * split  # error for each fold

    kf = KFold(n_splits=split, random_state=rand_seed, shuffle=True)
    i = 0
    for train_index, test_index in kf.split(X):
        X_train1, X_test1 = X[train_index], X[test_index]
        t_train1, t_test1 = t[train_index], t[test_index]

        # get classifier and error
        print("round: {0}".format(i))
        a, b, fold_err[i] = k_nn(1, 5, X_train1, t_train1, X_test1, t_test1)
        a, b, fold_err_sk[i] = k_nn_sklearn(1, 5, X_train1, t_train1, X_test1, t_test1)
        print("\n")

        i += 1

    # avg error for each k to determine the best k
    err = np.asarray(fold_err)
    err_sk = np.asarray(fold_err_sk)
    err_avg = np.mean(err, axis=0)
    err_avg_sk = np.mean(err_sk, axis=0)
    k_best = np.argmin(err_avg)
    k_best_sk = np.argmin(err_avg_sk)

    print("k_nn avg classifier: {0} avg error: {1}".format(k_best+1, err_avg[k_best]))
    print("k_nn avg classifier: {0} avg error: {1}".format(k_best_sk+1, err_avg_sk[k_best_sk]))
    print("\n")

    return k_best+1, k_best_sk+1


if __name__ == '__main__':
    main()
