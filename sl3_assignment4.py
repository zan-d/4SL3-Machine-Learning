import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

dataset = pd.read_csv(
    "C:/Users/School/Documents/University/2021-2022/First Semester/4SL3 - Machine Learning/spambase.data", header=None)
X_data = dataset.iloc[:, :-1].values
train = dataset.iloc[:, -1].values

rand_seed = 8647  # using last 4 digits of student number
X_train, X_valid, t_train, t_valid = train_test_split(X_data, train, test_size=0.33, random_state=rand_seed)


def main():
    # predictors
    predictors = range(50, 2501, 50)  ####### CHANGE THESE AFTER TEST!

    # cross validation for leaves between 2 and 400
    cross_val_errors = []  # for finding best number of leaves
    x_tree = []
    for max_leaves in range(2, 401):
        err_cross_val, err_valid = decision_tree(max_leaves)
        x_tree.append(max_leaves)
        cross_val_errors.append(err_cross_val)
        print(max_leaves)
    min_err = np.argmin(cross_val_errors) + 2  # since the 0th item in array corresponds to 2 leaves
    print("decision tree classifier min error, number of leaves: {0}".format(min_err))

    # graph cross validation
    plt.plot(x_tree, cross_val_errors)
    plt.title('decision tree classifier (ZD, 400148647)')
    plt.xlabel('number of leaves')
    plt.ylabel('cross validation error')
    plt.show()

    print("start decision tree...\n")
    # decision tree
    y_decision_tree = []  # for graph
    for i in predictors:
        print(i)
        err_cross_val, err_valid = decision_tree(49)
        y_decision_tree.append(err_valid)

    print("start bagging...\n")
    # 50 bagging classifiers
    y_bagging = []  # for graph
    for i in predictors:
        print(i)
        y_bagging.append(bagging(i))

    print("start random forest...\n")
    # 50 random forest
    y_rand_forest = []
    for i in predictors:
        print(i)
        y_rand_forest.append(rand_forest(i))

    print("start adaboost stump...\n")
    # adaboost stump
    y_adaboost_stump = []
    for i in predictors:
        print(i)
        y_adaboost_stump.append(adaboost_stump(i))

    print("start adaboost with restrictions...\n")
    # adaboost restrict
    y_adaboost_restrict = []
    for i in predictors:
        print(i)
        y_adaboost_restrict.append(adaboost_tree_restrict(i))

    print("start adaboost no restrictions...\n")
    # adaboost no restrictions
    y_no_restrict = []
    for i in predictors:
        print(i)
        y_no_restrict.append(adaboost_no_restrict(i))

    print("start graphs...\n")
    # graph
    plt.plot(predictors, y_decision_tree, color='black', label='decision tree')
    plt.plot(predictors, y_bagging, color='red', label='bagging')
    plt.plot(predictors, y_rand_forest, color='blue', label='random forest')
    plt.plot(predictors, y_adaboost_stump, color='green', label='adaboost stump')
    plt.plot(predictors, y_adaboost_restrict, color='orange', label='adaboost restrict')
    plt.plot(predictors, y_no_restrict, color='purple', label='adaboost no restrict')
    plt.title('ensemble methods (ZD, 400148647)')
    plt.legend(loc='upper right')
    plt.xlabel('# of predictors')
    plt.ylabel('test error')
    plt.show()


def decision_tree(max_leaves):
    clf = DecisionTreeClassifier(random_state=rand_seed, max_leaf_nodes=max_leaves).fit(X_train, t_train)
    scores = cross_val_score(clf, X_train, t_train, cv=3)
    err_cross_val = 1 - sum(scores) / len(scores)
    err_valid = 1 - clf.score(X_valid, t_valid)
    return err_cross_val, err_valid


def bagging(n):
    clf = BaggingClassifier(DecisionTreeClassifier(random_state=rand_seed), n_estimators=n, random_state=rand_seed).fit(X_train, t_train)
    err_valid = 1 - clf.score(X_valid, t_valid)
    return err_valid


def rand_forest(n):
    clf = RandomForestClassifier(n_estimators=n, random_state=rand_seed).fit(X_train, t_train)
    err_valid = 1 - clf.score(X_valid, t_valid)
    return err_valid


def adaboost_stump(n):
    clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=rand_seed, max_depth=1), n_estimators=n, random_state=rand_seed).fit(X_train, t_train)
    err_valid = 1 - clf.score(X_valid, t_valid)
    return err_valid


def adaboost_tree_restrict(n):
    clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=rand_seed, max_leaf_nodes=10), n_estimators=n,
                             random_state=rand_seed).fit(X_train, t_train)
    err_valid = 1 - clf.score(X_valid, t_valid)
    return err_valid


def adaboost_no_restrict(n):
    clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=rand_seed), n_estimators=n,
                             random_state=rand_seed).fit(X_train, t_train)
    err_valid = 1 - clf.score(X_valid, t_valid)
    return err_valid


if __name__ == '__main__':
    main()
