from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def benchmark(clf, summaries, genres):
    """
    Benchmark a model for its performance in predicting the genre of a summary
    :param clf:
    :param summaries:
    :param genres:
    :return:
    """
    kf = KFold(len(genres), n_folds=10)
    f_score = 0.0
    f_train_time = 0.0
    f_test_time = 0.0
    for train_index, test_index in kf:
        X_train, X_test = summaries[train_index], summaries[test_index]
        y_train, y_test = genres[train_index], genres[test_index]
        vectorizer = TfidfVectorizer(sublinear_tf=True, norm=None, smooth_idf=True, ngram_range=(1, 2), lowercase=True)
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)
        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)
        score = metrics.accuracy_score(y_test, pred)
        print("accuracy:   %0.3f" % score)
        print()
        clf_descr = str(clf).split('(')[0]
        f_test_time += test_time
        f_train_time += train_time
        f_score += score
    return clf_descr, f_score/10, f_train_time/10, f_test_time/10


def plot_benchmarks(summaries, genres):
    """
    Plot benchmarks for various common models and see how they perform
    :param summaries:
    :param genres:
    :return:
    """
    results = []

    # Train SGD with Elastic Net penalty
    print('=' * 80)
    print("Elastic-Net penalty")
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"), summaries, genres))

    # Train NearestCentroid without threshold
    print('=' * 80)
    print("NearestCentroid (aka Rocchio context_identification)")
    results.append(benchmark(NearestCentroid(), summaries, genres))

    print('=' * 80)
    print("Logistic Regression")
    results.append(benchmark(linear_model.LogisticRegression(n_jobs=-1, solver='liblinear', class_weight='balanced',
                                                             penalty='l2', C=1, fit_intercept=False), summaries, genres))

    # Train sparse Naive Bayes classifiers
    print('=' * 80)
    print("Naive Bayes")
    results.append(benchmark(MultinomialNB(alpha=.01), summaries, genres))
    results.append(benchmark(BernoulliNB(alpha=.01), summaries, genres))

    # make some plots

    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='r')
    plt.barh(indices + .3, training_time, .2, label="training time", color='g')
    plt.barh(indices + .6, test_time, .2, label="test time", color='b')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.show()

