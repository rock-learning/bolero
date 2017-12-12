import numpy as np
from bolero.utils.ranking_svm import RankingSVM
from bolero.utils.validation import check_random_state
from nose.tools import assert_greater, assert_true


def test_ranking_linear_1d():
    random_state = check_random_state(0)

    n_dims = 1
    n_train = int(30 * np.sqrt(n_dims))
    n_test = 500
    n_iter = int(50000 * np.sqrt(n_dims))
    epsilon = 1.0
    X_train = np.linspace(0, 1, n_train)[:, np.newaxis]
    X_test = np.linspace(0, 1, n_test)[:, np.newaxis]
    ranking_svm = RankingSVM(n_iter, epsilon, random_state=random_state)
    ranking_svm.fit(X_train)
    y_train = ranking_svm.predict(X_train)
    y_test = ranking_svm.predict(X_test)
    assert_true(np.all(y_train[1:] < y_train[:-1]))
    assert_true(np.all(y_test[1:] < y_test[:-1]))


def test_ranking_dist_40d():
    random_state = check_random_state(0)

    n_dims = 40
    n_train = int(70 * np.sqrt(n_dims))
    n_test = 500
    n_iter = int(50000 * np.sqrt(n_dims))
    epsilon = 1.0

    def generate_data(n_samples, n_dims, random_state):
        """Rank data by distance to 0.5."""
        X = random_state.rand(n_samples, n_dims)
        distances = np.sum((0.5 - X) ** 2, axis=1)
        X = X[np.argsort(distances)]
        return X, np.sort(distances)

    X_train, dist_train = generate_data(n_train, n_dims, random_state)
    X_test, dist_test = generate_data(n_test, n_dims, random_state)

    ranking_svm = RankingSVM(n_iter, epsilon, random_state=random_state)
    ranking_svm.fit(X_train)
    y_train = ranking_svm.predict(X_train)
    y_test = ranking_svm.predict(X_test)

    # Ranking is 80% correct between samples of rank distance 100
    assert_greater(np.count_nonzero(y_train[100:] < y_train[:-100]),
                   0.8 * (n_train - 100))
    assert_greater(np.count_nonzero(y_test[100:] < y_test[:-100]),
                   0.8 * (n_test - 100))


def test_rank_archimedean_spiral():
    def archimedean_spiral(n_steps=100, max_radius=1.0, turns=4.0):
        r = np.linspace(0.0, max_radius, n_steps)
        angle = r * 2.0 * np.pi * turns / max_radius
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        return np.hstack((x[:, np.newaxis], y[:, np.newaxis])), r

    X_train, r_train = archimedean_spiral(n_steps=100)
    X_test, r_test = archimedean_spiral(n_steps=1000, max_radius=1.1)

    rsvm = RankingSVM(random_state=0)
    rsvm.fit(X_train)

    y_train = rsvm.predict(X_train)
    y_test = rsvm.predict(X_test)
    assert_true(np.all(y_train[1:] < y_train[:-1]))
    assert_greater(np.count_nonzero(y_test[1:] < y_test[:-1]), 970)
