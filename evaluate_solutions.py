import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_solutions(solutions, data, recommender):
    y = np.zeros((solutions.shape[0], 3))
    y[:, 0] = -recommender.profile.predict(solutions) * 0.1
    y[:, 1] = get_diversity(solutions)
    y[:, 2] = get_novelty(solutions, data)
    # print('solutions:', solutions[0:5])
    # print('acurácia:', y[:, 0])
    # print('diversidade:', y[:, 1])
    print('novelty:', y[:, 2])
    return y


def get_diversity(solutions):
    sim = cosine_similarity(solutions, solutions)
    np.fill_diagonal(sim, 0)
    return -(1 - sim).sum(axis=1) * 1/(solutions.shape[0] - 1)


def get_novelty(solutions, data):
    sim = cosine_similarity(solutions, data)
    return -(1-sim).min(axis=1)
