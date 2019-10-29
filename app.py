from flask import Flask, request
import numpy as np
from IMDBContentBased import IMDBContentBased
import evaluate_solutions as ev
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)
recommender = IMDBContentBased()
recommender.profile_learning()


@app.route('/max', methods=['GET', 'POST'])
def get_max():
    max = np.max(recommender.X, axis=0)
    return {'response': max.values.tolist()}


@app.route('/min', methods=['GET', 'POST'])
def get_min():
    min = np.min(recommender.X, axis=0)
    return {'response': min.values.tolist()}


@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    message = request.get_json(silent=True)
    solucoes = message["solucoes"]
    solucoes = np.array(solucoes[0])
    y = recommender.profile.predict(solucoes)
    return {'response': [y]}


@app.route('/filtering', methods=['GET', 'POST'])
def filtering():
    message = request.get_json(silent=True)
    solucoes = message["solucoes"]
    sim = cosine_similarity(solucoes, recommender.X.values)
    pop_index = sim.argmax(axis=1)
    res = recommender.X.iloc[pop_index, :].drop_duplicates()
    res.to_csv('./datasets/Recomendacoes.csv')
    y = ev.evaluate_solutions(res, recommender.X_train, recommender)
    objs = pd.DataFrame()
    objs['Acurácia'] = y[:, 0]
    objs['Diversidade'] = y[:, 1]
    objs['Novidade'] = y[:, 2]
    objs.index = res.index
    print(objs)
    return {'response': []}


@app.route('/n-variables', methods=['GET', 'POST'])
def n_variables():
    return {'response': [recommender.X_train.shape[1]]}


@app.route('/evaluate-solutions', methods=['GET', 'POST'])
def evaluate_solutions():
    message = request.get_json(silent=True)
    solucoes = message["solucoes"]
    solucoes = np.array(solucoes)
    y = ev.evaluate_solutions(solucoes, recommender.X_train, recommender)
    return {'response': y.tolist()}


if __name__ == '__main__':
    app.run()
