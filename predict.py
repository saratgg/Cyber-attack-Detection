import numpy as np

def convert_to_numeric(values):
    return np.array(values, dtype=np.float64).reshape(1, -1)

import pickle

def predict_decision_tree(values_numeric):
    decision_tree_model = pickle.load(open('DecisionTreeClassifier.pkl', 'rb'))
    result_decision_tree = decision_tree_model.predict(values_numeric)
    return result_decision_tree

def predict_knn(values_numeric):
    knn_model = pickle.load(open('KNeighborsClassifier.pkl', 'rb'))
    result_knn = knn_model.predict(values_numeric)
    return result_knn

def predict_logistic_regression(values_numeric):
    logistic_regression_model = pickle.load(open('LogisticRegression.pkl', 'rb'))
    result_logistic_regression = logistic_regression_model.predict(values_numeric)
    return result_logistic_regression


def combine_predictions(result_decision_tree, result_knn, result_logistic_regression):
    if result_decision_tree == 1 or result_knn == 1 or result_logistic_regression == 1:
        return "Normal"
    else:
        return "Anomaly"


def predict(values):
    values_numeric = convert_to_numeric(values)
    result_decision_tree = predict_decision_tree(values_numeric)
    result_knn = predict_knn(values_numeric)
    result_logistic_regression = predict_logistic_regression(values_numeric)

    final_result = combine_predictions(result_decision_tree, result_knn, result_logistic_regression)
    return final_result
