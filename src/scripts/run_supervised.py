from src.supervised.svm import run_svm
from src.supervised.naive_bayes import run_naive_bayes
from src.supervised.random_forest import run_random_forest
from src.supervised.neural_networks import run_nn
if __name__ == "__main__":
    graph = 'cora'
    method = 'diff'
    # run_nn(graph, method)
    run_random_forest(graph, method)
    run_svm(graph, method)
    run_naive_bayes(graph, method)
