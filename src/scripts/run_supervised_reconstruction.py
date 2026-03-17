from src.supervised_reconstruction.svm import run_svm
from src.supervised_reconstruction.naive_bayes import run_naive_bayes
from src.supervised_reconstruction.random_forest import run_random_forest
from src.supervised_reconstruction.neural_networks import run_nn
if __name__ == "__main__":
    graph = 'karate_club_graph'
    method = 'diff'
    run_nn(graph, method)
    run_random_forest(graph, method)
    run_svm(graph, method)
    run_naive_bayes(graph, method)
