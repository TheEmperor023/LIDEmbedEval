from src.supervised_link_prediction.svm import run_svm
from src.supervised_link_prediction.naive_bayes import run_naive_bayes
from src.supervised_link_prediction.random_forest import run_random_forest
from src.supervised_link_prediction.neural_networks import run_nn
if __name__ == "__main__":
    graph = 'cora'
    method = 'diff'
    dataset_path = '/Users/vukdermanovic/grasp/graspe/data/datasets/1_to_n_sampled_datasets_n=4_p=0.5/link_pred_datasets/'
    report_path = '/Users/vukdermanovic/Faks/MasterRad/LIDEmbedEval/reports/Link Prediction/Supervised Reports/1_to_n_sampled_datasets_n=4_p=0.5/'
    run_nn(graph, method, dataset_path, report_path)
    run_random_forest(graph, method, dataset_path, report_path)
    run_svm(graph, method, dataset_path, report_path)
    run_naive_bayes(graph, method, dataset_path, report_path)
