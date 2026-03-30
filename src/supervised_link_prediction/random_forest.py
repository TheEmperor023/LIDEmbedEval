from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import numpy as np
from  pathlib import Path
from src.utils.make_random_forest_report import make_forest_report
import src.utils.load_dataset as loads
import os

def run_random_forest(graph, method, dataset_path, reports_path):
    # graph = 'florentine_families_graph'
    # method = 'diff'
    n2v_avg_accuracy = []
    n2v_avg_precision = []
    n2v_avg_recall = []
    n2v_avg_f1 = []
    n2v_avg_fpr = []
    lidn2v_avg_accuracy = []
    lidn2v_avg_precision = []
    lidn2v_avg_recall = []
    lidn2v_avg_f1 = []
    lidn2v_avg_fpr = []
    lidn2v_ext_avg_accuracy = []
    lidn2v_ext_avg_precision = []
    lidn2v_ext_avg_recall = []
    lidn2v_ext_avg_f1 = []
    lidn2v_ext_avg_fpr = []
    path_name = dataset_path + graph +'/train'
    path = Path(path_name)
    for f in path.iterdir():
        name, extension = os.path.splitext(f.name)
        if method == 'diff':
            X_train, y_train = loads.load_from_file_diff(f)
            X_test, y_test = loads.load_from_file_diff(dataset_path + graph +'/test/'+ name + ".testset")
        elif method == 'concat':
            X_train, y_train = loads.load_from_file(f)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        print("Embedding file:" + str(f.name))
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 100],
            'min_samples_split': [2, 5, 10]
        }



        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


        grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)


        best_rf = grid_search.best_estimator_


        y_pred = best_rf.predict(X_test)


        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        if ('n2v-' in f.name):
            n2v_avg_accuracy.append(acc)
            n2v_avg_precision.append(prec)
            n2v_avg_recall.append(rec)
            n2v_avg_f1.append(f1)
            n2v_avg_fpr.append(fpr)
        elif ('lidn2vew-' in f.name):
            lidn2v_avg_accuracy.append(acc)
            lidn2v_avg_precision.append(prec)
            lidn2v_avg_recall.append(rec)
            lidn2v_avg_f1.append(f1)
            lidn2v_avg_fpr.append(fpr)
        elif ('lidn2vewpq-' in f.name):
            lidn2v_ext_avg_accuracy.append(acc)
            lidn2v_ext_avg_precision.append(prec)
            lidn2v_ext_avg_recall.append(rec)
            lidn2v_ext_avg_f1.append(f1)
            lidn2v_ext_avg_fpr.append(fpr)


        print("\nRandom Forest Evaluation Report")
        print("=================================\n")
        print(f"Best Hyperparameters:\n{grid_search.best_params_}\n")
        print("Confusion Matrix:")
        print(cm)
        print()
        print(f"False Positive Rate (FPR): {fpr:.4f}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}\n")
        print("Classification Report:")
        print(report)
        folder_path = reports_path + graph + '/Random Forest/' + method
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        path = folder_path + '/' + f.name + 'forest_report.txt'
        make_forest_report(y_test, y_pred, path, f.name)
    print("LID-Aware N2V Metrics:")
    print(f"Average Accuracy: {np.mean(lidn2v_avg_accuracy):.2f}")
    print(f"Average Precision: {np.mean(lidn2v_avg_precision):.2f}")
    print(f"Average Recall: {np.mean(lidn2v_avg_recall):.2f}")
    print(f"Average F1-Score: {np.mean(lidn2v_avg_f1):.2f}")
    print(f"Average FPR: {np.mean(lidn2v_avg_fpr):.2f}")
    print()
    print("LID-Aware Extended N2V Metrics:")
    print(f"Average Accuracy: {np.mean(lidn2v_ext_avg_accuracy):.2f}")
    print(f"Average Precision: {np.mean(lidn2v_ext_avg_precision):.2f}")
    print(f"Average Recall: {np.mean(lidn2v_ext_avg_recall):.2f}")
    print(f"Average F1-Score: {np.mean(lidn2v_ext_avg_f1):.2f}")
    print(f"Average FPR: {np.mean(lidn2v_ext_avg_fpr):.2f}")
    print()
    print("N2V Metrics:")
    print(f"Average Accuracy: {np.mean(n2v_avg_accuracy):.2f}")
    print(f"Average Precision: {np.mean(n2v_avg_precision):.2f}")
    print(f"Average Recall: {np.mean(n2v_avg_recall):.2f}")
    print(f"Average F1-Score: {np.mean(n2v_avg_f1):.2f}")
    print(f"Average FPR: {np.mean(n2v_avg_fpr):.2f}")
    print()
    file_name = reports_path + graph + '/Random Forest/' + method + '_average_forest_report.txt'
    with open(file_name, "w") as f:
        f.write("LID-Aware N2V Metrics:\n")
        f.write(f"Average Accuracy: {np.mean(lidn2v_avg_accuracy):.2f}\n")
        f.write(f"Average Precision: {np.mean(lidn2v_avg_precision):.2f}\n")
        f.write(f"Average Recall: {np.mean(lidn2v_avg_recall):.2f}\n")
        f.write(f"Average F1-Score: {np.mean(lidn2v_avg_f1):.2f}\n")
        f.write(f"Average FPR: {np.mean(lidn2v_avg_fpr):.2f}\n")
        f.write('\n')
        f.write("LID-Aware Extended N2V Metrics:\n")
        f.write(f"Average Accuracy: {np.mean(lidn2v_ext_avg_accuracy):.2f}\n")
        f.write(f"Average Precision: {np.mean(lidn2v_ext_avg_precision):.2f}\n")
        f.write(f"Average Recall: {np.mean(lidn2v_ext_avg_recall):.2f}\n")
        f.write(f"Average F1-Score: {np.mean(lidn2v_ext_avg_f1):.2f}\n")
        f.write(f"Average FPR: {np.mean(lidn2v_ext_avg_fpr):.2f}\n")
        f.write('\n')
        f.write("N2V Metrics:\n")
        f.write(f"Average Accuracy: {np.mean(n2v_avg_accuracy):.2f}\n")
        f.write(f"Average Precision: {np.mean(n2v_avg_precision):.2f}\n")
        f.write(f"Average Recall: {np.mean(n2v_avg_recall):.2f}\n")
        f.write(f"Average F1-Score: {np.mean(n2v_avg_f1):.2f}\n")
        f.write(f"Average FPR: {np.mean(n2v_avg_fpr):.2f}\n")
        f.write('\n')

if __name__ == '__main__':
    run_random_forest("cora", 'diff', '/Users/vukdermanovic/grasp/graspe/data/datasets/1_to_n_sampled_datasets_n=4_p=1/link_pred_datasets', '/Users/vukdermanovic/Faks/MasterRad/LIDEmbedEval/reports/Link Prediction/Supervised Reports/1_to_n_sampled_datasets_n=4_p=1/')