from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from  pathlib import Path
from src.utils.make_svm_report import make_svm_report
import src.utils.load_dataset as loads
import numpy as np
import os

def run_svm(graph, method):
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
    hub_avg_accuracy = []
    hub_avg_precision = []
    hub_avg_recall = []
    hub_avg_f1 = []
    hub_avg_fpr = []
    scwalk_avg_accuracy = []
    scwalk_avg_precision = []
    scwalk_avg_recall = []
    scwalk_avg_f1 = []
    scwalk_avg_fpr = []
    path_name = '/Users/vukdermanovic/Faks/MasterRad/LIDEmbedEval/data/datasets/by_graph/' + graph
    path = Path(path_name)
    for f in path.iterdir():
        X,y = loads.load_from_file_diff(f)
        y = y.astype(int)
        print("Embedding file:" + str(f.name))

        pipeline = make_pipeline(StandardScaler(), SVC())


        param_grid = {
            'svc__kernel': ['linear', 'rbf'],
            'svc__C': [0.1, 1, 10],
            'svc__gamma': ['scale', 0.01, 0.1]  # only applies to RBF kernel
        }


        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', verbose=1)
        grid_search.fit(X, y)


        best_model = grid_search.best_estimator_


        y_pred = cross_val_predict(best_model, X, y, cv=cv)


        print("\nBest Parameters Found:", grid_search.best_params_)
        print("\nClassification Report (Cross-Validated):")
        print(classification_report(y, y_pred))


        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()



        fpr = fp / (fp + tn)
        if ('HUBDISTRIBUTION' in f.name):
            hub_avg_f1.append(f1)
            hub_avg_precision.append(prec)
            hub_avg_recall.append(rec)
            hub_avg_accuracy.append(acc)
            hub_avg_fpr.append(fpr)
        elif ('n2v-' in f.name):
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
        elif ('SCWALK' in f.name):
            scwalk_avg_accuracy.append(acc)
            scwalk_avg_precision.append(prec)
            scwalk_avg_recall.append(rec)
            scwalk_avg_f1.append(f1)
            scwalk_avg_fpr.append(fpr)

        print(f"Accuracy: {acc:.2f}")
        print(f"Precision: {prec:.2f}")
        print(f"Recall: {rec:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"False Positive Rate (FPR): {fpr*100:.2f}%")
        folder_path = '/Users/vukdermanovic/Faks/MasterRad/LIDEmbedEval/reports/Supervised Reports/'+ graph +'/SVM/'+ method
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        path = '/Users/vukdermanovic/Faks/MasterRad/LIDEmbedEval/reports/Supervised Reports/'+ graph +'/SVM/'+ method +"/" + f.name + '_svm_report.txt'
        make_svm_report(y, y_pred, path, f.name)


        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        # disp.plot()
        # plt.title("Confusion Matrix (Cross-Validated Predictions)")
        # plt.show()

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
    if(len(hub_avg_fpr) > 0):
        print("HUBDISTRIBUTION Metrics:")
        print(f"Average Accuracy: {np.mean(hub_avg_accuracy):.2f}")
        print(f"Average Precision: {np.mean(hub_avg_precision):.2f}")
        print(f"Average Recall: {np.mean(hub_avg_recall):.2f}")
        print(f"Average F1-Score: {np.mean(hub_avg_f1):.2f}")
        print(f"Average FPR: {np.mean(hub_avg_fpr):.2f}")
        print()
    if(len(scwalk_avg_fpr) > 0):
        print("SCWALK Metrics:")
        print(f"Average Accuracy: {np.mean(scwalk_avg_accuracy):.2f}")
        print(f"Average Precision: {np.mean(scwalk_avg_precision):.2f}")
        print(f"Average Recall: {np.mean(scwalk_avg_recall):.2f}")
        print(f"Average F1-Score: {np.mean(scwalk_avg_f1):.2f}")
        print(f"Average FPR: {np.mean(scwalk_avg_fpr):.2f}")
        print()
    file_name = '/Users/vukdermanovic/Faks/MasterRad/LIDEmbedEval/reports/Supervised Reports/'+ graph +'/SVM/'+ method +'_average_svm_report.txt'
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
        if(len(hub_avg_fpr) > 0):
            f.write("HUBDISTRIBUTION Metrics:\n")
            f.write(f"Average Accuracy: {np.mean(hub_avg_accuracy):.2f}\n")
            f.write(f"Average Precision: {np.mean(hub_avg_precision):.2f}\n")
            f.write(f"Average Recall: {np.mean(hub_avg_recall):.2f}\n")
            f.write(f"Average F1-Score: {np.mean(hub_avg_f1):.2f}\n")
            f.write(f"Average FPR: {np.mean(hub_avg_fpr):.2f}\n")
            f.write('\n')
        if(len(scwalk_avg_fpr) > 0):
            f.write("SCWALK Metrics:\n")
            f.write(f"Average Accuracy: {np.mean(scwalk_avg_accuracy):.2f}\n")
            f.write(f"Average Precision: {np.mean(scwalk_avg_precision):.2f}\n")
            f.write(f"Average Recall: {np.mean(scwalk_avg_recall):.2f}\n")
            f.write(f"Average F1-Score: {np.mean(scwalk_avg_f1):.2f}\n")
            f.write(f"Average FPR: {np.mean(scwalk_avg_fpr):.2f}\n")

