import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.src.callbacks import ReduceLROnPlateau
from keras import layers, models
from tensorflow.python.keras.engine.base_layer import Layer

import src.utils.load_dataset as loads
from pathlib import Path

from src.utils.make_nn_report import make_nn_report
import os

def run_nn(graph, method):
    # graph = 'karate_club_graph'
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
        if method == 'diff':
            X, y = loads.load_from_file_diff(f)
        elif method == 'concat':
            X, y = loads.load_from_file(f)
        y = y.astype(int)


        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        all_preds = []
        all_true = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n Fold {fold}")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = models.Sequential([
                layers.Input(shape=(X.shape[1],)),
                layers.Dense(512, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])


            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

            # Train the model
            model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test),
                      callbacks=[lr_scheduler], verbose=0)

            # Predict
            y_pred_probs = model.predict(X_test).ravel()
            y_pred = (y_pred_probs >= 0.5).astype(int)

            all_preds.extend(y_pred)
            all_true.extend(y_test)


        cm = confusion_matrix(all_true, all_preds)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn)
        acc = accuracy_score(all_true, all_preds)
        prec = precision_score(all_true, all_preds)
        rec = recall_score(all_true, all_preds)
        f1 = f1_score(all_true, all_preds)
        report = classification_report(all_true, all_preds)

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

        # Print metrics
        print("\n🧠 Neural Network Evaluation Report")
        print("==================================\n")
        print("Confusion Matrix:")
        print(cm)
        print(f"\nFalse Positive Rate: {fpr:.4f}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}\n")
        print("Classification Report:")
        print(report)

        folder_path = '/Users/vukdermanovic/Faks/MasterRad/LIDEmbedEval/reports/Supervised Reports/' + graph + '/Neural Network/' + method
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        path = '/Users/vukdermanovic/Faks/MasterRad/LIDEmbedEval/reports/Supervised Reports/' + graph + '/Neural Network/' + method + '/' + f.name + 'nn_report.txt'
        make_nn_report(all_true, all_preds, path, f.name, model)

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
    if (len(hub_avg_fpr) > 0):
        print("HUBDISTRIBUTION Metrics:")
        print(f"Average Accuracy: {np.mean(hub_avg_accuracy):.2f}")
        print(f"Average Precision: {np.mean(hub_avg_precision):.2f}")
        print(f"Average Recall: {np.mean(hub_avg_recall):.2f}")
        print(f"Average F1-Score: {np.mean(hub_avg_f1):.2f}")
        print(f"Average FPR: {np.mean(hub_avg_fpr):.2f}")
        print()
    if (len(scwalk_avg_fpr) > 0):
        print("SCWALK Metrics:")
        print(f"Average Accuracy: {np.mean(scwalk_avg_accuracy):.2f}")
        print(f"Average Precision: {np.mean(scwalk_avg_precision):.2f}")
        print(f"Average Recall: {np.mean(scwalk_avg_recall):.2f}")
        print(f"Average F1-Score: {np.mean(scwalk_avg_f1):.2f}")
        print(f"Average FPR: {np.mean(scwalk_avg_fpr):.2f}")
        print()
    file_name = '/Users/vukdermanovic/Faks/MasterRad/LIDEmbedEval/reports/Supervised Reports/' + graph + '/Neural Network/' + method + '_average_nn_report.txt'
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
        if (len(hub_avg_fpr) > 0):
            f.write("HUBDISTRIBUTION Metrics:\n")
            f.write(f"Average Accuracy: {np.mean(hub_avg_accuracy):.2f}\n")
            f.write(f"Average Precision: {np.mean(hub_avg_precision):.2f}\n")
            f.write(f"Average Recall: {np.mean(hub_avg_recall):.2f}\n")
            f.write(f"Average F1-Score: {np.mean(hub_avg_f1):.2f}\n")
            f.write(f"Average FPR: {np.mean(hub_avg_fpr):.2f}\n")
            f.write('\n')
        if (len(scwalk_avg_fpr) > 0):
            f.write("SCWALK Metrics:\n")
            f.write(f"Average Accuracy: {np.mean(scwalk_avg_accuracy):.2f}\n")
            f.write(f"Average Precision: {np.mean(scwalk_avg_precision):.2f}\n")
            f.write(f"Average Recall: {np.mean(scwalk_avg_recall):.2f}\n")
            f.write(f"Average F1-Score: {np.mean(scwalk_avg_f1):.2f}\n")
            f.write(f"Average FPR: {np.mean(scwalk_avg_fpr):.2f}\n")