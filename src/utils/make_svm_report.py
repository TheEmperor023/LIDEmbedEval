from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def make_svm_report(y, y_pred, file_name, embed_file):
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)


    with open(file_name, "w") as f:
        f.write("SVM Evaluation Report\n")
        f.write("=====================\n")
        f.write(f"Dataset file: {embed_file}\n\n")

        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\n")

        f.write(f"False Positive Rate (FPR): {fpr:.4f}\n")
        f.write(f"Accuracy:  {accuracy_score(y, y_pred):.4f}\n")
        f.write(f"Precision: {precision_score(y, y_pred):.4f}\n")
        f.write(f"Recall:    {recall_score(y, y_pred):.4f}\n")
        f.write(f"F1 Score:  {f1_score(y, y_pred):.4f}\n\n")

        f.write("Detailed Classification Report:\n")
        f.write(classification_report(y, y_pred))
