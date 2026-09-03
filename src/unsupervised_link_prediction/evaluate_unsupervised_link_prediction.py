import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_unsupervised_link_prediction(
    dataset_path: str, output_report_path: str = "evaluation_report.txt"
) -> dict:
    """Evaluates unsupervised link prediction performance from a colon-separated embeddings file,

    including Hits@Positives (Hits@K where K = total positive edges), and saves the report to a file.

    Parameters:
    -----------
    dataset_path : str
        Path to the colon-delimited CSV/TXT file (node1:node2:embed1:embed2:label).
    output_report_path : str
        Path where the output text report will be saved.

    Returns:
    --------
    dict
        A nested dictionary containing computed metrics for each similarity operator.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    # 1. Parse Colon-Delimited CSV
    print(f"Loading data from: {dataset_path}...")
    df = pd.read_csv(
        dataset_path,
        sep=":",
        header=None,
        names=["node1", "node2", "embed1", "embed2", "label"],
    )

    def parse_vector_series(series: pd.Series) -> torch.Tensor:
        parsed = series.apply(
            lambda x: [float(val) for val in str(x).split(",")]
        ).tolist()
        return torch.tensor(parsed, dtype=torch.float32)

    z_u = parse_vector_series(df["embed1"])
    z_v = parse_vector_series(df["embed2"])
    y_true = df["label"].values.astype(int)

    total_positives = int(np.sum(y_true))
    if total_positives == 0 or len(np.unique(y_true)) < 2:
        raise ValueError(
            "Evaluation requires both positive (1) and negative (0) labels in the dataset."
        )

    # 2. Compute Pairwise Similarity Scores
    similarity_operators = {
        "Dot Product": (z_u * z_v).sum(dim=-1).numpy(),
        "Cosine Similarity": (
            F.normalize(z_u, p=2, dim=-1) * F.normalize(z_v, p=2, dim=-1)
        )
        .sum(dim=-1)
        .numpy(),
        "Negated L2 Distance": -torch.norm(z_u - z_v, p=2, dim=-1).numpy(),
    }

    # 3. Hits@K Metric Calculation
    def hits_at_k(y_true_arr, y_scores_arr, k):
        actual_k = min(k, len(y_scores_arr))
        top_k_idx = np.argsort(y_scores_arr)[::-1][:actual_k]
        hits = np.sum(y_true_arr[top_k_idx])
        return hits / actual_k if actual_k > 0 else 0.0

    # 4. Calculate Metrics and Format Output
    results = {}
    report_lines = [
        "==================================================",
        " UNSUPERVISED LINK PREDICTION EVALUATION REPORT",
        "==================================================",
        f"Dataset: {dataset_path}",
        f"Total Samples Evaluated: {len(y_true)}",
        f"Positive Samples (K): {total_positives} | Negative Samples: {len(y_true) - total_positives}",
        "--------------------------------------------------\n",
    ]

    for operator_name, y_scores in similarity_operators.items():
        auc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        h10 = hits_at_k(y_true, y_scores, k=10)
        h50 = hits_at_k(y_true, y_scores, k=50)
        hits_at_pos = hits_at_k(y_true, y_scores, k=total_positives)

        results[operator_name] = {
            "ROC-AUC": auc,
            "PR-AUC": pr_auc,
            "Hits@10": h10,
            "Hits@50": h50,
            f"Hits@{total_positives} (Hits@Pos)": hits_at_pos,
        }

        report_lines.extend(
            [
                f"Operator: {operator_name}",
                f"  ROC-AUC                : {auc:.5f}",
                f"  PR-AUC                 : {pr_auc:.5f}",
                f"  Hits@10                : {h10:.5f}",
                f"  Hits@50                : {h50:.5f}",
                f"  Hits@{total_positives:<5} (Hits@Positives) : {hits_at_pos:.5f}",
                "--------------------------------------------------",
            ]
        )

    # 5. Save Report to File
    output_dir = os.path.dirname(output_report_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    report_text = "\n".join(report_lines)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nResults successfully saved to: {output_report_path}")

    return results


# Example Usage:
if __name__ == "__main__":
    evaluate_unsupervised_link_prediction(
        dataset_path='/Users/vukdermanovic/grasp/graspe/data/datasets/1_to_n_sampled_datasets_n=1_p=1/link_pred_datasets/cora/test/n2v-cora-10-4_0-0_25.testset',
        output_report_path="results/link_prediction_metrics.txt",
    )