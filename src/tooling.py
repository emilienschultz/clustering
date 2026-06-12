"""Reporting helpers built on top of the clustering pipeline output.

`model_comparison_table` turns a `process_dataset(...)['all_models']` frame into
the Table 2 layout: per validity index, the best and second-best distance-based
configurations (k-means / AHC / HDBSCAN) plus the best LCA model, shown apart.
"""

import pandas as pd

# Internal model name -> display label
ALGO_NAMES = {
    "HDBSCAN": "HDBSCAN",
    "kmeans": "K-means",
    "AHC": "AHC",
    "latent": "LCA",
}

# (column, display label, ascending?) -- ascending=True means a lower score is better
INDEX_SPEC = [
    ("silhouette", "Silhouette (SL)", False),
    ("calinski_harabasz", "Calinski-Harab. (CH)", False),
    ("davies_bouldin", "Davies-Bouldin (DB)", True),
    ("dunn", "Generalized Dunn (GD43)", False),
]


def model_comparison_table(all_models, n_samples, unbalanced_ratio=0.9, decimal=","):
    """Build the Table 2 model-comparison frame from `result['all_models']`.

    For each validity index, returns the best and second-best distance-based
    configurations (k-means / AHC / HDBSCAN) and the best LCA model, shown
    separately. A "configuration" is one (algorithm, n_clust) pair; among
    duplicate parameter grids that yield the same (algorithm, n_clust) the
    best-scoring one is kept. A `*` flags unbalanced solutions where the
    largest cluster holds at least `unbalanced_ratio` of all individuals.

    Parameters
    ----------
    all_models : pd.DataFrame
        The `all_models` frame returned by `process_dataset`.
    n_samples : int
        Number of observations (used for the unbalanced-cluster flag).
    unbalanced_ratio : float
        Largest-cluster share above which a model is flagged with `*`.
    decimal : str
        Decimal separator for the formatted scores (',' for French style).

    Returns
    -------
    pd.DataFrame with columns Section / Clustering Validity Index / Algorithm /
    Clusters Nb / Score.
    """
    df = all_models.copy()
    df["unbalanced"] = df["max_clust_size"] >= unbalanced_ratio * n_samples

    def fmt_n(row):
        return f"{int(row['n_clust'])}{'*' if row['unbalanced'] else ''}"

    def fmt_score(v):
        return f"{v:.2f}".replace(".", decimal)

    def ranked(sub, col, asc):
        return (
            sub.dropna(subset=[col])
            .sort_values(col, ascending=asc)
            .drop_duplicates(subset=["model", "n_clust"])
        )

    dist = df[df["model"] != "latent"]
    lca = df[df["model"] == "latent"]

    rows = []
    sections = [
        ("Best configurations", dist, 0),
        ("Second-best configurations", dist, 1),
        ("Best LCA models", lca, 0),
    ]
    for section, source, rank in sections:
        for col, label, asc in INDEX_SPEC:
            top = ranked(source, col, asc)
            if len(top) <= rank:
                continue
            r = top.iloc[rank]
            rows.append(
                {
                    "Section": section,
                    "Clustering Validity Index": label,
                    "Algorithm": ALGO_NAMES[r["model"]],
                    "Clusters Nb": fmt_n(r),
                    "Score": fmt_score(r[col]),
                }
            )
    return pd.DataFrame(rows)
