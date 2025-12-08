# Analysis Specification

This application now offers both **K-Means** and **Hierarchical (Ward) clustering** for Demand Space. The method can be toggled in the Segmentation Setting tab and applies to both segments-as-points and variables-as-points modes.

## Clustering Options
- **K-Means**: fast, centroid-based partitioning where `k` is specified in advance.
- **Hierarchical (Ward)**: builds an agglomerative dendrogram and cuts it at `k`, preserving the merge history when exploring cluster structure.

## Similarity / Profile Construction
- **Segments-as-points**: segment profiles can mix numeric factor scores/targets with categorical targets. Numeric features use means; categorical targets contribute normalized distribution columns (e.g., `target::level`). A R-style option builds a target-by-segment distribution pivot (columns sum to 1 per segment) and clusters using those proportions.
- **Variables-as-points**: still runs PCA or MDS on the numericized variable matrix but now supports as few as two selected variables.

## Practical Notes
- Manual cluster edits (drag/drop) override base clustering regardless of algorithm choice.
- Export uses effective cluster assignments and includes per-cluster sheets plus raw data with the cluster label.
- Non-numeric targets no longer block segmentation; distribution-based features are generated automatically when needed.
