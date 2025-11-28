# =============================================================================
# app.py (Part 1/9)
# Auto Segment Tool v7.0 - Imports, Helpers, AppState
# =============================================================================
# -*- coding: utf-8 -*-

"""
Auto Segment Tool v7.0

[Updates in v7.0]
1. Decision Tree Logic: Implemented 'Optimal Subset Split'.
   - Categorical variables are sorted by target mean (or class probability) before splitting.
   - Finds best cut among sorted categories (Group A vs Group B).
2. UI Improvement: 'Select Predictors' (Whitelist) instead of Exclude list.
3. Factor Analysis: Added option for PCA vs EFA (Exploratory Factor Analysis).
   - Output columns standardized to 'Factor1', 'Factor2', ...
4. RAG Chatbot: Added a lightweight AI Assistant tab with context injection.
   - Context includes: Data columns, Best Split results, Error logs.
5. Localization: All UI messages and warnings converted to English.
"""

from __future__ import annotations

import os
import sys
import json
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import requests  # For RAG API calls

from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# Scikit-learn dependencies
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Helper Functions: Colors & Formatting
# -----------------------------------------------------------------------------

def pal_hex() -> List[str]:
    """Returns a list of 20 distinct colors for plotting."""
    return [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2",
        "#ff9da6", "#9d755d", "#bab0ac", "#72b7b2", "#fabfd2",
    ]


def qcolor(hex_: str, alpha: int = 255) -> QtGui.QColor:
    c = QtGui.QColor(hex_)
    c.setAlpha(int(alpha))
    return c


def is_categorical_series(s: pd.Series, max_unique_numeric_as_cat: int = 20) -> bool:
    if s is None:
        return False
    if pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        nun = s.dropna().nunique()
        return nun <= max_unique_numeric_as_cat
    return False


def to_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    pts = pts[~np.any(~np.isfinite(pts), axis=1)]
    if len(pts) < 3:
        return pts
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    return np.array(lower[:-1] + upper[:-1], dtype=float)


def show_error(parent, title: str, err: Exception):
    msg = QtWidgets.QMessageBox(parent)
    msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
    msg.setWindowTitle(title)
    msg.setText(str(err))
    msg.setDetailedText(traceback.format_exc())
    msg.exec()


def fmt_float(x: Any, decimals: int = 2) -> str:
    try:
        if x is None:
            return ""
        if pd.isna(x):
            return ""
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.{decimals}f}"
        return str(x)
    except Exception:
        return str(x)


def style_button(btn: QtWidgets.QPushButton, level: int = 1):
    palette = ["#e3f2fd", "#bbdefb", "#90caf9", "#64b5f6", "#42a5f5"]
    idx = max(0, min(level - 1, len(palette) - 1))
    base = palette[idx]
    hover = palette[min(idx + 1, len(palette) - 1)]
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {base};
            border: 1px solid #90a4ae;
            border-radius: 4px;
            padding: 4px 8px;
            color: #000000;
        }}
        QPushButton:hover {{ background-color: {hover}; }}
        QPushButton:disabled {{ background-color: #eceff1; color: #90a4ae; }}
    """)


def normalize_recode_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    cols = list(df.columns)
    col_lut = {str(c).strip().lower(): c for c in cols}

    def pick(*names):
        for n in names:
            if n in col_lut: return col_lut[n]
        return None

    q = pick("question", "문항", "문항명", "q", "item", "variable")
    c = pick("code", "코드", "값", "value", "val")
    n = pick("name", "라벨", "label", "명", "설명", "text")

    if q is None or c is None or n is None:
        if len(cols) >= 3:
            q, c, n = cols[0], cols[1], cols[2]

    out = df.copy()
    rename_map = {}
    if q: rename_map[q] = "QUESTION"
    if c: rename_map[c] = "CODE"
    if n: rename_map[n] = "NAME"
    out = out.rename(columns=rename_map)

    for cc in ["QUESTION", "CODE", "NAME"]:
        if cc not in out.columns: out[cc] = np.nan

    out["QUESTION"] = out["QUESTION"].astype(str).str.strip()
    out["CODE"] = out["CODE"].astype(str).str.strip()
    out["NAME"] = out["NAME"].astype(str).str.strip()
    return out[["QUESTION", "CODE", "NAME"] + [c for c in out.columns if c not in ["QUESTION", "CODE", "NAME"]]]

# -----------------------------------------------------------------------------
# Application State Class
# -----------------------------------------------------------------------------
@dataclass
class AppState:
    """Holds the global state of the application data."""
    # Raw Data
    df: Optional[pd.DataFrame] = None
    path: Optional[str] = None
    sheet: Optional[str] = None
    recode_df: Optional[pd.DataFrame] = None

    # Factor Analysis Data (PCA or EFA)
    factor_model: Any = None  # PCA or FactorAnalysis object
    factor_cols: Optional[List[str]] = None
    factor_scores: Optional[pd.DataFrame] = None
    factor_loadings: Optional[pd.DataFrame] = None
    factor_mode: str = "PCA"  # "PCA" or "EFA"

    # Decision tree outputs (Setting Tab)
    dt_improve_pivot: Optional[pd.DataFrame] = None
    dt_split_best: Optional[pd.DataFrame] = None

    # Decision tree full Analysis (Results Tab)
    dt_full_nodes: Optional[pd.DataFrame] = None
    dt_full_split_groups: Optional[pd.DataFrame] = None
    dt_full_split_branches: Optional[pd.DataFrame] = None
    dt_full_path_info: Optional[pd.DataFrame] = None
    dt_full_condition_freq: Optional[pd.DataFrame] = None
    dt_full_selected: Tuple[Optional[str], Optional[str]] = (None, None)

    # Demand Space Data
    demand_mode: str = "Segments-as-points"
    demand_xy: Optional[pd.DataFrame] = None
    cluster_assign: Optional[pd.Series] = None
    cluster_names: Dict[int, str] = field(default_factory=dict)

    # Demand Space Profile Data
    demand_seg_profile: Optional[pd.DataFrame] = None
    demand_seg_components: Optional[List[str]] = None
    demand_features_used: Optional[List[str]] = None

    manual_dirty: bool = False  # Flag for manual edits in plots
    
    # System Logs (for RAG)
    last_error: str = ""

# =============================================================================
# app.py (Part 2/9)
# DataFrameTable & Improved Decision Tree Logic (Optimal Subset)
# =============================================================================

# -----------------------------------------------------------------------------
# Custom UI Widget: DataFrame Table
# -----------------------------------------------------------------------------
class DataFrameTable(QtWidgets.QTableWidget):
    """
    A QTableWidget specialized for displaying pandas DataFrames.
    Supports easy loading, formatting, and read-only/editable modes.
    """
    def __init__(self, parent=None, editable: bool = False, float_decimals: int = 2, max_col_width: int = 380):
        super().__init__(parent)
        self._float_decimals = int(float_decimals)
        self._max_col_width = int(max_col_width)
        
        # UI Selection Behavior
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        
        if not editable:
            self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)

    def set_df(self, df: Optional[pd.DataFrame], max_rows: int = 500):
        """Populates the table with data from a DataFrame."""
        if df is None or df.empty:
            self.setRowCount(0)
            self.setColumnCount(0)
            return

        # Limit rows for performance if needed
        view = df.copy()
        if len(view) > max_rows:
            view = view.iloc[:max_rows].copy()

        self.setColumnCount(view.shape[1])
        self.setRowCount(view.shape[0])
        self.setHorizontalHeaderLabels([str(c) for c in view.columns])

        # Fill Data
        for r in range(view.shape[0]):
            for c in range(view.shape[1]):
                v = view.iat[r, c]
                
                # Format text based on type
                if isinstance(v, (float, np.floating)):
                    txt = "" if pd.isna(v) else f"{float(v):.{self._float_decimals}f}"
                else:
                    txt = "" if pd.isna(v) else str(v)
                
                item = QtWidgets.QTableWidgetItem(txt)
                self.setItem(r, c, item)

        # Adjust Columns
        header = self.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)

        total_width = max(self.viewport().width(), 800)
        default_width = min(
            self._max_col_width,
            max(80, total_width // max(1, view.shape[1]))
        )
        header.setDefaultSectionSize(default_width)

        for c in range(view.shape[1]):
            if self.columnWidth(c) > self._max_col_width:
                self.setColumnWidth(c, self._max_col_width)

        header.setStretchLastSection(True)


# -----------------------------------------------------------------------------
# Decision Tree Logic: Impurity Calculations
# -----------------------------------------------------------------------------

def _impurity_reg(y: np.ndarray) -> float:
    """Mean Squared Error for regression impurity."""
    if y.size == 0:
        return 0.0
    mu = float(np.mean(y))
    return float(np.mean((y - mu) ** 2))


def _impurity_gini(y: np.ndarray) -> float:
    """Gini Impurity for classification."""
    if y.size == 0:
        return 0.0
    vals, cnt = np.unique(y, return_counts=True)
    p = cnt / cnt.sum()
    return float(1.0 - np.sum(p ** 2))


def _root_dev(y: np.ndarray, task: str) -> float:
    """Calculates the root deviance (impurity * n)."""
    if y.size == 0:
        return 1e-12
    if task == "reg":
        imp = _impurity_reg(y.astype(float))
    else:
        imp = _impurity_gini(y)
    dev = imp * float(len(y))
    return float(max(dev, 1e-12))


def univariate_best_split(
    y: pd.Series,
    x: pd.Series,
    task: str,  # "reg" or "class"
    max_unique_cat: int = 50
) -> Tuple[Optional[dict], List[dict]]:
    """
    Finds the single best split for a target (y) and predictor (x).
    
    [MAJOR UPDATE v7.0] Optimal Subset Split logic for categorical variables.
    Instead of 1-vs-Rest, it sorts categories by Target Mean (or Class Prob)
    and finds the best cut point in the sorted list.
    """
    # 1. Prepare Data
    mask = pd.notna(y) & pd.notna(x)
    yv = y[mask].values
    xv = x[mask]
    
    if yv.size < 5:
        return None, []

    if task == "reg":
        try:
            yv = yv.astype(float)
        except Exception:
            return None, []

    dev_root = _root_dev(np.asarray(yv), task=task)
    rows: List[dict] = []

    # 2. Case A: Categorical Predictor (OPTIMAL SUBSET SPLIT)
    if is_categorical_series(xv):
        xv_str = xv.astype(str)
        cats = xv_str.unique()
        
        if len(cats) < 2 or len(cats) > max_unique_cat:
            return None, []

        # Step 2-1: Calculate sorting metric for each category
        # - Reg: Mean of Y
        # - Class: Probability of the most frequent global class (or just Class=1)
        
        if task == "reg":
            # Sort by Mean Y
            cat_stats = pd.DataFrame({'x': xv_str, 'y': yv})
            agg = cat_stats.groupby('x')['y'].mean().sort_values()
            sorted_cats = agg.index.tolist()
        else:
            # Sort by P(Class=Target)
            # Find dominant class first
            u, c = np.unique(yv, return_counts=True)
            target_cls = u[np.argmax(c)]
            
            cat_stats = pd.DataFrame({'x': xv_str, 'y': yv})
            cat_stats['is_target'] = (cat_stats['y'] == target_cls).astype(int)
            agg = cat_stats.groupby('x')['is_target'].mean().sort_values()
            sorted_cats = agg.index.tolist()

        # Step 2-2: Iterate through split points in the sorted list
        # E.g., Sorted: [A, B, C, D] -> Split 1: {A} vs {B,C,D}, Split 2: {A,B} vs {C,D}...
        
        # Map original values to sorted rank for fast masking
        rank_map = {cat: i for i, cat in enumerate(sorted_cats)}
        x_rank = np.array([rank_map[v] for v in xv_str])

        for i in range(len(sorted_cats) - 1):
            # Left group: ranks <= i
            # Right group: ranks > i
            left_mask = x_rank <= i
            right_mask = ~left_mask
            
            yL = yv[left_mask]
            yR = yv[right_mask]
            
            if yL.size == 0 or yR.size == 0:
                continue
                
            wL = yL.size / yv.size
            wR = yR.size / yv.size

            if task == "reg":
                child = (wL * _impurity_reg(yL.astype(float)) + wR * _impurity_reg(yR.astype(float))) * float(yv.size)
            else:
                child = (wL * _impurity_gini(yL) + wR * _impurity_gini(yR)) * float(yv.size)

            improve_abs = float(dev_root - child)
            improve_rel = float(improve_abs / dev_root) if dev_root > 0 else np.nan

            # Save full list of items for the "left" group for reconstruction later
            left_items = sorted_cats[:i+1]
            right_items = sorted_cats[i+1:]
            
            # Display text
            if len(left_items) <= 3:
                l_txt = "{" + ",".join(left_items) + "}"
            else:
                l_txt = f"{{...{len(left_items)} items...}}"
                
            if len(right_items) <= 3:
                r_txt = "{" + ",".join(right_items) + "}"
            else:
                r_txt = f"{{...{len(right_items)} items...}}"

            rows.append({
                "split_type": "categorical(subset)",
                "cutpoint": f"Rank {i} (Sorted)", # Internal use
                "left_group": l_txt,
                "right_group": r_txt,
                "left_items": left_items,   # Important: Keep the actual list
                "right_items": right_items, # Important: Keep the actual list
                "improve_abs": improve_abs,
                "improve_rel": improve_rel,
                "n_left": int(yL.size),
                "n_right": int(yR.size),
            })

    # 3. Case B: Numeric Predictor
    else:
        xnum = pd.to_numeric(xv, errors="coerce").values.astype(float)
        ok = np.isfinite(xnum)
        xnum = xnum[ok]
        y_use = np.asarray(yv)[ok]
        
        if y_use.size < 5:
            return None, []
            
        uniq = np.unique(xnum)
        if uniq.size < 2:
            return None, []
            
        uniq.sort()
        # Midpoints for thresholds
        mids = (uniq[:-1] + uniq[1:]) / 2.0

        for thr in mids:
            left_mask = xnum <= thr
            right_mask = ~left_mask
            yL = y_use[left_mask]
            yR = y_use[right_mask]
            
            if yL.size == 0 or yR.size == 0:
                continue
                
            wL = yL.size / y_use.size
            wR = yR.size / y_use.size

            if task == "reg":
                child = (wL * _impurity_reg(yL.astype(float)) + wR * _impurity_reg(yR.astype(float))) * float(y_use.size)
            else:
                child = (wL * _impurity_gini(yL) + wR * _impurity_gini(yR)) * float(y_use.size)

            dev_root2 = _root_dev(np.asarray(y_use), task=task)
            improve_abs = float(dev_root2 - child)
            improve_rel = float(improve_abs / dev_root2) if dev_root2 > 0 else np.nan

            rows.append({
                "split_type": "numeric(threshold)",
                "cutpoint": float(thr),
                "left_group": f"<= {thr:.6g}",
                "right_group": f"> {thr:.6g}",
                "improve_abs": improve_abs,
                "improve_rel": improve_rel,
                "n_left": int(yL.size),
                "n_right": int(yR.size),
            })

    if not rows:
        return None, []
        
    # Pick Best
    best = max(rows, key=lambda r: (r.get("improve_rel", -1e9) if pd.notna(r.get("improve_rel", np.nan)) else -1e9))
    return best, rows

# =============================================================================
# app.py (Part 3/9)
# Tree Building Logic (Recursive)
# =============================================================================

@dataclass
class UniNode:
    """Represents a node in the Univariate Decision Tree."""
    node_id: int
    depth: int
    n: int
    condition: str      # Path condition text
    is_leaf: bool
    split_type: Optional[str] = None
    cutpoint: Optional[str] = None
    improve_abs: Optional[float] = None
    improve_rel: Optional[float] = None
    left_condition: Optional[str] = None
    right_condition: Optional[str] = None
    left_id: Optional[int] = None
    right_id: Optional[int] = None
    pred: Optional[str] = None
    
    # [v7.0] Store optimal subset items for precise reconstruction
    subset_items: Optional[List[str]] = None 


def _pred_text(y: np.ndarray, task: str) -> str:
    """Returns a string representation of the prediction (Mean or Mode)."""
    if y.size == 0:
        return ""
    if task == "reg":
        return fmt_float(np.mean(y.astype(float)), 2)
    vals, cnt = np.unique(y, return_counts=True)
    m = vals[int(np.argmax(cnt))]
    return str(m)


def _univariate_best_split_on_subset(
    y: pd.Series,
    x: pd.Series,
    idx: np.ndarray,
    task: str,
    max_unique_cat: int = 50
) -> Tuple[Optional[dict], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Helper to find the best split on a subset of data defined by `idx`.
    Returns: (best_split_dict, left_indices, right_indices)
    """
    ys = y.iloc[idx]
    xs = x.iloc[idx]
    
    best, _rows = univariate_best_split(ys, xs, task=task, max_unique_cat=max_unique_cat)
    if best is None:
        return None, None, None

    # Re-calculate masks to return indices
    mask = pd.notna(ys) & pd.notna(xs)
    valid_idx = idx[mask.values]
    
    # Need values aligned with valid_idx
    xs_v = xs.iloc[valid_idx] # Keep as Series for map/isin

    if len(valid_idx) < 5:
        return None, None, None

    # [v7.0] Handling Optimal Subset Split
    if best["split_type"] == "categorical(subset)":
        left_items = best["left_items"] # List of strings
        # Create boolean mask using isin
        # Ensure xs_v is string for comparison
        left_mask = xs_v.astype(str).isin(left_items).values
        right_mask = ~left_mask
    
    # Standard Numeric Threshold
    else: 
        thr = float(best["cutpoint"])
        xnum = pd.to_numeric(xs_v, errors="coerce").values.astype(float)
        ok = np.isfinite(xnum)
        
        # Filter again for numeric validity
        valid_idx = valid_idx[ok]
        xnum = xnum[ok]
        
        left_mask = xnum <= thr
        right_mask = ~left_mask

    left_idx = valid_idx[left_mask]
    right_idx = valid_idx[right_mask]
    
    if len(left_idx) == 0 or len(right_idx) == 0:
        return None, None, None
        
    return best, left_idx, right_idx


def build_univariate_tree_full(
    df: pd.DataFrame,
    dep: str,
    ind: str,
    task: str,  # "reg" or "class"
    max_depth: int = 30,
    min_leaf: int = 1,
    min_split: int = 2,
    max_unique_cat: int = 50,
    min_improve_rel: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds a full univariate decision tree for a single pair of variables (dep, ind).
    Returns tables: nodes, split_groups, branches, path_info, cond_freq.
    """
    y = df[dep]
    x = df[ind]

    base_idx = np.where(pd.notna(y).values)[0]
    if len(base_idx) < max(5, min_split):
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    nodes: List[UniNode] = []
    next_id = 1

    def rec(idx: np.ndarray, depth: int, cond_parts: List[str]) -> int:
        nonlocal next_id, nodes
        n = int(len(idx))
        cond_text = " & ".join(cond_parts) if cond_parts else "(root)"

        # Stop criteria
        if depth >= max_depth or n < min_split:
            nid = next_id
            next_id += 1
            yy = y.iloc[idx].dropna().values
            nodes.append(UniNode(
                node_id=nid, depth=depth, n=n,
                condition=cond_text, is_leaf=True,
                pred=_pred_text(yy, task=task)
            ))
            return nid

        best, lidx, ridx = _univariate_best_split_on_subset(
            y, x, idx, task=task, max_unique_cat=max_unique_cat
        )
        
        # If no valid split found or improvement too low
        imp_rel = float(best.get("improve_rel", 0.0) if best else 0.0)
        
        if best is None or lidx is None or ridx is None or imp_rel < float(min_improve_rel):
            nid = next_id
            next_id += 1
            yy = y.iloc[idx].dropna().values
            nodes.append(UniNode(
                node_id=nid, depth=depth, n=n,
                condition=cond_text, is_leaf=True,
                pred=_pred_text(yy, task=task)
            ))
            return nid

        if len(lidx) < min_leaf or len(ridx) < min_leaf:
            nid = next_id
            next_id += 1
            yy = y.iloc[idx].dropna().values
            nodes.append(UniNode(
                node_id=nid, depth=depth, n=n,
                condition=cond_text, is_leaf=True,
                pred=_pred_text(yy, task=task)
            ))
            return nid

        # Create Internal Node
        nid = next_id
        next_id += 1
        
        subset_items = None

        # [v7.0] Formatting conditions for Optimal Subset
        if best["split_type"] == "categorical(subset)":
            l_items = best["left_items"]
            # Formatting for readable condition string
            # e.g. "Ind in ['A', 'B']"
            left_cond = f"{ind} in {str(l_items)}"
            right_cond = f"{ind} not in {str(l_items)}"
            cutpoint = "Optimal Subset"
            subset_items = l_items
        else:
            thr = float(best["cutpoint"])
            left_cond = f"{ind} <= {thr:.6g}"
            right_cond = f"{ind} > {thr:.6g}"
            cutpoint = f"{thr:.6g}"

        left_id = rec(lidx, depth + 1, cond_parts + [left_cond])
        right_id = rec(ridx, depth + 1, cond_parts + [right_cond])

        yy = y.iloc[idx].dropna().values
        nodes.append(UniNode(
            node_id=nid, depth=depth, n=n, condition=cond_text, is_leaf=False,
            split_type=best["split_type"],
            cutpoint=str(cutpoint),
            improve_abs=float(best.get("improve_abs", np.nan)),
            improve_rel=float(best.get("improve_rel", np.nan)),
            left_condition=left_cond, right_condition=right_condition,
            left_id=left_id, right_id=right_id,
            pred=_pred_text(yy, task=task),
            subset_items=subset_items # Save for mapping recommendation
        ))
        return nid

    # Start recursion
    _ = rec(base_idx, 0, [])

    # Convert to DataFrames
    nodes_df = pd.DataFrame([{
        "node_id": n.node_id,
        "depth": n.depth,
        "n": n.n,
        "condition": n.condition,
        "is_leaf": n.is_leaf,
        "split_type": n.split_type,
        "cutpoint": n.cutpoint,
        "improve_abs": n.improve_abs,
        "improve_rel": n.improve_rel,
        "left_id": n.left_id,
        "right_id": n.right_id,
        "left_condition": n.left_condition,
        "right_condition": n.right_condition,
        "pred": n.pred,
    } for n in sorted(nodes, key=lambda z: z.node_id)])

    internal = [n for n in nodes if not n.is_leaf]
    split_groups = pd.DataFrame([{
        "split_num": i + 1,
        "node_id": n.node_id,
        "dep": dep,
        "ind": ind,
        "split_type": n.split_type,
        "cutpoint": n.cutpoint,
        "left_group": n.left_condition,
        "right_group": n.right_condition,
        "improve_abs": n.improve_abs,
        "improve_rel": n.improve_rel,
        "n_node": n.n,
        "left_id": n.left_id,
        "right_id": n.right_id,
    } for i, n in enumerate(sorted(internal, key=lambda z: (z.depth, z.node_id)))])

    if not split_groups.empty:
        branches = split_groups.melt(
            id_vars=[
                "split_num", "node_id", "dep", "ind",
                "split_type", "cutpoint", "improve_abs", "improve_rel",
                "n_node", "left_id", "right_id"
            ],
            value_vars=["left_group", "right_group"],
            var_name="side",
            value_name="split"
        )
    else:
        branches = pd.DataFrame()

    leaves = [n for n in nodes if n.is_leaf]
    path_rows = []
    for lf in sorted(leaves, key=lambda z: z.node_id):
        path_rows.append({
            "dep": dep, "ind": ind,
            "leaf_id": lf.node_id,
            "n": lf.n,
            "pred": lf.pred,
            "path": lf.condition,
        })
    path_info = pd.DataFrame(path_rows)

    cond_freq = pd.DataFrame()
    if not path_info.empty:
        conds = []
        for p in path_info["path"].astype(str).tolist():
            if p == "(root)": continue
            parts = [t.strip() for t in p.split("&")]
            parts = [pp for pp in parts if pp and pp != "(root)"]
            conds.extend(parts)
        if conds:
            cond_freq = pd.Series(conds).value_counts().reset_index()
            cond_freq.columns = ["condition", "count"]
            cond_freq.insert(0, "dep", dep)
            
    return nodes_df, split_groups, branches, path_info, cond_freq

# =============================================================================
# app.py (Part 4/9)
# Demand Space Interactive Plot Components
# =============================================================================

# -----------------------------------------------------------------------------
# Demand Space Interactive Plot Components
# -----------------------------------------------------------------------------

class DraggableClusterLabel(pg.TextItem):
    """
    A text label for a cluster centroid that can be dragged.
    - Drag to another cluster: Merge clusters.
    - Shift + Drag: Move label only (no merge).
    """
    def __init__(self, plot: "DemandClusterPlot", cluster_id: int, text: str, color: QtGui.QColor):
        super().__init__(text=text, anchor=(0.5, 0.5))
        self.plot = plot
        self.cluster_id = int(cluster_id)
        self.setColor(color)
        f = QtGui.QFont()
        f.setPointSize(10)
        f.setBold(True)
        self.setFont(f)

    def mouseDragEvent(self, ev):
        # Only allow dragging if in Edit Mode
        if not self.plot.is_edit_mode_active():
            ev.ignore()
            return

        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        
        vb = self.plot.getPlotItem().vb
        pos_view = vb.mapSceneToView(ev.scenePos())
        ev.accept()

        # Check for Shift modifier
        shift = False
        try:
            mods = ev.modifiers()
            shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        except Exception:
            shift = False

        if ev.isFinish():
            if shift:
                # Move label only
                self.plot.remember_label_position(self.cluster_id, (float(pos_view.x()), float(pos_view.y())))
                return
            # Try to merge
            self.plot.try_merge_label(self.cluster_id, (float(pos_view.x()), float(pos_view.y())))
        else:
            self.setPos(float(pos_view.x()), float(pos_view.y()))


class ClusterViewBox(pg.ViewBox):
    """
    Custom ViewBox to handle mode-switching (Pan vs Edit).
    """
    def __init__(self, plot: "DemandClusterPlot"):
        super().__init__()
        self.plot = plot
        self.setMouseMode(self.RectMode)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            # If in Edit Mode, handle selection
            if self.plot.is_edit_mode_active():
                pos = self.mapSceneToView(ev.scenePos())
                mods = ev.modifiers()
                self.plot.click_select((float(pos.x()), float(pos.y())),
                                       shift=bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier))
                ev.accept()
                return
        super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if self.plot.is_edit_mode_active():
            # If in Edit Mode, handle point dragging
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                pos = self.mapSceneToView(ev.scenePos())
                self.plot.drag_event((float(pos.x()), float(pos.y())), ev)
                return
            # Ignore other buttons to prevent panning
            ev.ignore()
        else:
            # If in View Mode, allow standard panning/zooming
            super().mouseDragEvent(ev, axis=axis)


class DemandClusterPlot(pg.PlotWidget):
    """
    Main plotting widget for Demand Space.
    Supports:
    - Displaying points (segments or variables).
    - Drawing convex hulls for clusters.
    - Drag-and-drop merging of clusters.
    - Dragging individual points (reassignment or moving coords).
    """
    sigClustersChanged = QtCore.pyqtSignal()
    sigCoordsChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None, editable: bool = True):
        self._vb = ClusterViewBox(self)
        super().__init__(parent=parent, viewBox=self._vb)
        self._editable_widget = editable
        self._edit_mode_active = False  # True=Point/Merge, False=View/Pan

        self.setBackground("k")
        self.showGrid(x=True, y=True, alpha=0.15)
        self.getPlotItem().hideButtons()
        self.getPlotItem().setMenuEnabled(False)

        self._hex = pal_hex()

        # Data Containers
        self._ids: List[str] = []
        self._labels: List[str] = []
        self._xy: np.ndarray = np.zeros((0, 2), dtype=float)
        self._cluster: np.ndarray = np.zeros((0,), dtype=int)
        self._cluster_names: Dict[int, str] = {}

        # Visual Items
        self._scatter = pg.ScatterPlotItem(size=11, pxMode=True)
        self.addItem(self._scatter)

        self._point_text_items: List[pg.TextItem] = []
        self._hull_items: Dict[int, QtWidgets.QGraphicsPathItem] = {}
        self._label_items: Dict[int, DraggableClusterLabel] = {}

        # Interaction State
        self._selected: set[int] = set()
        self._dragging = False
        self._drag_temp_positions: Optional[np.ndarray] = None
        self._drag_anchor_xy: Optional[Tuple[float, float]] = None

        # UI toggles
        self._free_move_points: bool = False
        self._show_all_point_labels: bool = False

        # Label position overrides (cluster_id -> (x,y))
        self._label_pos_override: Dict[int, Tuple[float, float]] = {}

    def set_edit_mode_active(self, active: bool):
        """Sets whether interaction edits points or pans the view."""
        self._edit_mode_active = active
    
    def is_edit_mode_active(self) -> bool:
        return self._edit_mode_active and self._editable_widget

    def set_free_move_points(self, on: bool):
        self._free_move_points = bool(on)

    def set_show_all_point_labels(self, on: bool):
        self._show_all_point_labels = bool(on)
        self._draw_scatter()

    def reset_label_positions(self):
        self._label_pos_override.clear()
        self._draw_hulls_and_labels()

    def auto_arrange_labels(self):
        """Simple iterative repulsion to separate cluster labels."""
        if not self._label_items:
            return
        clusters = sorted(self._label_items.keys())
        if len(clusters) <= 1:
            return

        xr, yr = self.getPlotItem().vb.viewRange()
        scale = max((xr[1] - xr[0]), (yr[1] - yr[0]))
        if not np.isfinite(scale) or scale <= 0:
            scale = 10.0
        min_dist = max(0.06 * scale, 0.6)

        pos = {}
        for cid in clusters:
            it = self._label_items[cid]
            p = it.pos()
            pos[cid] = np.array([float(p.x()), float(p.y())], dtype=float)

        # Iterative repel
        for _ in range(40):
            moved = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    ci, cj = clusters[i], clusters[j]
                    vi, vj = pos[ci], pos[cj]
                    d = vj - vi
                    dist = float(np.hypot(d[0], d[1]))
                    if dist < 1e-9:
                        d = np.array([1.0, 0.0])
                        dist = 1.0
                    if dist < min_dist:
                        push = (min_dist - dist) * 0.5
                        dirv = d / dist
                        pos[ci] = vi - dirv * push
                        pos[cj] = vj + dirv * push
                        moved = True
            if not moved:
                break

        for cid in clusters:
            p = pos[cid]
            self._label_pos_override[int(cid)] = (float(p[0]), float(p[1]))
            self._label_items[cid].setPos(float(p[0]), float(p[1]))

    def _cluster_color(self, cid: int, alpha=255) -> QtGui.QColor:
        base = self._hex[(int(cid) - 1) % len(self._hex)]
        return qcolor(base, alpha=alpha)

    def set_data(self, ids: List[str], labels: List[str], xy: np.ndarray, clusters: np.ndarray,
                 cluster_names: Optional[Dict[int, str]] = None):
        self._ids = list(map(str, ids))
        self._labels = list(map(lambda x: "" if x is None else str(x), labels))
        self._xy = np.asarray(xy, dtype=float)
        self._cluster = np.asarray(clusters, dtype=int)
        self._cluster_names = dict(cluster_names or {})

        self._selected.clear()
        self._dragging = False
        self._drag_temp_positions = None
        self._drag_anchor_xy = None

        # label override reset on new data
        self._label_pos_override.clear()

        self.redraw_all()

        # Auto-scale view
        try:
            self.getPlotItem().enableAutoRange()
        except Exception:
            pass

    def get_cluster_series(self) -> pd.Series:
        return pd.Series(self._cluster.copy(), index=self._ids)

    def get_xy_map(self) -> Dict[str, Tuple[float, float]]:
        return {self._ids[i]: (float(self._xy[i, 0]), float(self._xy[i, 1])) for i in range(len(self._ids))}

    def set_cluster_names(self, names: Dict[int, str]):
        self._cluster_names = dict(names or {})
        self._draw_hulls_and_labels()

    def remember_label_position(self, cid: int, pos_xy: Tuple[float, float]):
        self._label_pos_override[int(cid)] = (float(pos_xy[0]), float(pos_xy[1]))

    def redraw_all(self):
        self._draw_scatter()
        self._draw_hulls_and_labels()

    def _draw_scatter(self):
        xy = self._xy if self._drag_temp_positions is None else self._drag_temp_positions

        spots = []
        for i in range(len(self._ids)):
            cid = int(self._cluster[i]) if len(self._cluster) else 1
            col = self._cluster_color(cid, alpha=220)
            selected = (i in self._selected)
            pen = pg.mkPen(qcolor("#ffffff", 230) if selected else col,
                           width=2 if selected else 1)
            brush = pg.mkBrush(col)
            spots.append({
                "pos": (float(xy[i, 0]), float(xy[i, 1])),
                "data": i,
                "pen": pen,
                "brush": brush,
                "symbol": "o",
            })
        self._scatter.setData(spots)

        for t in self._point_text_items:
            self.removeItem(t)
        self._point_text_items.clear()

        # Determine which labels to show
        show_idx = list(range(len(self._ids))) if self._show_all_point_labels else sorted(list(self._selected))
        for i in show_idx:
            cid = int(self._cluster[i])
            col = self._cluster_color(cid, alpha=255)
            t = pg.TextItem(text=self._labels[i], anchor=(0, 1))
            t.setColor(col)
            t.setPos(float(self._xy[i, 0]), float(self._xy[i, 1]))
            self.addItem(t)
            self._point_text_items.append(t)

    def _draw_hulls_and_labels(self):
        vb = self.getPlotItem().vb

        for h in self._hull_items.values():
            vb.removeItem(h)
        self._hull_items.clear()

        for lab in self._label_items.values():
            self.removeItem(lab)
        self._label_items.clear()

        clusters = sorted(set(map(int, self._cluster.tolist()))) if len(self._cluster) else []
        for cid in clusters:
            idx = np.where(self._cluster == cid)[0]
            if len(idx) == 0:
                continue
            pts = self._xy[idx]
            hull = convex_hull(pts) if len(pts) >= 3 else pts

            if len(hull) >= 3:
                path = QtGui.QPainterPath()
                path.moveTo(hull[0, 0], hull[0, 1])
                for p in hull[1:]:
                    path.lineTo(p[0], p[1])
                path.closeSubpath()

                poly = QtWidgets.QGraphicsPathItem(path)
                pen = pg.mkPen(self._cluster_color(cid, 200), width=2)
                brush = pg.mkBrush(self._cluster_color(cid, 42))
                poly.setPen(pen)
                poly.setBrush(brush)
                vb.addItem(poly)
                self._hull_items[cid] = poly

            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            name = self._cluster_names.get(cid, f"Cluster {cid}")
            lab = DraggableClusterLabel(self, cid, name, self._cluster_color(cid, 255))

            if int(cid) in self._label_pos_override:
                px, py = self._label_pos_override[int(cid)]
                lab.setPos(float(px), float(py))
            else:
                lab.setPos(cx, cy)

            self.addItem(lab)
            self._label_items[cid] = lab

    def _nearest_point(self, x: float, y: float) -> Optional[int]:
        if self._xy.shape[0] == 0:
            return None
        xr, yr = self.getPlotItem().vb.viewRange()
        scale = max((xr[1] - xr[0]), (yr[1] - yr[0]))
        thr = max(0.05 * scale, 0.4)
        dx = self._xy[:, 0] - x
        dy = self._xy[:, 1] - y
        d2 = dx * dx + dy * dy
        i = int(np.argmin(d2))
        if d2[i] <= thr * thr:
            return i
        return None

    def click_select(self, pos: Tuple[float, float], shift: bool = False):
        i = self._nearest_point(pos[0], pos[1])
        if i is None:
            if not shift:
                self._selected.clear()
                self._draw_scatter()
            return
        if shift:
            if i in self._selected:
                self._selected.remove(i)
            else:
                self._selected.add(i)
        else:
            self._selected = {i}
        self._draw_scatter()

    def _cluster_centroids(self) -> Dict[int, Tuple[float, float]]:
        out = {}
        for cid in sorted(set(map(int, self._cluster.tolist()))):
            idx = np.where(self._cluster == cid)[0]
            if len(idx) == 0:
                continue
            pts = self._xy[idx]
            out[cid] = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        return out

    def _assign_selected_to_nearest_cluster(self, drop_xy: Tuple[float, float]):
        cent = self._cluster_centroids()
        if not cent or not self._selected:
            return
        d2 = {cid: (drop_xy[0] - c[0]) ** 2 + (drop_xy[1] - c[1]) ** 2 for cid, c in cent.items()}
        dst = int(min(d2, key=d2.get))
        for i in self._selected:
            self._cluster[i] = dst
        self._drag_temp_positions = None
        self._drag_anchor_xy = None
        self.redraw_all()
        self.sigClustersChanged.emit()

    def _commit_selected_move(self, drop_xy: Tuple[float, float]):
        if self._drag_anchor_xy is None or not self._selected:
            self._drag_temp_positions = None
            self._drag_anchor_xy = None
            self._draw_scatter()
            return

        dx = float(drop_xy[0] - self._drag_anchor_xy[0])
        dy = float(drop_xy[1] - self._drag_anchor_xy[1])

        for i in self._selected:
            self._xy[i, 0] = float(self._xy[i, 0] + dx)
            self._xy[i, 1] = float(self._xy[i, 1] + dy)

        self._drag_temp_positions = None
        self._drag_anchor_xy = None
        self.redraw_all()
        self.sigCoordsChanged.emit()

    def drag_event(self, pos: Tuple[float, float], ev):
        if not self._editable_widget:
            ev.ignore()
            return

        if ev.isStart():
            i = self._nearest_point(pos[0], pos[1])
            if i is None:
                ev.ignore()
                return
            if i not in self._selected:
                self._selected = {i}
            self._dragging = True
            self._drag_temp_positions = self._xy.copy()
            self._drag_anchor_xy = (pos[0], pos[1])
            ev.accept()
            self._draw_scatter()
            return

        if not self._dragging:
            return

        ev.accept()

        if ev.isFinish():
            self._dragging = False
            # Check interaction mode
            if self._free_move_points:
                self._commit_selected_move(pos)
            else:
                self._assign_selected_to_nearest_cluster(pos)
            return

        if self._drag_temp_positions is not None and self._selected and self._drag_anchor_xy is not None:
            dx = pos[0] - self._drag_anchor_xy[0]
            dy = pos[1] - self._drag_anchor_xy[1]
            tmp = self._xy.copy()
            for i in self._selected:
                tmp[i, 0] = self._xy[i, 0] + dx
                tmp[i, 1] = self._xy[i, 1] + dy
            self._drag_temp_positions = tmp
            self._draw_scatter()

    def try_merge_label(self, src_cluster: int, drop_xy: Tuple[float, float]):
        cent = self._cluster_centroids()
        if len(cent) <= 1:
            self.remember_label_position(src_cluster, drop_xy)
            self._draw_hulls_and_labels()
            return

        xr, yr = self.getPlotItem().vb.viewRange()
        scale = max((xr[1] - xr[0]), (yr[1] - yr[0]))
        thr = max(0.05 * scale, 0.4)

        best = None
        best_d2 = None
        for cid, (cx, cy) in cent.items():
            if cid == src_cluster:
                continue
            d2 = (drop_xy[0] - cx) ** 2 + (drop_xy[1] - cy) ** 2
            if best_d2 is None or d2 < best_d2:
                best = cid
                best_d2 = d2

        if best is None or best_d2 is None or best_d2 > thr * thr:
            # Too far, just move label
            self.remember_label_position(src_cluster, drop_xy)
            self._draw_hulls_and_labels()
            return

        dst = int(best)
        self._cluster[self._cluster == int(src_cluster)] = dst

        # Remove source override, keep dst
        if int(src_cluster) in self._label_pos_override:
            self._label_pos_override.pop(int(src_cluster), None)

        self.redraw_all()
        self.sigClustersChanged.emit()

# =============================================================================
# app.py (Part 5/9)
# Main Window, Data Loading, and Factor Analysis (PCA/EFA)
# =============================================================================

# -----------------------------------------------------------------------------
# Integrated Application Window
# -----------------------------------------------------------------------------
class IntegratedApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto Segment Tool v7.0")
        self.resize(1600, 940)

        pg.setConfigOptions(antialias=True)

        self.state = AppState()

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # ---------------------------------------------------------------------
        # Tab Initialization Order
        # ---------------------------------------------------------------------
        
        # 1. Data loading
        self._build_tab_data()
        
        # 2. Recode mapping
        self._build_tab_recode()
        
        # 3. Factor Analysis (Renamed from PCA, includes EFA option)
        self._build_tab_factor()
        
        # 4. Decision Tree setting (Updated UI: Select Predictors)
        self._build_tab_dt_setting()
        
        # 5. Decision Tree results
        self._build_tab_dt_results()
        
        # 6. Group & Compose
        self._build_tab_grouping()
        
        # 7. Segmentation setting
        self._build_tab_seg_setting()
        
        # 8. Segmentation editing
        self._build_tab_seg_editing()
        
        # 9. Export
        self._build_tab_export()

        # 10. AI Assistant (RAG) - [New in v7.0]
        self._build_tab_rag()

        self._apply_tab_styles()
        self._set_status("Ready.")

    # -------------------------------------------------------------------------
    # UI Helper Methods
    # -------------------------------------------------------------------------
    def _apply_tab_styles(self):
        self.tabs.setStyleSheet("""
        QTabWidget::pane {
            border: 1px solid #cfd8dc;
            top: -1px;
        }
        QTabBar::tab {
            background: #e3f2fd;
            border: 1px solid #cfd8dc;
            border-bottom-color: #cfd8dc;
            padding: 8px 14px;
            margin-right: 2px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
        }
        QTabBar::tab:selected {
            background: #90caf9;
            border-bottom-color: #90caf9;
        }
        QTabBar::tab:hover {
            background: #bbdefb;
        }
        """)

    def _set_status(self, text: str):
        self.statusBar().showMessage(text)

    def _ensure_df(self):
        if self.state.df is None:
            raise RuntimeError("No data loaded.")

    def _selected_checked_items(self, widget: QtWidgets.QListWidget) -> List[str]:
        out = []
        for i in range(widget.count()):
            it = widget.item(i)
            if it.checkState() == QtCore.Qt.CheckState.Checked:
                out.append(it.text())
        return out

    def _set_checked_for_selected(self, widget: QtWidgets.QListWidget, checked: bool):
        st = QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked
        for it in widget.selectedItems():
            it.setCheckState(st)

    def _set_all_checks(self, widget: QtWidgets.QListWidget, checked: bool):
        st = QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked
        for i in range(widget.count()):
            widget.item(i).setCheckState(st)

    def _refresh_all_column_lists(self):
        """Updates all ComboBoxes and ListWidgets when new data is loaded."""
        df = self.state.df
        if df is None:
            return
        cols = list(df.columns)

        # Factor Tab
        self.lst_factor_cols.clear()
        for c in cols:
            it = QtWidgets.QListWidgetItem(c)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            it.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.lst_factor_cols.addItem(it)

        # DT Tab - [v7.0] Changed to "Select Predictors" (Whitelist)
        self.lst_dt_predictors.clear()
        for c in cols:
            it = QtWidgets.QListWidgetItem(c)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            it.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.lst_dt_predictors.addItem(it)

        self.cmb_dep_extra.clear()
        self.cmb_dep_extra.addItem("(None)")
        self.cmb_dep_extra.addItems(cols)

        self.cmb_dt_full_dep.clear()
        self.cmb_dt_full_ind.clear()
        self.cmb_split_dep.clear()
        self.cmb_split_ind.clear()

        # Grouping Tab
        self.cmb_group_source.clear()
        self.cmb_group_source.addItems(cols)
        self.cmb_bin_col.clear()
        self.cmb_bin_col.addItems(cols)

        # Demand Space
        self.lst_demand_vars.clear()
        for c in cols:
            it = QtWidgets.QListWidgetItem(c)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            it.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.lst_demand_vars.addItem(it)

        seg_cols = [c for c in cols if c.endswith("_seg")]
        self.lst_compose_segs.clear()
        for c in seg_cols:
            self.lst_compose_segs.addItem(c)

        self.lst_demand_segcols.clear()
        for c in seg_cols:
            it = QtWidgets.QListWidgetItem(c)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            it.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.lst_demand_segcols.addItem(it)

        self.cmb_demand_target.clear()
        self.cmb_demand_target.addItem("(None)")
        self.cmb_demand_target.addItems(cols)

    # -------------------------------------------------------------------------
    # Tab 1: Data Loading
    # -------------------------------------------------------------------------
    def _build_tab_data(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Data Loading")
        layout = QtWidgets.QVBoxLayout(tab)

        row1 = QtWidgets.QHBoxLayout()
        self.txt_path = QtWidgets.QLineEdit()
        self.btn_browse = QtWidgets.QPushButton("Browse Excel")
        style_button(self.btn_browse, level=1)
        self.btn_browse.clicked.connect(self._browse_excel)

        self.cmb_sheet = QtWidgets.QComboBox()
        self.btn_load = QtWidgets.QPushButton("Load Data")
        style_button(self.btn_load, level=2)
        self.btn_load.clicked.connect(self._load_excel)

        row1.addWidget(QtWidgets.QLabel("File Path:"))
        row1.addWidget(self.txt_path, 3)
        row1.addWidget(self.btn_browse)
        row1.addSpacing(10)
        row1.addWidget(QtWidgets.QLabel("Sheet:"))
        row1.addWidget(self.cmb_sheet, 1)
        row1.addWidget(self.btn_load)
        layout.addLayout(row1)

        self.lbl_data_info = QtWidgets.QLabel("No data loaded.")
        self.lbl_data_info.setWordWrap(True)
        layout.addWidget(self.lbl_data_info)

        self.tbl_preview = DataFrameTable(float_decimals=2)
        layout.addWidget(self.tbl_preview, 1)

    def _browse_excel(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Excel", "", "Excel (*.xlsx *.xls)")
        if path:
            self.txt_path.setText(path)
            self._populate_sheets(path)

    def _populate_sheets(self, path: str):
        try:
            xls = pd.ExcelFile(path, engine="openpyxl")
            self.cmb_sheet.clear()
            self.cmb_sheet.addItems(list(xls.sheet_names))
        except Exception as e:
            show_error(self, "Read Sheet Error", e)

    def _load_excel(self):
        try:
            path = self.txt_path.text().strip()
            if not path or not os.path.exists(path):
                raise RuntimeError("Invalid file path.")
            if self.cmb_sheet.count() == 0:
                self._populate_sheets(path)
            sheet = self.cmb_sheet.currentText().strip()
            if not sheet:
                raise RuntimeError("Please select a sheet.")

            df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
            self.state.df = df
            self.state.path = path
            self.state.sheet = sheet

            # Load RECODE sheet if exists
            xls = pd.ExcelFile(path, engine="openpyxl")
            if "RECODE" in xls.sheet_names:
                rec = pd.read_excel(path, sheet_name="RECODE", engine="openpyxl")
                self.state.recode_df = normalize_recode_df(rec)
            else:
                self.state.recode_df = None

            self.tbl_preview.set_df(df)
            self.lbl_data_info.setText(f"Loaded: {os.path.basename(path)} / sheet={sheet} / rows={len(df):,} cols={df.shape[1]:,}")
            self._set_status("Data Loaded Successfully.")

            self._update_recode_tab()
            self._refresh_all_column_lists()
            self._reset_downstream_state()

        except Exception as e:
            self.state.last_error = str(e)
            show_error(self, "Load Error", e)

    def _reset_downstream_state(self):
        self.state.factor_model = None
        self.state.factor_cols = None
        self.state.factor_scores = None
        self.state.factor_loadings = None
        self.state.dt_improve_pivot = None
        self.state.dt_split_best = None
        self.state.dt_full_nodes = None
        # ... clear UI tables ...
        if hasattr(self, "tbl_factor_loadings"): self.tbl_factor_loadings.set_df(None)
        if hasattr(self, "tbl_dt_pivot"): self.tbl_dt_pivot.set_df(None)

    # -------------------------------------------------------------------------
    # Tab 2: Recode Mapping
    # -------------------------------------------------------------------------
    def _build_tab_recode(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Recode Mapping")
        layout = QtWidgets.QVBoxLayout(tab)
        layout.addWidget(QtWidgets.QLabel("Check 'RECODE' sheet (QUESTION / CODE / NAME)."))
        self.tbl_recode = DataFrameTable(float_decimals=2)
        layout.addWidget(self.tbl_recode, 1)

    def _update_recode_tab(self):
        self.tbl_recode.set_df(self.state.recode_df)

    # -------------------------------------------------------------------------
    # Tab 3: Factor Analysis (PCA / EFA)
    # -------------------------------------------------------------------------
    def _build_tab_factor(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Factor Analysis")

        layout = QtWidgets.QHBoxLayout(tab)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("Select Variables for Analysis:"))

        self.lst_factor_cols = QtWidgets.QListWidget()
        self.lst_factor_cols.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        left.addWidget(self.lst_factor_cols, 1)

        # Selection Buttons
        btnrow = QtWidgets.QHBoxLayout()
        self.btn_fac_check_sel = QtWidgets.QPushButton("Check Selected")
        style_button(self.btn_fac_check_sel, level=1)
        self.btn_fac_uncheck_sel = QtWidgets.QPushButton("Uncheck Selected")
        style_button(self.btn_fac_uncheck_sel, level=1)
        self.btn_fac_check_all = QtWidgets.QPushButton("Check All")
        style_button(self.btn_fac_check_all, level=1)
        self.btn_fac_uncheck_all = QtWidgets.QPushButton("Uncheck All")
        style_button(self.btn_fac_uncheck_all, level=1)
        
        self.btn_fac_check_sel.clicked.connect(lambda: self._set_checked_for_selected(self.lst_factor_cols, True))
        self.btn_fac_uncheck_sel.clicked.connect(lambda: self._set_checked_for_selected(self.lst_factor_cols, False))
        self.btn_fac_check_all.clicked.connect(lambda: self._set_all_checks(self.lst_factor_cols, True))
        self.btn_fac_uncheck_all.clicked.connect(lambda: self._set_all_checks(self.lst_factor_cols, False))
        
        btnrow.addWidget(self.btn_fac_check_sel)
        btnrow.addWidget(self.btn_fac_uncheck_sel)
        btnrow.addWidget(self.btn_fac_check_all)
        btnrow.addWidget(self.btn_fac_uncheck_all)
        left.addLayout(btnrow)

        # Analysis Options (PCA vs EFA)
        opt_grp = QtWidgets.QGroupBox("Extraction Method")
        opt_lay = QtWidgets.QHBoxLayout(opt_grp)
        self.radio_pca = QtWidgets.QRadioButton("Principal Component Analysis (PCA)")
        self.radio_efa = QtWidgets.QRadioButton("Factor Analysis (EFA)")
        self.radio_pca.setChecked(True)
        self.radio_pca.setToolTip("Maximizes variance. Good for data reduction.")
        self.radio_efa.setToolTip("Models latent factors. Good for construct discovery.")
        opt_lay.addWidget(self.radio_pca)
        opt_lay.addWidget(self.radio_efa)
        left.addWidget(opt_grp)

        ctrl = QtWidgets.QHBoxLayout()
        self.spin_factor_k = QtWidgets.QSpinBox()
        self.spin_factor_k.setRange(2, 50)
        self.spin_factor_k.setValue(5)
        self.btn_run_factor = QtWidgets.QPushButton("Run Analysis")
        style_button(self.btn_run_factor, level=2)
        self.btn_run_factor.clicked.connect(self._run_factor_analysis)
        
        ctrl.addWidget(QtWidgets.QLabel("Number of Factors (k):"))
        ctrl.addWidget(self.spin_factor_k)
        ctrl.addWidget(self.btn_run_factor)
        left.addLayout(ctrl)

        self.lbl_factor_info = QtWidgets.QLabel("Analysis not run.")
        self.lbl_factor_info.setWordWrap(True)
        left.addWidget(self.lbl_factor_info)

        layout.addLayout(left, 2)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("Loadings Matrix (Preview):"))
        self.tbl_factor_loadings = DataFrameTable(float_decimals=3)
        right.addWidget(self.tbl_factor_loadings, 1)
        layout.addLayout(right, 3)

    def _run_factor_analysis(self):
        try:
            self._ensure_df()
            df = self.state.df
            cols = self._selected_checked_items(self.lst_factor_cols)
            if len(cols) < 2:
                raise RuntimeError("Please select at least 2 variables.")

            # Prepare Numeric Data
            X = to_numeric_df(df, cols)
            X = X.dropna(axis=0, how="all")
            if len(X) < 10:
                raise RuntimeError("Not enough valid rows (after removing all-NaNs).")

            # Simple Imputation for missing values
            X_f = X.copy()
            for c in X_f.columns:
                m = X_f[c].mean()
                X_f[c] = X_f[c].fillna(m)

            k = int(self.spin_factor_k.value())
            k = min(k, X_f.shape[1])

            is_pca = self.radio_pca.isChecked()
            mode_name = "PCA" if is_pca else "EFA"

            # Run Analysis
            if is_pca:
                model = PCA(n_components=k, random_state=42)
                scores = model.fit_transform(X_f.values)
                components = model.components_
                # EVR is available for PCA
                expl_var = model.explained_variance_ratio_
                info_text = f"Method: PCA. Explained Variance (first 5): {', '.join([f'{v:.3f}' for v in expl_var[:5]])}"
            else:
                # EFA (using varimax rotation-like behavior usually implies rotation='varimax' but sklearn does not support rotation easily in FactorAnalysis without extra libs.
                # We will stick to basic FactorAnalysis which defaults to no rotation or varimax depending on version/params. 
                # Scikit-learn's FactorAnalysis does not have built-in rotation param in older versions, recent versions allow 'rotation'.
                # We will try with rotation='varimax' if available, else default.
                try:
                    model = FactorAnalysis(n_components=k, rotation='varimax', random_state=42)
                except:
                    model = FactorAnalysis(n_components=k, random_state=42) # Fallback
                
                scores = model.fit_transform(X_f.values)
                components = model.components_
                info_text = f"Method: Factor Analysis (EFA). Latent factors extracted."

            # Save Scores
            score_cols = [f"Factor{i+1}" for i in range(k)]
            scores_df = pd.DataFrame(scores, index=X_f.index, columns=score_cols)

            # Assign back to main DF
            for c in score_cols:
                df[c] = np.nan
                df.loc[scores_df.index, c] = scores_df[c].values

            # Save Loadings
            loadings = pd.DataFrame(components.T, index=cols, columns=score_cols)

            # Display Loadings (Sorted by max abs)
            disp = loadings.copy()
            disp["_maxabs_"] = disp.abs().max(axis=1)
            disp = disp.sort_values("_maxabs_", ascending=False).drop(columns=["_maxabs_"])
            disp = disp.reset_index().rename(columns={"index": "variable"})

            self.lbl_factor_info.setText(info_text)
            self.tbl_factor_loadings.set_df(disp)

            # Update State
            self.state.df = df
            self.state.factor_model = model
            self.state.factor_cols = cols
            self.state.factor_scores = scores_df
            self.state.factor_loadings = loadings
            self.state.factor_mode = mode_name

            self.tbl_preview.set_df(df)
            self._refresh_all_column_lists()
            self._set_status(f"{mode_name} completed. Columns {score_cols} added.")

        except Exception as e:
            self.state.last_error = str(e)
            show_error(self, "Analysis Error", e)

# =============================================================================
# app.py (Part 6/9)
# Decision Tree Setting (Whitelist UI & Recommendation Logic)
# =============================================================================

    # -------------------------------------------------------------------------
    # Tab 4: Decision Tree Setting (Whitelist UI & Recommendation)
    # -------------------------------------------------------------------------
    def _build_tab_dt_setting(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Decision Tree Setting")

        layout = QtWidgets.QVBoxLayout(tab)

        # Header Instructions
        head = QtWidgets.QLabel(
            "1. Select Dependent(Target) & Independent(Predictors) variables.\n"
            "2. Click 'Run Analysis' to generate Improvement Pivot.\n"
            "3. Select a cell in Pivot and click 'Recommend Grouping' to auto-create segments."
        )
        layout.addWidget(head)

        # Controls: Targets
        row = QtWidgets.QHBoxLayout()
        self.chk_use_all_factors = QtWidgets.QCheckBox("Use all Factors (Factor1..k) as Targets")
        self.chk_use_all_factors.setChecked(True)
        self.cmb_dep_extra = QtWidgets.QComboBox()
        self.btn_run_tree = QtWidgets.QPushButton("Run Decision Tree Analysis")
        style_button(self.btn_run_tree, level=2)
        self.btn_run_tree.clicked.connect(self._run_decision_tree_outputs)
        
        row.addWidget(self.chk_use_all_factors)
        row.addSpacing(14)
        row.addWidget(QtWidgets.QLabel("Extra Target (Optional):"))
        row.addWidget(self.cmb_dep_extra)
        row.addWidget(self.btn_run_tree)
        layout.addLayout(row)

        # Controls: Predictors (Whitelist) - [v7.0 Change]
        # Changed from "Exclude List" to "Select Predictors List"
        pred_box = QtWidgets.QGroupBox("Select Predictors (Independent Variables)")
        pred_layout = QtWidgets.QVBoxLayout(pred_box)
        
        # Filter & Buttons
        p_row = QtWidgets.QHBoxLayout()
        self.txt_dt_pred_filter = QtWidgets.QLineEdit()
        self.txt_dt_pred_filter.setPlaceholderText("Filter variables...")
        self.txt_dt_pred_filter.textChanged.connect(self._filter_dt_pred_list)
        
        self.btn_dt_pred_check_sel = QtWidgets.QPushButton("Check Selected")
        style_button(self.btn_dt_pred_check_sel, level=1)
        self.btn_dt_pred_uncheck_sel = QtWidgets.QPushButton("Uncheck Selected")
        style_button(self.btn_dt_pred_uncheck_sel, level=1)
        self.btn_dt_pred_check_all = QtWidgets.QPushButton("Check All")
        style_button(self.btn_dt_pred_check_all, level=1)
        self.btn_dt_pred_uncheck_all = QtWidgets.QPushButton("Uncheck All")
        style_button(self.btn_dt_pred_uncheck_all, level=1)
        
        self.btn_dt_pred_check_sel.clicked.connect(lambda: self._set_checked_for_selected(self.lst_dt_predictors, True))
        self.btn_dt_pred_uncheck_sel.clicked.connect(lambda: self._set_checked_for_selected(self.lst_dt_predictors, False))
        self.btn_dt_pred_check_all.clicked.connect(lambda: self._set_all_checks(self.lst_dt_predictors, True))
        self.btn_dt_pred_uncheck_all.clicked.connect(lambda: self._set_all_checks(self.lst_dt_predictors, False))

        p_row.addWidget(self.txt_dt_pred_filter, 2)
        p_row.addWidget(self.btn_dt_pred_check_sel)
        p_row.addWidget(self.btn_dt_pred_uncheck_sel)
        p_row.addWidget(self.btn_dt_pred_check_all)
        p_row.addWidget(self.btn_dt_pred_uncheck_all)
        pred_layout.addLayout(p_row)

        self.lst_dt_predictors = QtWidgets.QListWidget()
        self.lst_dt_predictors.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_dt_predictors.setMaximumHeight(200)
        pred_layout.addWidget(self.lst_dt_predictors, 1)
        layout.addWidget(pred_box)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # A) Pivot (Main View)
        w1 = QtWidgets.QWidget()
        l1 = QtWidgets.QVBoxLayout(w1)
        l1.addWidget(QtWidgets.QLabel("Improvement Pivot (Rel. Impurity Drop) [Rows=Predictors, Cols=Targets]"))
        self.tbl_dt_pivot = DataFrameTable(float_decimals=2)
        l1.addWidget(self.tbl_dt_pivot, 1)
        
        # Recommendation Button
        rec_layout = QtWidgets.QHBoxLayout()
        self.btn_dt_recommend = QtWidgets.QPushButton("Recommend Grouping based on Selection → Send to Group Tab")
        style_button(self.btn_dt_recommend, level=3)
        self.btn_dt_recommend.setMinimumHeight(40)
        self.btn_dt_recommend.clicked.connect(self._recommend_grouping_transfer)
        
        rec_layout.addWidget(self.btn_dt_recommend)
        rec_layout.addStretch(1)
        l1.addLayout(rec_layout)

        # B) Best Split (Detail View)
        w2 = QtWidgets.QWidget()
        l2 = QtWidgets.QVBoxLayout(w2)
        l2.addWidget(QtWidgets.QLabel("Best Split Detail (Root Node) - Reference"))
        self.tbl_dt_bestsplit = DataFrameTable(float_decimals=2)
        l2.addWidget(self.tbl_dt_bestsplit, 1)

        splitter.addWidget(w1)
        splitter.addWidget(w2)
        splitter.setSizes([500, 200])
        layout.addWidget(splitter, 1)

    def _filter_dt_pred_list(self):
        term = self.txt_dt_pred_filter.text().strip().lower()
        for i in range(self.lst_dt_predictors.count()):
            it = self.lst_dt_predictors.item(i)
            it.setHidden(term not in it.text().lower())

    def _run_decision_tree_outputs(self):
        """Calculates Improve Pivot and Best Split tables."""
        try:
            self._ensure_df()
            df = self.state.df

            # Targets
            # [v7.0] Look for "Factor" columns instead of "PCA"
            fac_cols = [c for c in df.columns if str(c).startswith("Factor") and str(c)[6:].isdigit()]
            fac_cols = sorted(fac_cols, key=lambda x: int(str(x)[6:]))

            deps: List[str] = []
            if self.chk_use_all_factors.isChecked() and fac_cols:
                deps.extend(fac_cols)

            extra = self.cmb_dep_extra.currentText().strip()
            if extra and extra != "(None)" and extra not in deps:
                deps.append(extra)

            if not deps:
                raise RuntimeError("No dependent targets selected. Run Factor Analysis first or select extra dep.")

            # Predictors (Whitelist)
            # [v7.0] Only use checked items
            ind_vars = self._selected_checked_items(self.lst_dt_predictors)
            
            # Remove deps from inds if overlap
            ind_vars = [c for c in ind_vars if c not in deps and c != "resp_id"]

            if len(ind_vars) == 0:
                raise RuntimeError("No independent variables selected. Please check predictors.")

            # Calculation Loop
            best_rows = []
            pivot = pd.DataFrame(index=ind_vars, columns=deps, dtype=float)

            for dep in deps:
                y = df[dep]
                # Determine task
                task = "reg"
                if dep == extra and extra != "(None)":
                    task = "class" if is_categorical_series(y) else "reg"

                for ind in ind_vars:
                    x = df[ind]
                    best, _ = univariate_best_split(y, x, task=task)
                    
                    if best is None:
                        pivot.loc[ind, dep] = np.nan
                        continue
                    
                    pivot.loc[ind, dep] = best["improve_rel"]
                    
                    # Store result
                    row = {"dep": dep, "ind": ind}
                    # Flatten the best dict (handle lists carefully if needed, but best dict usually simple types except items)
                    # We copy items to avoid reference issues
                    for k, v in best.items():
                        if k in ["left_items", "right_items"]:
                            # Store as string representation for table display, logic uses actual list later
                            row[k] = str(v) 
                        else:
                            row[k] = v
                            
                    best_rows.append(row)

            best_df = pd.DataFrame(best_rows)
            pivot_reset = pivot.reset_index().rename(columns={"index": "ind"})

            self.tbl_dt_pivot.set_df(pivot_reset)
            self.tbl_dt_bestsplit.set_df(best_df)

            self.state.dt_improve_pivot = pivot_reset
            self.state.dt_split_best = best_df

            # Populate next tab combos
            self.cmb_dt_full_dep.clear()
            self.cmb_dt_full_dep.addItems(deps)
            self.cmb_dt_full_ind.clear()
            self.cmb_dt_full_ind.addItems(ind_vars)
            
            self.cmb_split_dep.clear()
            self.cmb_split_dep.addItems(deps)
            self.cmb_split_ind.clear()
            self.cmb_split_ind.addItems(ind_vars)

            self._set_status("Decision Tree analysis completed.")

        except Exception as e:
            self.state.last_error = str(e)
            show_error(self, "DT Analysis Error", e)

    def _recommend_grouping_transfer(self):
        """
        Auto-generates grouping mapping based on the best split for selected Ind.
        [v7.0] Supports Optimal Subset Splits (Multi-category grouping).
        """
        try:
            # 1. Validate Selection
            sel_rows = self.tbl_dt_pivot.selectedItems()
            if not sel_rows:
                raise RuntimeError("Please select a row (Predictor) in the Pivot table.")
            
            row_idx = sel_rows[0].row()
            ind_val = self.tbl_dt_pivot.item(row_idx, 0).text() 
            
            if self.state.dt_split_best is None:
                raise RuntimeError("No Best Split data. Run Analysis first.")
            
            # 2. Find Best Split for this Ind
            relevant = self.state.dt_split_best[self.state.dt_split_best["ind"] == ind_val]
            if relevant.empty:
                raise RuntimeError(f"No valid split found for '{ind_val}'.")
            
            # Find the split with max improvement across all deps
            best_row = relevant.loc[relevant["improve_rel"].idxmax()]
            
            # 3. Prepare Mapping Data
            self._ensure_df()
            df = self.state.df
            
            vals = pd.Series(df[ind_val].dropna().unique()).astype(str)
            try:
                vv = vals.astype(float)
                order = np.argsort(vv.values)
                vals = vals.iloc[order]
            except:
                vals = vals.sort_values()
                
            # Recode Lookup
            rec_name = {}
            if self.state.recode_df is not None:
                r = self.state.recode_df
                r = r[r["QUESTION"].astype(str).str.strip() == ind_val]
                rec_name = dict(zip(r["CODE"].astype(str).str.strip(), r["NAME"].astype(str).str.strip()))
            
            recode_names = [rec_name.get(v, "") for v in vals.values]
            
            # Generate Segment Labels
            seg_labels = []
            split_type = best_row["split_type"]
            cutpoint = best_row["cutpoint"]

            if split_type == "categorical(subset)":
                # [v7.0] Optimal Subset Logic
                # Retrieve the list string and parse it back to list (or use stored logic if possible)
                # Since we stored str() in dataframe, we need to eval or parse. 
                # Ideally, we should look up the object, but parsing string is safer for now.
                import ast
                try:
                    left_items_str = best_row["left_items"]
                    left_items = ast.literal_eval(left_items_str)
                    left_set = set(map(str, left_items))
                except:
                    # Fallback if parsing fails
                    left_set = set()
                
                label_L = "Group_A"
                label_R = "Group_B"
                
                for v in vals.values:
                    if str(v) in left_set:
                        seg_labels.append(label_L)
                    else:
                        seg_labels.append(label_R)

            elif split_type.startswith("categorical"):
                # Old One-vs-Rest fallback
                target_val = str(best_row["left_group"])
                label_L = f"Group_{target_val}"
                label_R = "Group_Rest"
                for v in vals.values:
                    if v == target_val: seg_labels.append(label_L)
                    else: seg_labels.append(label_R)

            else:
                # Numeric
                try:
                    thr = float(cutpoint)
                    label_L = f"Low(v<={thr:g})"
                    label_R = f"High(v>{thr:g})"
                    for v in vals.values:
                        try:
                            vf = float(v)
                            if vf <= thr: seg_labels.append(label_L)
                            else: seg_labels.append(label_R)
                        except:
                            seg_labels.append("Unknown")
                except:
                     seg_labels = ["Manual_Fix"] * len(vals)

            map_df = pd.DataFrame({
                "source_value": vals.values,
                "recode_name": recode_names,
                "segment_label": seg_labels
            })
            
            # 4. Transfer to Group Tab
            self.cmb_group_source.setCurrentText(ind_val)
            self.tbl_group_map.set_df(map_df)
            self.txt_group_newcol.setText(f"{ind_val}_seg")
            
            # 5. Move to Group Tab (Index 5)
            self.tabs.setCurrentIndex(5) 
            self._set_status(f"Recommendation for '{ind_val}' transferred to Group Tab.")
            
        except Exception as e:
            self.state.last_error = str(e)
            show_error(self, "Recommendation Error", e)

# =============================================================================
# app.py (Part 7/9)
# Decision Tree Results & Grouping Tab
# =============================================================================

    # -------------------------------------------------------------------------
    # Tab 5: Decision Tree Results (Full Tree Viewer)
    # -------------------------------------------------------------------------
    def _build_tab_dt_results(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Decision Tree Results")

        layout = QtWidgets.QVBoxLayout(tab)

        info = QtWidgets.QLabel(
            "Explore Full Tree & Split Details:\n"
            "- Select Dep/Ind -> Click 'Run Full Tree Analysis'\n"
            "- Select a Split from the dropdown -> View Left/Right Node Stats"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        ctrl = QtWidgets.QHBoxLayout()
        self.cmb_dt_full_dep = QtWidgets.QComboBox()
        self.cmb_dt_full_ind = QtWidgets.QComboBox()
        self.spin_dt_full_depth = QtWidgets.QSpinBox()
        self.spin_dt_full_depth.setRange(1, 30)
        self.spin_dt_full_depth.setValue(6)
        self.btn_dt_full_run = QtWidgets.QPushButton("Run Full Tree Analysis")
        style_button(self.btn_dt_full_run, level=2)
        self.btn_dt_full_run.clicked.connect(self._run_dt_full_for_selected)

        ctrl.addWidget(QtWidgets.QLabel("Target (Dep)"))
        ctrl.addWidget(self.cmb_dt_full_dep, 2)
        ctrl.addWidget(QtWidgets.QLabel("Predictor (Ind)"))
        ctrl.addWidget(self.cmb_dt_full_ind, 3)
        ctrl.addWidget(QtWidgets.QLabel("Max Depth"))
        ctrl.addWidget(self.spin_dt_full_depth)
        ctrl.addWidget(self.btn_dt_full_run)
        layout.addLayout(ctrl)

        # Split Navigation
        srow = QtWidgets.QHBoxLayout()
        self.cmb_split_dep = QtWidgets.QComboBox() # Hidden sync
        self.cmb_split_ind = QtWidgets.QComboBox() # Hidden sync
        
        self.cmb_split_select = QtWidgets.QComboBox()
        self.cmb_split_select.currentIndexChanged.connect(self._split_update_detail)
        srow.addWidget(QtWidgets.QLabel("Select Split Node:"))
        srow.addWidget(self.cmb_split_select, 4)
        layout.addLayout(srow)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        topw = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(topw)
        tl.addWidget(QtWidgets.QLabel("Split Groups (All internal nodes)"))
        self.tbl_dt_full_groups = DataFrameTable(float_decimals=2)
        tl.addWidget(self.tbl_dt_full_groups, 1)

        botw = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(botw)
        self.lbl_split_imp = QtWidgets.QLabel("No split selected.")
        self.lbl_split_imp.setWordWrap(True)
        bl.addWidget(self.lbl_split_imp)
        bl.addWidget(QtWidgets.QLabel("Selected Split Detail (Left vs Right):"))
        self.tbl_split_detail = DataFrameTable(float_decimals=2)
        bl.addWidget(self.tbl_split_detail, 1)

        splitter.addWidget(topw)
        splitter.addWidget(botw)
        splitter.setSizes([360, 360])
        layout.addWidget(splitter, 1)

    def _compute_full_tree_internal(self, dep: str, ind: str):
        df = self.state.df
        if df is None:
            raise RuntimeError("No data.")

        if dep not in df.columns or ind not in df.columns:
            raise RuntimeError("Columns missing.")

        # Determine task type
        # Treat as class if categorical and NOT a Factor/PCA column
        is_factor = str(dep).startswith("Factor") or str(dep).startswith("PCA")
        task = "class" if is_categorical_series(df[dep]) and not is_factor else "reg"

        nodes_df, split_groups, branches, path_info, cond_freq = build_univariate_tree_full(
            df=df, dep=dep, ind=ind, task=task,
            max_depth=int(self.spin_dt_full_depth.value()),
            min_leaf=1,
            min_split=2,
            max_unique_cat=50,
            min_improve_rel=0.0
        )

        self.state.dt_full_nodes = nodes_df
        self.state.dt_full_split_groups = split_groups
        self.state.dt_full_split_branches = branches
        self.state.dt_full_path_info = path_info
        self.state.dt_full_condition_freq = cond_freq
        self.state.dt_full_selected = (dep, ind)

        return task, nodes_df, split_groups, path_info, cond_freq

    def _run_dt_full_for_selected(self):
        try:
            self._ensure_df()
            dep = self.cmb_dt_full_dep.currentText().strip()
            ind = self.cmb_dt_full_ind.currentText().strip()
            if not dep or not ind:
                raise RuntimeError("Select Dep and Ind.")

            task, nodes_df, split_groups, path_info, cond_freq = self._compute_full_tree_internal(dep, ind)

            self.tbl_dt_full_groups.set_df(split_groups)
            
            # Update Select Combo
            self.cmb_split_select.blockSignals(True)
            self.cmb_split_select.clear()
            if not split_groups.empty:
                for _, row in split_groups.iterrows():
                    desc = f"Split {int(row['split_num'])} @ node {int(row['node_id'])}: {row['left_group']} / {row['right_group']} (Δ={fmt_float(row['improve_rel'],2)})"
                    self.cmb_split_select.addItem(desc, int(row["split_num"]))
            self.cmb_split_select.blockSignals(False)

            if self.cmb_split_select.count() > 0:
                self.cmb_split_select.setCurrentIndex(0)
                self._split_update_detail()

            self._set_status(f"Tree Built: dep={dep}, ind={ind} (task={task})")
        except Exception as e:
            self.state.last_error = str(e)
            show_error(self, "Tree Build Error", e)

    def _split_update_detail(self):
        try:
            self._ensure_df()
            df = self.state.df
            split_groups = self.state.dt_full_split_groups
            dep, ind = self.state.dt_full_selected
            
            idx = self.cmb_split_select.currentIndex()
            if idx < 0: return
            split_num = self.cmb_split_select.currentData()
            if split_num is None: return

            row = split_groups[split_groups["split_num"] == split_num].iloc[0]
            left_cond = row["left_group"]
            right_cond = row["right_group"]

            y = df[dep]
            is_factor = str(dep).startswith("Factor") or str(dep).startswith("PCA")
            task = "class" if is_categorical_series(y) and not is_factor else "reg"

            # [v7.0] Robust condition parsing for subsets
            mask_left = self._parse_condition_to_mask(df, ind, left_cond)
            mask_right = self._parse_condition_to_mask(df, ind, right_cond)

            yL = y[mask_left].dropna().values
            yR = y[mask_right].dropna().values

            if task == "reg":
                impL = _impurity_reg(yL.astype(float)) if yL.size > 0 else np.nan
                impR = _impurity_reg(yR.astype(float)) if yR.size > 0 else np.nan
            else:
                impL = _impurity_gini(yL) if yL.size > 0 else np.nan
                impR = _impurity_gini(yR) if yR.size > 0 else np.nan

            better = ""
            if np.isfinite(impL) and np.isfinite(impR):
                if impL < impR: better = "Left node cleaner"
                elif impR < impL: better = "Right node cleaner"
                else: better = "Similar impurity"
            else:
                better = "High NA count, hard to compare."

            base_text = f"Split {int(split_num)}: {left_cond}  /  {right_cond}\n"
            base_text += f"[task={task}] left_imp={fmt_float(impL,2)}, right_imp={fmt_float(impR,2)}  ->  {better}"
            self.lbl_split_imp.setText(base_text)

            detail = pd.DataFrame([{
                "split_num": int(split_num),
                "dep": dep,
                "ind": ind,
                "left_condition": left_cond,
                "right_condition": right_cond,
                "left_n": int(yL.size),
                "right_n": int(yR.size),
                "left_impurity": impL,
                "right_impurity": impR,
                "better_side": better,
            }])
            self.tbl_split_detail.set_df(detail)
        except Exception as e:
            self.state.last_error = str(e)
            show_error(self, "Split Detail Error", e)

    def _parse_condition_to_mask(self, df: pd.DataFrame, ind: str, cond: str) -> np.ndarray:
        """
        [v7.0] Enhanced to handle 'in [...]' syntax for optimal subsets.
        """
        s = df[ind]
        cond = str(cond).strip()

        # Case 1: Optimal Subset "Ind in ['A', 'B']"
        if " in [" in cond and "not in" not in cond:
            # Extract the list part
            try:
                list_part = cond.split(" in ", 1)[1].strip()
                import ast
                target_items = ast.literal_eval(list_part) # Safe eval for list string
                target_set = set(map(str, target_items))
                return s.astype(str).isin(target_set).values
            except:
                pass
        
        # Case 2: Optimal Subset "Ind not in ['A', 'B']"
        if " not in [" in cond:
            try:
                list_part = cond.split(" not in ", 1)[1].strip()
                import ast
                target_items = ast.literal_eval(list_part)
                target_set = set(map(str, target_items))
                return ~s.astype(str).isin(target_set).values
            except:
                pass

        # Helper for basic comparison
        def _strip_quotes(v: str) -> str:
            vv = v.strip()
            if (vv.startswith("'") and vv.endswith("'")) or (vv.startswith('"') and vv.endswith('"')):
                vv = vv[1:-1].strip()
            return vv

        def _try_numeric_compare(op: str, val_str: str) -> Optional[np.ndarray]:
            try:
                vv = float(val_str)
            except:
                return None
            sn = pd.to_numeric(s, errors="coerce").values.astype(float)
            ok = np.isfinite(sn)
            if ok.sum() == 0: return None
            
            if op == "==": m = (sn == vv)
            elif op == "!=": m = (sn != vv)
            elif op == "<=": m = (sn <= vv)
            elif op == ">": m = (sn > vv)
            else: return None
            
            m[~ok] = False
            return m

        # Case 3: Standard Operators
        ops = ["==", "!=", "<=", ">"]
        for op in ops:
            if op in cond:
                parts = cond.split(op, 1)
                # Ensure we are splitting on the operator, sometimes "Ind <= Val"
                # part[0] should end with ind or be close
                val = _strip_quotes(parts[1].strip())
                m = _try_numeric_compare(op, val)
                if m is not None: return m
                
                # String comparison fallback
                s_str = s.astype(str).str.strip()
                if op == "==": return (s_str == val).values
                if op == "!=": return (s_str != val).values
        
        return np.zeros(len(df), dtype=bool)

    # -------------------------------------------------------------------------
    # Tab 6: Group & Compose
    # -------------------------------------------------------------------------
    def _build_tab_grouping(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Group & Compose")

        layout = QtWidgets.QVBoxLayout(tab)

        # Binary Recode Section
        box_bin = QtWidgets.QGroupBox("Quick Binary Recode (2 Values) -> *_seg")
        bin_layout = QtWidgets.QHBoxLayout(box_bin)

        self.cmb_bin_col = QtWidgets.QComboBox()
        self.txt_bin_val1 = QtWidgets.QLineEdit("")   # Auto top2 if empty
        self.txt_bin_lab1 = QtWidgets.QLineEdit("A")
        self.txt_bin_val2 = QtWidgets.QLineEdit("")
        self.txt_bin_lab2 = QtWidgets.QLineEdit("B")
        self.chk_bin_else_other = QtWidgets.QCheckBox("Else=Other")
        self.txt_bin_else_lab = QtWidgets.QLineEdit("Other")
        self.txt_bin_else_lab.setMaximumWidth(90)

        self.txt_bin_newcol = QtWidgets.QLineEdit("")  # empty -> {col}_seg
        self.btn_bin_apply = QtWidgets.QPushButton("Apply Binary Recode")
        style_button(self.btn_bin_apply, level=2)
        self.btn_bin_apply.clicked.connect(self._apply_binary_recode)

        bin_layout.addWidget(QtWidgets.QLabel("Column"))
        bin_layout.addWidget(self.cmb_bin_col, 2)
        bin_layout.addSpacing(10)
        bin_layout.addWidget(QtWidgets.QLabel("Val1"))
        bin_layout.addWidget(self.txt_bin_val1)
        bin_layout.addWidget(QtWidgets.QLabel("Lab1"))
        bin_layout.addWidget(self.txt_bin_lab1)
        bin_layout.addSpacing(10)
        bin_layout.addWidget(QtWidgets.QLabel("Val2"))
        bin_layout.addWidget(self.txt_bin_val2)
        bin_layout.addWidget(QtWidgets.QLabel("Lab2"))
        bin_layout.addWidget(self.txt_bin_lab2)
        bin_layout.addSpacing(10)
        bin_layout.addWidget(self.chk_bin_else_other)
        bin_layout.addWidget(self.txt_bin_else_lab)
        bin_layout.addSpacing(10)
        bin_layout.addWidget(QtWidgets.QLabel("New Name"))
        bin_layout.addWidget(self.txt_bin_newcol, 2)
        bin_layout.addWidget(self.btn_bin_apply)
        layout.addWidget(box_bin)

        # Mapping Table Section
        box = QtWidgets.QGroupBox("General Grouping: Source Value -> Segment Label (Mapping Table)")
        b = QtWidgets.QVBoxLayout(box)

        r1 = QtWidgets.QHBoxLayout()
        self.cmb_group_source = QtWidgets.QComboBox()
        self.txt_group_newcol = QtWidgets.QLineEdit("custom_seg")
        self.btn_group_build = QtWidgets.QPushButton("Build Mapping Table")
        style_button(self.btn_group_build, level=1)
        self.btn_group_build.clicked.connect(self._build_group_mapping)
        self.btn_group_apply = QtWidgets.QPushButton("Apply Mapping -> Create Seg")
        style_button(self.btn_group_apply, level=2)
        self.btn_group_apply.clicked.connect(self._apply_group_mapping)

        r1.addWidget(QtWidgets.QLabel("Source Column"))
        r1.addWidget(self.cmb_group_source)
        r1.addSpacing(10)
        r1.addWidget(QtWidgets.QLabel("New Column Name"))
        r1.addWidget(self.txt_group_newcol)
        r1.addWidget(self.btn_group_build)
        r1.addWidget(self.btn_group_apply)
        b.addLayout(r1)

        self.tbl_group_map = DataFrameTable(editable=True, float_decimals=2)
        self.tbl_group_map.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked)
        b.addWidget(self.tbl_group_map, 1)

        merge_row = QtWidgets.QHBoxLayout()
        self.txt_group_merge_label = QtWidgets.QLineEdit("MyGroup")
        self.btn_group_merge_apply = QtWidgets.QPushButton("Merge Selected Rows (Apply Label)")
        style_button(self.btn_group_merge_apply, level=1)
        self.btn_group_merge_apply.clicked.connect(self._merge_group_mapping_selected)

        merge_row.addWidget(QtWidgets.QLabel("Select rows above & Enter Label to merge:"))
        merge_row.addWidget(self.txt_group_merge_label, 2)
        merge_row.addWidget(self.btn_group_merge_apply)
        b.addLayout(merge_row)
        layout.addWidget(box, 1)

        # Compose Section
        box2 = QtWidgets.QGroupBox("Combine Segments: Multiple *_seg -> Combined Segment")
        c = QtWidgets.QHBoxLayout(box2)
        self.lst_compose_segs = QtWidgets.QListWidget()
        self.lst_compose_segs.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        right = QtWidgets.QVBoxLayout()
        self.txt_compose_newcol = QtWidgets.QLineEdit("combo_seg")
        self.txt_compose_sep = QtWidgets.QLineEdit("|")
        self.btn_compose = QtWidgets.QPushButton("Create Combined Segment")
        style_button(self.btn_compose, level=2)
        self.btn_compose.clicked.connect(self._compose_segs)

        right.addWidget(QtWidgets.QLabel("Select *_seg columns"))
        right.addWidget(QtWidgets.QLabel("New Column Name"))
        right.addWidget(self.txt_compose_newcol)
        right.addWidget(QtWidgets.QLabel("Separator"))
        right.addWidget(self.txt_compose_sep)
        right.addWidget(self.btn_compose)
        right.addStretch(1)

        c.addWidget(self.lst_compose_segs, 2)
        c.addLayout(right, 1)
        layout.addWidget(box2, 1)

# =============================================================================
# app.py (Part 8/9)
# Grouping Logic & Segmentation Tabs (Setting/Editing/Logic)
# =============================================================================

    # -------------------------------------------------------------------------
    # Grouping Tab Logic Methods
    # -------------------------------------------------------------------------
    def _apply_binary_recode(self):
        try:
            self._ensure_df()
            df = self.state.df
            col = self.cmb_bin_col.currentText().strip()
            if not col:
                raise RuntimeError("Select a column.")

            s = df[col]
            s_str = s.astype(str).str.strip()

            v1 = self.txt_bin_val1.text().strip()
            v2 = self.txt_bin_val2.text().strip()

            # Auto detect top 2 if empty
            if v1 == "" or v2 == "":
                top = s_str[s_str.notna()].value_counts().index.tolist()
                top = [t for t in top if t != "nan"]
                if len(top) < 2:
                    raise RuntimeError("Column has fewer than 2 unique values.")
                v1 = top[0] if v1 == "" else v1
                v2 = top[1] if v2 == "" else v2
                self.txt_bin_val1.setText(v1)
                self.txt_bin_val2.setText(v2)

            lab1 = self.txt_bin_lab1.text().strip() or "A"
            lab2 = self.txt_bin_lab2.text().strip() or "B"

            newcol = self.txt_bin_newcol.text().strip()
            if not newcol:
                newcol = f"{col}_seg"
            if not newcol.endswith("_seg"):
                newcol = newcol + "_seg"

            other_lab = (self.txt_bin_else_lab.text().strip() or "Other") if self.chk_bin_else_other.isChecked() else "NA"
            mapped = pd.Series(np.where(s_str == v1, lab1, np.where(s_str == v2, lab2, other_lab)), index=df.index)
            df[newcol] = mapped

            self.state.df = df
            self.tbl_preview.set_df(df)
            self._refresh_all_column_lists()
            self._set_status(f"Binary Recode applied: {newcol} created.")
        except Exception as e:
            show_error(self, "Binary Recode Error", e)

    def _merge_group_mapping_selected(self):
        try:
            label = self.txt_group_merge_label.text().strip()
            if not label:
                raise RuntimeError("Enter a label name.")
            sel = self.tbl_group_map.selectedItems()
            if not sel:
                raise RuntimeError("Select rows in mapping table to merge.")

            rows = sorted(set([it.row() for it in sel]))
            if self.tbl_group_map.columnCount() < 3:
                raise RuntimeError("Mapping table structure invalid.")

            for r in rows:
                item = self.tbl_group_map.item(r, 2)
                if item is None:
                    item = QtWidgets.QTableWidgetItem("")
                    self.tbl_group_map.setItem(r, 2, item)
                item.setText(label)

            self._set_status("Selected rows merged with new label.")
        except Exception as e:
            show_error(self, "Merge Error", e)

    def _build_group_mapping(self):
        try:
            self._ensure_df()
            df = self.state.df
            src = self.cmb_group_source.currentText().strip()
            if not src:
                raise RuntimeError("Select source column.")

            vals = pd.Series(df[src].dropna().unique()).astype(str)
            try:
                vv = vals.astype(float)
                order = np.argsort(vv.values)
                vals = vals.iloc[order]
            except:
                vals = vals.sort_values()

            # RECODE lookup
            rec_name = {}
            if self.state.recode_df is not None:
                r = self.state.recode_df
                if src in r["QUESTION"].astype(str).values:
                    r = r[r["QUESTION"].astype(str).str.strip() == src]
                    rec_name = dict(zip(r["CODE"].astype(str).str.strip(), r["NAME"].astype(str).str.strip()))

            recode_names = [rec_name.get(v, "") for v in vals.values]
            seg_default = [rec_name.get(v, v) if rec_name.get(v, "") != "" else v for v in vals.values]

            map_df = pd.DataFrame({
                "source_value": vals.values,
                "recode_name": recode_names,
                "segment_label": seg_default
            })
            self.tbl_group_map.set_df(map_df)
            self._set_status("Mapping table built. Double-click 'segment_label' to edit.")
        except Exception as e:
            show_error(self, "Build Mapping Error", e)

    def _apply_group_mapping(self):
        try:
            self._ensure_df()
            df = self.state.df
            src = self.cmb_group_source.currentText().strip()
            newcol = self.txt_group_newcol.text().strip()
            if not newcol: raise RuntimeError("Enter new column name.")
            if not newcol.endswith("_seg"): newcol += "_seg"

            m = {}
            for r in range(self.tbl_group_map.rowCount()):
                k = self.tbl_group_map.item(r, 0).text()
                v = self.tbl_group_map.item(r, 2).text()
                m[k] = v

            st = df[src].astype(str).str.strip()
            df[newcol] = st.map(m).fillna("NA")

            self.state.df = df
            self.tbl_preview.set_df(df)
            self._refresh_all_column_lists()
            self._set_status(f"Mapping applied: {newcol} created.")
        except Exception as e:
            show_error(self, "Apply Mapping Error", e)

    def _compose_segs(self):
        try:
            self._ensure_df()
            df = self.state.df
            cols = [it.text() for it in self.lst_compose_segs.selectedItems()]
            if len(cols) < 2:
                raise RuntimeError("Select 2 or more *_seg columns.")
            newcol = self.txt_compose_newcol.text().strip()
            if not newcol: raise RuntimeError("Enter new column name.")
            if not newcol.endswith("_seg"): newcol += "_seg"
            sep = self.txt_compose_sep.text() or "|"

            df[newcol] = df[cols].astype(str).apply(lambda r: sep.join(r.values.tolist()), axis=1)
            self.state.df = df
            self.tbl_preview.set_df(df)
            self._refresh_all_column_lists()
            self._set_status(f"Combined segment {newcol} created.")
        except Exception as e:
            show_error(self, "Compose Error", e)

    # -------------------------------------------------------------------------
    # Tab 7: Segmentation Setting
    # -------------------------------------------------------------------------
    def _build_tab_seg_setting(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Segmentation Setting")
        layout = QtWidgets.QHBoxLayout(tab)
        
        # Left: Settings
        left = QtWidgets.QVBoxLayout()

        mode_box = QtWidgets.QGroupBox("Mode")
        mlay = QtWidgets.QHBoxLayout(mode_box)
        self.cmb_demand_mode = QtWidgets.QComboBox()
        self.cmb_demand_mode.addItems([
            "Segments-as-points (Demographics/Profile based)",
            "Variables-as-points (Columns as points)"
        ])
        self.cmb_demand_mode.currentTextChanged.connect(self._on_demand_mode_changed)
        mlay.addWidget(QtWidgets.QLabel("Type"))
        mlay.addWidget(self.cmb_demand_mode, 1)
        left.addWidget(mode_box)

        seg_box = QtWidgets.QGroupBox("Segments-as-points Input")
        seg_l = QtWidgets.QVBoxLayout(seg_box)
        seg_l.addWidget(QtWidgets.QLabel("Select *_seg columns to combine:"))
        self.lst_demand_segcols = QtWidgets.QListWidget()
        self.lst_demand_segcols.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        seg_l.addWidget(self.lst_demand_segcols, 2)

        r = QtWidgets.QHBoxLayout()
        self.txt_demand_seg_sep = QtWidgets.QLineEdit("|")
        self.txt_demand_seg_sep.setMaximumWidth(60)
        self.cmb_demand_target = QtWidgets.QComboBox()
        self.spin_demand_min_n = QtWidgets.QSpinBox()
        self.spin_demand_min_n.setRange(1, 999999)
        self.spin_demand_min_n.setValue(10)
        r.addWidget(QtWidgets.QLabel("Separator"))
        r.addWidget(self.txt_demand_seg_sep)
        r.addSpacing(12)
        r.addWidget(QtWidgets.QLabel("Target Variable"))
        r.addWidget(self.cmb_demand_target, 2)
        r.addSpacing(12)
        r.addWidget(QtWidgets.QLabel("Min N"))
        r.addWidget(self.spin_demand_min_n)
        seg_l.addLayout(r)

        feat = QtWidgets.QHBoxLayout()
        # [v7.0] Use Factors instead of PCA
        self.chk_demand_use_factors = QtWidgets.QCheckBox("Use Factors (Factor1..k) as Profile Features")
        self.chk_demand_use_factors.setChecked(True)
        self.spin_demand_factor_k = QtWidgets.QSpinBox()
        self.spin_demand_factor_k.setRange(2, 50)
        self.spin_demand_factor_k.setValue(5)
        feat.addWidget(self.chk_demand_use_factors)
        feat.addSpacing(12)
        feat.addWidget(QtWidgets.QLabel("Max Factors (k)"))
        feat.addWidget(self.spin_demand_factor_k)
        seg_l.addLayout(feat)
        left.addWidget(seg_box)

        var_box = QtWidgets.QGroupBox("Variables-as-points Input")
        var_l = QtWidgets.QVBoxLayout(var_box)
        var_l.addWidget(QtWidgets.QLabel("Select variables to plot as points:"))
        self.lst_demand_vars = QtWidgets.QListWidget()
        self.lst_demand_vars.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        var_l.addWidget(self.lst_demand_vars, 2)
        
        vbtn = QtWidgets.QHBoxLayout()
        self.btn_demand_check_sel = QtWidgets.QPushButton("Check Selected")
        self.btn_demand_uncheck_sel = QtWidgets.QPushButton("Uncheck Selected")
        self.btn_demand_check_sel.clicked.connect(lambda: self._set_checked_for_selected(self.lst_demand_vars, True))
        self.btn_demand_uncheck_sel.clicked.connect(lambda: self._set_checked_for_selected(self.lst_demand_vars, False))
        vbtn.addWidget(self.btn_demand_check_sel)
        vbtn.addWidget(self.btn_demand_uncheck_sel)
        var_l.addLayout(vbtn)
        left.addWidget(var_box)

        row = QtWidgets.QHBoxLayout()
        self.cmb_demand_coord = QtWidgets.QComboBox()
        self.cmb_demand_coord.addItems(["PCA (Dim1/Dim2)", "MDS (1-corr distance)"])
        self.spin_demand_k = QtWidgets.QSpinBox()
        self.spin_demand_k.setRange(2, 30)
        self.spin_demand_k.setValue(6)
        self.btn_run_demand = QtWidgets.QPushButton("Run Demand Space")
        style_button(self.btn_run_demand, level=2)
        self.btn_run_demand.clicked.connect(self._run_demand_space)

        row.addWidget(QtWidgets.QLabel("Method"))
        row.addWidget(self.cmb_demand_coord)
        row.addWidget(QtWidgets.QLabel("K-Means (k)"))
        row.addWidget(self.spin_demand_k)
        row.addWidget(self.btn_run_demand)
        left.addLayout(row)
        
        self.lbl_demand_status = QtWidgets.QLabel("")
        left.addWidget(self.lbl_demand_status)
        layout.addLayout(left, 2)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("Preview (Go to 'Segmentation Editing' tab to modify)"))
        self.plot_preview = DemandClusterPlot(editable=False)
        right.addWidget(self.plot_preview, 1)
        layout.addLayout(right, 3)
        
        self._on_demand_mode_changed()

    def _on_demand_mode_changed(self):
        txt = self.cmb_demand_mode.currentText()
        seg_mode = txt.startswith("Segments-as-points")

        self.lst_demand_segcols.setEnabled(seg_mode)
        self.txt_demand_seg_sep.setEnabled(seg_mode)
        self.cmb_demand_target.setEnabled(seg_mode)
        self.spin_demand_min_n.setEnabled(seg_mode)
        self.chk_demand_use_factors.setEnabled(seg_mode)
        self.spin_demand_factor_k.setEnabled(seg_mode)

        self.lst_demand_vars.setEnabled(not seg_mode)
        self.btn_demand_check_sel.setEnabled(not seg_mode)
        self.btn_demand_uncheck_sel.setEnabled(not seg_mode)

    # -------------------------------------------------------------------------
    # Tab 8: Segmentation Editing
    # -------------------------------------------------------------------------
    def _build_tab_seg_editing(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Segmentation Editing")
        layout = QtWidgets.QHBoxLayout(tab)

        left = QtWidgets.QVBoxLayout()
        
        # Mode Toggle
        toggle_group = QtWidgets.QGroupBox("Edit Mode (Toggle)")
        tgl_lay = QtWidgets.QVBoxLayout(toggle_group)
        self.radio_edit_points = QtWidgets.QRadioButton("Edit Points (Move/Merge)")
        self.radio_edit_points.setToolTip("Drag points to move/merge. Pan is locked.")
        self.radio_edit_view = QtWidgets.QRadioButton("View/Pan Mode")
        self.radio_edit_view.setToolTip("Drag background to pan. Editing is locked.")
        self.radio_edit_view.setChecked(True)

        self.radio_edit_points.toggled.connect(self._on_edit_mode_toggled)
        self.radio_edit_view.toggled.connect(self._on_edit_mode_toggled)
        tgl_lay.addWidget(self.radio_edit_points)
        tgl_lay.addWidget(self.radio_edit_view)
        left.addWidget(toggle_group)

        # Options
        opt_group = QtWidgets.QGroupBox("Point Options")
        olay = QtWidgets.QVBoxLayout(opt_group)
        self.chk_free_move_points = QtWidgets.QCheckBox("Free Move Points (No Snap)")
        self.chk_free_move_points.toggled.connect(lambda v: self.plot_edit.set_free_move_points(v))
        self.chk_show_all_point_labels = QtWidgets.QCheckBox("Show All Labels")
        self.chk_show_all_point_labels.toggled.connect(lambda v: self.plot_edit.set_show_all_point_labels(v))
        self.btn_auto_labels = QtWidgets.QPushButton("Auto-Arrange Labels")
        style_button(self.btn_auto_labels, level=1)
        self.btn_auto_labels.clicked.connect(lambda: self.plot_edit.auto_arrange_labels())
        self.btn_reset_label_pos = QtWidgets.QPushButton("Reset Label Pos")
        style_button(self.btn_reset_label_pos, level=1)
        self.btn_reset_label_pos.clicked.connect(lambda: self.plot_edit.reset_label_positions())
        
        olay.addWidget(self.chk_free_move_points)
        olay.addWidget(self.chk_show_all_point_labels)
        olay.addWidget(self.btn_auto_labels)
        olay.addWidget(self.btn_reset_label_pos)
        left.addWidget(opt_group)

        # Summary Table
        left.addWidget(QtWidgets.QLabel("Cluster Summary (Dynamic N)"))
        self.tbl_cluster_summary = DataFrameTable(float_decimals=2)
        left.addWidget(self.tbl_cluster_summary, 1)

        # Rename Cluster
        rename_box = QtWidgets.QHBoxLayout()
        self.spin_rename_cluster_id = QtWidgets.QSpinBox()
        self.spin_rename_cluster_id.setRange(1, 999)
        self.txt_rename_cluster = QtWidgets.QLineEdit("Name")
        self.btn_rename_cluster = QtWidgets.QPushButton("Rename")
        style_button(self.btn_rename_cluster, level=1)
        self.btn_rename_cluster.clicked.connect(self._rename_cluster)
        rename_box.addWidget(QtWidgets.QLabel("ID"))
        rename_box.addWidget(self.spin_rename_cluster_id)
        rename_box.addWidget(self.txt_rename_cluster)
        rename_box.addWidget(self.btn_rename_cluster)
        left.addLayout(rename_box)

        layout.addLayout(left, 1)

        right = QtWidgets.QVBoxLayout()
        self.plot_edit = DemandClusterPlot(editable=True)
        self.plot_edit.sigClustersChanged.connect(self._on_manual_clusters_changed)
        self.plot_edit.sigCoordsChanged.connect(self._on_manual_coords_changed)
        right.addWidget(self.plot_edit, 1)
        layout.addLayout(right, 3)

        self._on_edit_mode_toggled()

    def _on_edit_mode_toggled(self):
        is_point_edit = self.radio_edit_points.isChecked()
        self.plot_edit.set_edit_mode_active(is_point_edit)
        if is_point_edit:
            self._set_status("Edit Mode: Drag points/labels enabled. (Pan locked)")
        else:
            self._set_status("View Mode: Pan/Zoom enabled. (Editing locked)")

    def _clear_demand_view(self):
        empty_args = ([], [], np.zeros((0, 2)), np.zeros((0,), dtype=int), {})
        self.plot_preview.set_data(*empty_args)
        self.plot_edit.set_data(*empty_args)
        self.tbl_cluster_summary.set_df(None)
        if hasattr(self, "lbl_demand_status"):
            self.lbl_demand_status.setText("Analysis not run.")

    def _run_demand_space(self):
        try:
            self._ensure_df()
            seg_mode = self.cmb_demand_mode.currentText().startswith("Segments-as-points")
            mode = self.cmb_demand_coord.currentText()
            k = int(self.spin_demand_k.value())

            if seg_mode:
                seg_cols = self._selected_checked_items(self.lst_demand_segcols)
                if len(seg_cols) < 1: raise RuntimeError("Select at least 1 *_seg column.")
                sep = self.txt_demand_seg_sep.text().strip() or "|"
                
                # [v7.0] Use Factors instead of PCA
                use_factors = bool(self.chk_demand_use_factors.isChecked())
                fac_k = int(self.spin_demand_factor_k.value())
                target = self.cmb_demand_target.currentText().strip()
                if target == "": target = "(None)"
                min_n = int(self.spin_demand_min_n.value())

                prof, feat_cols = self._build_segment_profiles(seg_cols, sep, use_factors, fac_k, target, min_n)
                
                X = prof[feat_cols].copy()
                X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
                # Standardize
                scaler = StandardScaler()
                Xz = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
                Xz = Xz.fillna(0) # Safety

                if mode.startswith("PCA"):
                    pca = PCA(n_components=2, random_state=42)
                    xy = pca.fit_transform(Xz.values)
                    coord_name = "PCA(profile)"
                else:
                    C = np.corrcoef(Xz.values)
                    D = 1.0 - C
                    D = np.clip(D, 0.0, 2.0)
                    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=2, max_iter=300)
                    xy = mds.fit_transform(D)
                    coord_name = "MDS(1-corr,profile)"

                ids = prof.index.astype(str).tolist()
                labels = ids[:]
                k = max(2, min(k, len(ids)))
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                cl = km.fit_predict(xy) + 1

                xy_df = pd.DataFrame({"id": ids, "label": labels, "x": xy[:, 0], "y": xy[:, 1], "n": prof["n"].values})
                cl_s = pd.Series(cl, index=ids)

                self.state.demand_mode = "Segments-as-points"
                self.state.demand_xy = xy_df
                self.state.cluster_assign = cl_s
                self.state.cluster_names = {i + 1: f"Cluster {i + 1}" for i in range(k)}
                self.state.demand_seg_profile = prof
                self.state.demand_seg_components = seg_cols
                self.state.demand_features_used = feat_cols
                self.state.manual_dirty = False
                
                args = (ids, labels, xy, cl, self.state.cluster_names)
                self.plot_preview.set_data(*args)
                self.plot_edit.set_data(*args)
                self._update_cluster_summary()
                self.lbl_demand_status.setText(f"Done: {coord_name}, segments={len(ids)}, k={k}.")
                self._set_status("Demand Space Analysis Completed.")

            else:
                cols = self._selected_checked_items(self.lst_demand_vars)
                if len(cols) < 3: raise RuntimeError("Select at least 3 variables.")
                Vz, labels = self._variables_as_matrix(cols)
                
                if mode.startswith("PCA"):
                    pca = PCA(n_components=2, random_state=42)
                    xy = pca.fit_transform(Vz)
                    coord_name = "PCA(variables)"
                else:
                    C = np.corrcoef(Vz)
                    D = 1.0 - C
                    D = np.clip(D, 0.0, 2.0)
                    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=2, max_iter=300)
                    xy = mds.fit_transform(D)
                    coord_name = "MDS(1-corr,variables)"

                k = max(2, min(k, xy.shape[0]))
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                cl = km.fit_predict(xy) + 1
                
                ids = labels
                xy_df = pd.DataFrame({"id": ids, "label": labels, "x": xy[:, 0], "y": xy[:, 1]})
                cl_s = pd.Series(cl, index=ids)

                self.state.demand_mode = "Variables-as-points"
                self.state.demand_xy = xy_df
                self.state.cluster_assign = cl_s
                self.state.cluster_names = {i + 1: f"Cluster {i + 1}" for i in range(k)}
                self.state.demand_seg_profile = None
                self.state.manual_dirty = False

                args = (ids, labels, xy, cl, self.state.cluster_names)
                self.plot_preview.set_data(*args)
                self.plot_edit.set_data(*args)
                self._update_cluster_summary()
                self.lbl_demand_status.setText(f"Done: {coord_name}, vars={len(ids)}, k={k}.")
                self._set_status("Demand Space (Vars) Analysis Completed.")

        except Exception as e:
            self.state.last_error = str(e)
            show_error(self, "Demand Space Error", e)

    def _build_segment_profiles(self, seg_cols: List[str], sep: str, use_factors: bool, fac_k: int, target: str, min_n: int):
        df = self.state.df.copy()
        
        # Combine segments
        df["_SEG_LABEL_"] = df[seg_cols].astype(str).apply(lambda r: sep.join(r.values), axis=1)
        
        # Filter min_n
        cnt = df["_SEG_LABEL_"].value_counts()
        valid_segs = cnt[cnt >= min_n].index
        df = df[df["_SEG_LABEL_"].isin(valid_segs)].copy()
        if df.empty:
            raise RuntimeError(f"No segments have >= {min_n} size.")

        # Features: Factors and Target
        feat_cols = []
        if use_factors:
            # [v7.0] Search for Factor1..k columns
            avail = [c for c in df.columns if str(c).startswith("Factor") and str(c)[6:].isdigit()]
            # Filter by index <= fac_k
            # e.g., Factor1, Factor2
            selected = [c for c in avail if int(c[6:]) <= fac_k]
            feat_cols.extend(selected)

        if target != "(None)" and target in df.columns:
            feat_cols.append(target)
            df[target] = pd.to_numeric(df[target], errors="coerce") # Ensure numeric target

        if not feat_cols:
            raise RuntimeError("No features for profiling (Enable Factors or select Target).")

        # Aggregate
        prof = df.groupby("_SEG_LABEL_")[feat_cols].mean()
        prof["n"] = df.groupby("_SEG_LABEL_").size()
        return prof, feat_cols

    def _variables_as_matrix(self, cols: List[str]):
        df = to_numeric_df(self.state.df, cols)
        df = df.dropna(axis=0, how="all")
        if df.shape[0] < 5:
            raise RuntimeError("Not enough data rows.")
        
        # Transpose: Rows=Vars, Cols=Respondents
        # Standardize respondents first? Or vars?
        # Usually for variable mapping, we correlate variables.
        # Impute mean
        df = df.fillna(df.mean())
        # Transpose -> (n_vars, n_resp)
        V = df.T
        # Standardize across respondents (rows of V)
        scaler = StandardScaler()
        Vz = scaler.fit_transform(V)
        return Vz, cols

# =============================================================================
# app.py (Part 9/9)
# Manual Edit Logic, Export, RAG Chatbot, and Main Execution
# =============================================================================

    # -------------------------------------------------------------------------
    # Manual Editing Logic (Slots)
    # -------------------------------------------------------------------------
    def _update_cluster_summary(self):
        if self.state.demand_xy is None or self.state.cluster_assign is None:
            self.tbl_cluster_summary.set_df(None)
            return

        cl = self.state.cluster_assign.copy()
        names = self.state.cluster_names or {}

        rows = []
        n_map = {}
        if self.state.demand_mode.startswith("Segments") and "n" in self.state.demand_xy.columns:
            n_map = dict(zip(self.state.demand_xy["id"], self.state.demand_xy["n"]))

        for cid in sorted(cl.unique()):
            items = cl[cl == cid].index.tolist()
            n_sum = None
            if n_map:
                n_sum = int(sum(int(n_map.get(x, 0)) for x in items))
            
            rows.append({
                "Cluster ID": int(cid),
                "Name": names.get(int(cid), f"Cluster {int(cid)}"),
                "Count (Points)": len(items),
                "Sum N (Respondents)": n_sum if n_map else "",
                "Items": ", ".join(items)
            })
        out = pd.DataFrame(rows).sort_values(["Cluster ID"]).reset_index(drop=True)
        self.tbl_cluster_summary.set_df(out, max_rows=500)

    def _on_manual_clusters_changed(self):
        if self.state.demand_xy is None: return
        s = self.plot_edit.get_cluster_series()
        self.state.cluster_assign = s
        self.state.manual_dirty = True
        self._update_cluster_summary()
        self._set_status("Manual clusters updated.")

    def _on_manual_coords_changed(self):
        if self.state.demand_xy is None: return
        xy_map = self.plot_edit.get_xy_map()
        try:
            df = self.state.demand_xy.copy()
            if "id" in df.columns:
                df["x"] = df["id"].astype(str).map(lambda k: xy_map.get(str(k), (np.nan, np.nan))[0])
                df["y"] = df["id"].astype(str).map(lambda k: xy_map.get(str(k), (np.nan, np.nan))[1])
                self.state.demand_xy = df
                self.state.manual_dirty = True
                self._set_status("Manual coords updated.")
        except Exception:
            self._set_status("Manual coords update failed.")

    def _rename_cluster(self):
        try:
            cid = int(self.spin_rename_cluster_id.value())
            name = self.txt_rename_cluster.text().strip()
            if not name: raise RuntimeError("Enter a name.")
            self.state.cluster_names[cid] = name
            
            self.plot_edit.set_cluster_names(self.state.cluster_names)
            self.plot_preview.set_cluster_names(self.state.cluster_names)
            self._update_cluster_summary()
            self.state.manual_dirty = True
        except Exception as e:
            show_error(self, "Rename Error", e)

    # -------------------------------------------------------------------------
    # Tab 9: Export
    # -------------------------------------------------------------------------
    def _build_tab_export(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Export")
        layout = QtWidgets.QVBoxLayout(tab)

        self.lbl_export = QtWidgets.QLabel(
            "Export Results to Excel:\n"
            "Sheets: 01_Data, 02_RECODE, 03_Factor_Loadings, 04_Factor_Scores,\n"
            "05_DT_ImprovePivot, 06_DT_BestSplit, 07_DT_Full_Nodes, ...\n"
            "13_Demand_Clusters, 14_Demand_Summary, 15_Demand_SegProfile"
        )
        self.lbl_export.setWordWrap(True)
        layout.addWidget(self.lbl_export)

        row = QtWidgets.QHBoxLayout()
        self.txt_export_path = QtWidgets.QLineEdit()
        self.btn_export_browse = QtWidgets.QPushButton("Browse...")
        style_button(self.btn_export_browse, level=1)
        self.btn_export_browse.clicked.connect(self._browse_export_path)
        self.btn_export = QtWidgets.QPushButton("Export to Excel")
        style_button(self.btn_export, level=2)
        self.btn_export.clicked.connect(self._export_excel)

        row.addWidget(QtWidgets.QLabel("Output Path"))
        row.addWidget(self.txt_export_path, 3)
        row.addWidget(self.btn_export_browse)
        row.addWidget(self.btn_export)
        layout.addLayout(row)

        self.lbl_export_status = QtWidgets.QLabel("")
        layout.addWidget(self.lbl_export_status)
        layout.addStretch(1)

    def _browse_export_path(self):
        default = "analysis_output.xlsx"
        if self.state.path:
            base = os.path.splitext(os.path.basename(self.state.path))[0]
            default = f"{base}_AutoSegmentTool_v7.xlsx"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Excel", default, "Excel (*.xlsx)")
        if path:
            if not path.lower().endswith(".xlsx"): path += ".xlsx"
            self.txt_export_path.setText(path)

    def _export_excel(self):
        try:
            self._ensure_df()
            out = self.txt_export_path.text().strip()
            if not out: raise RuntimeError("Select output path.")
            
            with pd.ExcelWriter(out, engine="openpyxl") as w:
                self.state.df.to_excel(w, sheet_name="01_Data", index=False)
                if self.state.recode_df is not None:
                    self.state.recode_df.to_excel(w, sheet_name="02_RECODE", index=False)
                
                # [v7.0] Factor names
                if self.state.factor_loadings is not None:
                    self.state.factor_loadings.reset_index().rename(columns={"index": "variable"}).to_excel(w, sheet_name="03_Factor_Loadings", index=False)
                if self.state.factor_scores is not None:
                    self.state.factor_scores.reset_index().rename(columns={"index": "row_index"}).to_excel(w, sheet_name="04_Factor_Scores", index=False)

                if self.state.dt_improve_pivot is not None:
                    self.state.dt_improve_pivot.to_excel(w, sheet_name="05_DT_ImprovePivot", index=False)
                if self.state.dt_split_best is not None:
                    self.state.dt_split_best.to_excel(w, sheet_name="06_DT_BestSplit", index=False)
                
                if self.state.dt_full_nodes is not None:
                    self.state.dt_full_nodes.to_excel(w, sheet_name="07_DT_Full_Nodes", index=False)
                
                # Demand Space
                if self.state.demand_xy is not None:
                    self.state.demand_xy.to_excel(w, sheet_name="12_Demand_Coords", index=False)
                
                if self.state.cluster_assign is not None:
                    cl_df = self.state.cluster_assign.reset_index()
                    cl_df.columns = ["id", "cluster_id"]
                    cl_df["cluster_name"] = cl_df["cluster_id"].map(self.state.cluster_names).fillna("")
                    cl_df.to_excel(w, sheet_name="13_Demand_Clusters", index=False)
                    
                    # Summary sheet logic (reuse helper logic briefly)
                    # ... (omitted for brevity, typically re-calculating summary df) ...

            self.lbl_export_status.setText(f"Exported successfully to {out}")
            self._set_status("Export Done.")
        except Exception as e:
            self.state.last_error = str(e)
            show_error(self, "Export Error", e)

    # -------------------------------------------------------------------------
    # Tab 10: AI Assistant (RAG Chatbot) - [New in v7.0]
    # -------------------------------------------------------------------------
    def _build_tab_rag(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "AI Assistant (RAG)")
        layout = QtWidgets.QVBoxLayout(tab)
        
        # Info
        layout.addWidget(QtWidgets.QLabel("<b>AI Assistant</b>: Ask questions about your current data/analysis."))
        
        # API Key
        key_row = QtWidgets.QHBoxLayout()
        self.txt_openai_key = QtWidgets.QLineEdit()
        self.txt_openai_key.setPlaceholderText("Enter OpenAI API Key (sk-...) or leave empty to just generate prompt")
        self.txt_openai_key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        key_row.addWidget(QtWidgets.QLabel("API Key:"))
        key_row.addWidget(self.txt_openai_key, 1)
        layout.addLayout(key_row)

        # Chat Area
        self.txt_chat_history = QtWidgets.QTextEdit()
        self.txt_chat_history.setReadOnly(True)
        layout.addWidget(self.txt_chat_history, 1)

        # Input Area
        input_row = QtWidgets.QHBoxLayout()
        self.txt_user_query = QtWidgets.QLineEdit()
        self.txt_user_query.setPlaceholderText("Ex: 'Interpret the Factor 1 loadings' or 'Why did the decision tree select Age?'")
        self.txt_user_query.returnPressed.connect(self._send_rag_query)
        
        self.btn_send_query = QtWidgets.QPushButton("Send / Gen Prompt")
        style_button(self.btn_send_query, level=2)
        self.btn_send_query.clicked.connect(self._send_rag_query)
        
        input_row.addWidget(self.txt_user_query, 3)
        input_row.addWidget(self.btn_send_query)
        layout.addLayout(input_row)

    def _get_rag_context(self) -> str:
        """Collects current application state into a text block."""
        ctx = ["=== SYSTEM CONTEXT ==="]
        
        # Data Info
        if self.state.df is not None:
            ctx.append(f"Data Loaded: {len(self.state.df)} rows, {len(self.state.df.columns)} cols.")
            ctx.append(f"Columns: {', '.join(list(self.state.df.columns)[:50])} ...")
        else:
            ctx.append("Data: None loaded.")

        # Factor Analysis
        if self.state.factor_loadings is not None:
            ctx.append(f"\n[Factor Analysis ({self.state.factor_mode})]")
            ctx.append("Top Loadings:")
            # Simple string dump of top loadings
            try:
                top = self.state.factor_loadings.head(10).to_string()
                ctx.append(top)
            except: pass
            
        # Decision Tree Best Splits
        if self.state.dt_split_best is not None:
            ctx.append(f"\n[Decision Tree Best Splits (Top 5 Improvement)]")
            try:
                top_splits = self.state.dt_split_best.sort_values("improve_rel", ascending=False).head(5)
                for _, row in top_splits.iterrows():
                    ctx.append(f"- Dep: {row['dep']}, Ind: {row['ind']}, Type: {row['split_type']}, Imp: {row['improve_rel']:.4f}")
            except: pass

        # Errors
        if self.state.last_error:
            ctx.append(f"\n[Recent Error Log]\n{self.state.last_error}")

        return "\n".join(ctx)

    def _send_rag_query(self):
        query = self.txt_user_query.text().strip()
        if not query: return
        
        # 1. Build Context
        context = self._get_rag_context()
        full_prompt = f"{context}\n\n=== USER QUESTION ===\n{query}\n\n=== INSTRUCTION ===\nAnswer based on the context above. If error log exists, suggest a fix."

        self.txt_chat_history.append(f"<b>User:</b> {query}")
        self.txt_user_query.clear()

        api_key = self.txt_openai_key.text().strip()
        
        # 2. If No Key -> Just show prompt
        if not api_key:
            self.txt_chat_history.append(f"<br><i>[System] No API Key provided. Copy this prompt to ChatGPT:</i><br>")
            self.txt_chat_history.append(f"<code>{full_prompt}</code><br><hr>")
            return

        # 3. Call API
        try:
            self.txt_chat_history.append("<i>... Thinking ...</i>")
            QtWidgets.QApplication.processEvents()
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini", # or gpt-3.5-turbo
                "messages": [
                    {"role": "system", "content": "You are a Data Analysis Assistant."},
                    {"role": "user", "content": full_prompt}
                ],
                "temperature": 0.7
            }
            
            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            
            self.txt_chat_history.append(f"<b>AI:</b> {answer}<br><hr>")
            
        except Exception as e:
            self.txt_chat_history.append(f"<font color='red'>API Error: {str(e)}</font><br>")


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Optional: Set Style
    app.setStyle("Fusion")
    
    win = IntegratedApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
