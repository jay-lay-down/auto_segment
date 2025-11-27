# =========================
# app.py (Full Source Code - Verbose Version)
# =========================
# -*- coding: utf-8 -*-
"""
Auto Segment Tool v4.8 (Line Count Restored)

[변경 사항 확인]
1. 탭 구조 변경: Demand Space -> [Segmentation setting] + [Segmentation editing] 분리
2. 탭 순서: Recode mapping ... -> Export
3. Editing 탭: 점 편집 / 화면 이동 토글 적용
4. 코드 스타일: 압축된 코드를 다시 풀어서 원본 가독성 및 라인 수 복구

Requires: PyQt6, pyqtgraph, pandas, numpy, scikit-learn, openpyxl
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from PyQt6 import QtCore, QtGui, QtWidgets

import pyqtgraph as pg
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.cluster import KMeans, AgglomerativeClustering


# -----------------------------
# Helpers
# -----------------------------
def pal_hex() -> List[str]:
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
    # NOTE: 정수인데 유니크가 적으면 범주로 취급
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
    """Monotone chain convex hull for points (N,2)."""
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


def safe_sheetname(name: str) -> str:
    bad = r'[]:*?/\\'
    out = "".join("_" if ch in bad else ch for ch in name)
    return out[:31] if len(out) > 31 else out


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
    """
    버튼 색 단계:
      - level 1: 가장 앞 단계 (연한 파랑)
      - level이 커질수록 점점 진해짐 (흐름 순서 느낌)
    """
    palette = [
        "#e3f2fd",  # 1
        "#bbdefb",  # 2
        "#90caf9",  # 3
        "#64b5f6",  # 4
        "#42a5f5",  # 5
    ]
    idx = max(0, min(level - 1, len(palette) - 1))
    base = palette[idx]
    hover = palette[min(idx + 1, len(palette) - 1)]

    btn.setStyleSheet(
        f"""
        QPushButton {{
            background-color: {base};
            border: 1px solid #90a4ae;
            border-radius: 4px;
            padding: 4px 8px;
        }}
        QPushButton:hover {{
            background-color: {hover};
        }}
        QPushButton:disabled {{
            background-color: #eceff1;
            color: #90a4ae;
        }}
        """
    )


def normalize_recode_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    RECODE 시트 컬럼을 QUESTION/CODE/NAME 으로 정규화.
    - 대소문자/한글/유사명칭 보정
    """
    if df is None:
        return None
    if df.empty:
        return df

    cols = list(df.columns)
    col_lut = {str(c).strip().lower(): c for c in cols}

    def pick(*names):
        for n in names:
            if n in col_lut:
                return col_lut[n]
        return None

    q = pick("question", "문항", "문항명", "q", "item", "variable")
    c = pick("code", "코드", "값", "value", "val")
    n = pick("name", "라벨", "label", "명", "설명", "text")

    # fallback: 앞 3개 컬럼
    if q is None or c is None or n is None:
        if len(cols) >= 3:
            q = cols[0] if q is None else q
            c = cols[1] if c is None else c
            n = cols[2] if n is None else n

    out = df.copy()
    rename_map = {}
    if q is not None:
        rename_map[q] = "QUESTION"
    if c is not None:
        rename_map[c] = "CODE"
    if n is not None:
        rename_map[n] = "NAME"

    out = out.rename(columns=rename_map)

    # 필요한 컬럼만 우선 노출 (있으면)
    for cc in ["QUESTION", "CODE", "NAME"]:
        if cc not in out.columns:
            out[cc] = np.nan

    out["QUESTION"] = out["QUESTION"].astype(str).str.strip()
    out["CODE"] = out["CODE"].astype(str).str.strip()
    out["NAME"] = out["NAME"].astype(str).str.strip()

    # 보기용 컬럼순
    out = out[["QUESTION", "CODE", "NAME"] + [c for c in out.columns if c not in ["QUESTION", "CODE", "NAME"]]]
    return out


# -----------------------------
# App state
# -----------------------------
@dataclass
class AppState:
    df: Optional[pd.DataFrame] = None
    path: Optional[str] = None
    sheet: Optional[str] = None
    recode_df: Optional[pd.DataFrame] = None

    # PCA (respondent-level)
    pca_model: Optional[PCA] = None
    pca_cols: Optional[List[str]] = None
    pca_scores: Optional[pd.DataFrame] = None   # index = row index
    pca_loadings: Optional[pd.DataFrame] = None

    # Decision tree outputs (root)
    dt_improve_pivot: Optional[pd.DataFrame] = None
    dt_split_best: Optional[pd.DataFrame] = None

    # Decision tree full (selected dep×ind)
    dt_full_nodes: Optional[pd.DataFrame] = None
    dt_full_split_groups: Optional[pd.DataFrame] = None
    dt_full_split_branches: Optional[pd.DataFrame] = None
    dt_full_path_info: Optional[pd.DataFrame] = None
    dt_full_condition_freq: Optional[pd.DataFrame] = None
    dt_full_selected: Tuple[Optional[str], Optional[str]] = (None, None)

    # Tree -> HCLUST results
    hclust_assign: Optional[pd.DataFrame] = None  # ind var -> cluster

    # Demand space
    demand_mode: str = "Segments-as-points"
    demand_xy: Optional[pd.DataFrame] = None  # columns: id,label,x,y,(n)
    cluster_assign: Optional[pd.Series] = None  # index=id => cluster int
    cluster_names: Dict[int, str] = field(default_factory=dict)

    # for segment-profile demand space
    demand_seg_profile: Optional[pd.DataFrame] = None  # index=segment label; columns=features + n_count
    demand_seg_components: Optional[List[str]] = None  # *_seg cols used
    demand_features_used: Optional[List[str]] = None   # PCA + target derived columns

    manual_dirty: bool = False


# -----------------------------
# DataFrame viewer table (2-decimal display)
# -----------------------------
class DataFrameTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None, editable: bool = False, float_decimals: int = 2, max_col_width: int = 380):
        super().__init__(parent)
        self._float_decimals = int(float_decimals)
        self._max_col_width = int(max_col_width)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        if not editable:
            self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)

    def set_df(self, df: Optional[pd.DataFrame], max_rows: int = 500):
        if df is None or df.empty:
            self.setRowCount(0)
            self.setColumnCount(0)
            return

        view = df.copy()
        if len(view) > max_rows:
            view = view.iloc[:max_rows].copy()

        self.setColumnCount(view.shape[1])
        self.setRowCount(view.shape[0])
        self.setHorizontalHeaderLabels([str(c) for c in view.columns])

        for r in range(view.shape[0]):
            for c in range(view.shape[1]):
                v = view.iat[r, c]
                if isinstance(v, (float, np.floating)):
                    txt = "" if pd.isna(v) else f"{float(v):.{self._float_decimals}f}"
                else:
                    txt = "" if pd.isna(v) else str(v)
                item = QtWidgets.QTableWidgetItem(txt)
                self.setItem(r, c, item)

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


# -----------------------------
# Decision Tree (R-like) - root best split + full univariate tree
# -----------------------------
def _impurity_reg(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    mu = float(np.mean(y))
    return float(np.mean((y - mu) ** 2))  # MSE


def _impurity_gini(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    vals, cnt = np.unique(y, return_counts=True)
    p = cnt / cnt.sum()
    return float(1.0 - np.sum(p ** 2))


def _root_dev(y: np.ndarray, task: str) -> float:
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
    Root split only (for Improve Pivot):
    - categorical: level vs rest
    - numeric: threshold at midpoints
    """
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

    if is_categorical_series(xv):
        xv = xv.astype("category")
        levels = list(xv.cat.categories)
        if len(levels) < 2 or len(levels) > max_unique_cat:
            return None, []

        for lv in levels:
            left_mask = (xv == lv).values
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

            rows.append({
                "split_type": "categorical(one-vs-rest)",
                "cutpoint": f"{{{lv}}} vs {{rest}}",
                "left_group": str(lv),
                "right_group": "rest",
                "improve_abs": improve_abs,
                "improve_rel": improve_rel,
                "n_left": int(yL.size),
                "n_right": int(yR.size),
            })

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
    best = max(rows, key=lambda r: (r.get("improve_rel", -1e9) if pd.notna(r.get("improve_rel", np.nan)) else -1e9))
    return best, rows


@dataclass
class UniNode:
    node_id: int
    depth: int
    n: int
    condition: str  # path condition text (AND-joined)
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


def _pred_text(y: np.ndarray, task: str) -> str:
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
    Returns best split dict for this subset, plus left_idx, right_idx (original indices).
    """
    ys = y.iloc[idx]
    xs = x.iloc[idx]
    best, _rows = univariate_best_split(ys, xs, task=task, max_unique_cat=max_unique_cat)
    if best is None:
        return None, None, None

    mask = pd.notna(ys) & pd.notna(xs)
    valid_idx = idx[mask.values]
    ys_v = ys[mask].values
    xs_v = xs[mask]

    if len(valid_idx) < 5:
        return None, None, None

    if best["split_type"].startswith("categorical"):
        lv = best["left_group"]
        xs_v = xs_v.astype("category").astype(str)
        left_mask = (xs_v.values == str(lv))
        right_mask = ~left_mask
    else:
        thr = float(best["cutpoint"])
        xnum = pd.to_numeric(xs_v, errors="coerce").values.astype(float)
        ok = np.isfinite(xnum)
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
    dep×ind 단변량 트리를 직접 재귀 구축해서
    Split_Groups / Split_Branches / Path_Info / Condition_Freq 스타일 테이블 생성.
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
        if best is None or lidx is None or ridx is None:
            nid = next_id
            next_id += 1
            yy = y.iloc[idx].dropna().values
            nodes.append(UniNode(
                node_id=nid, depth=depth, n=n,
                condition=cond_text, is_leaf=True,
                pred=_pred_text(yy, task=task)
            ))
            return nid

        imp_rel = float(best.get("improve_rel", 0.0) if pd.notna(best.get("improve_rel", np.nan)) else 0.0)
        if imp_rel < float(min_improve_rel):
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

        nid = next_id
        next_id += 1

        if best["split_type"].startswith("categorical"):
            lv = str(best["left_group"])
            left_cond = f"{ind} == {lv}"
            right_cond = f"{ind} != {lv}"
            cutpoint = best["cutpoint"]
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
            left_condition=left_cond, right_condition=right_cond,
            left_id=left_id, right_id=right_id,
            pred=_pred_text(yy, task=task)
        ))
        return nid

    _ = rec(base_idx, 0, [])

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
            if p == "(root)":
                continue
            parts = [t.strip() for t in p.split("&")]
            parts = [pp for pp in parts if pp and pp != "(root)"]
            conds.extend(parts)
        if conds:
            cond_freq = pd.Series(conds).value_counts().reset_index()
            cond_freq.columns = ["condition", "count"]
            cond_freq.insert(0, "dep", dep)
    return nodes_df, split_groups, branches, path_info, cond_freq

# =========================
# Demand Space interactive plot (manual drag/merge 유지)
# =========================

class DraggableClusterLabel(pg.TextItem):
    """
    Drag cluster label near another cluster centroid => merge.
    Shift + 라벨 드래그: 병합 안 하고 라벨 위치만 이동.
    Edit Mode(점 편집)일 때만 작동하도록 제어.
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
        # 만약 점 편집 모드가 아니면(즉, 화면 이동 모드면) 라벨 드래그 무시 -> 배경 pan 됨
        if not self.plot.is_edit_mode_active():
            ev.ignore()
            return

        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        
        vb = self.plot.getPlotItem().vb
        pos_view = vb.mapSceneToView(ev.scenePos())
        ev.accept()

        # Shift pressed?
        shift = False
        try:
            mods = ev.modifiers()
            shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        except Exception:
            shift = False

        if ev.isFinish():
            if shift:
                # 병합 없이 라벨 위치만 저장
                self.plot.remember_label_position(self.cluster_id, (float(pos_view.x()), float(pos_view.y())))
                return
            self.plot.try_merge_label(self.cluster_id, (float(pos_view.x()), float(pos_view.y())))
        else:
            self.setPos(float(pos_view.x()), float(pos_view.y()))


class ClusterViewBox(pg.ViewBox):
    """
    Mode-aware ViewBox:
    - Edit Mode: Block Pan, Allow Item/Point drag.
    - View Mode: Allow Standard Pan, Block Item/Point drag interactions (handled in Plot logic).
    """
    def __init__(self, plot: "DemandClusterPlot"):
        super().__init__()
        self.plot = plot
        self.setMouseMode(self.RectMode)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            # Edit Mode일 때만 클릭 선택 허용
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
            # Edit Mode: Pan은 막고, Point Drag 로직만 수행
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                pos = self.mapSceneToView(ev.scenePos())
                self.plot.drag_event((float(pos.x()), float(pos.y())), ev)
                return
            # 우클릭 등 다른 버튼은 무시하거나 기본 동작
            ev.ignore()
        else:
            # View/Pan Mode: 기본 ViewBox Pan 동작 허용
            super().mouseDragEvent(ev, axis=axis)


class DemandClusterPlot(pg.PlotWidget):
    """
    - Fixed coordinates (coords stay; drag only reassign cluster)
      + 점 자유이동 ON: 점 drag = 좌표 이동(스토리텔링 배치)
      + 점 자유이동 OFF(기존): 점 drag = 클러스터 재할당(가까운 centroid)
    - Drag label => merge clusters (src -> dst)
      + Shift+라벨 드래그: merge 없이 위치만 이동
    - Palette colors + convex hull polygons + centroid labels
    - 점 라벨 기본: 선택만 표시 (겹침 지옥 방지)
    - 라벨 자동정리: 겹침 완화
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

        self._ids: List[str] = []
        self._labels: List[str] = []
        self._xy: np.ndarray = np.zeros((0, 2), dtype=float)
        self._cluster: np.ndarray = np.zeros((0,), dtype=int)
        self._cluster_names: Dict[int, str] = {}

        self._scatter = pg.ScatterPlotItem(size=11, pxMode=True)
        self.addItem(self._scatter)

        self._point_text_items: List[pg.TextItem] = []
        self._hull_items: Dict[int, QtWidgets.QGraphicsPathItem] = {}
        self._label_items: Dict[int, DraggableClusterLabel] = {}

        self._selected: set[int] = set()
        self._dragging = False
        self._drag_temp_positions: Optional[np.ndarray] = None
        self._drag_anchor_xy: Optional[Tuple[float, float]] = None

        # UI toggles
        self._free_move_points: bool = False
        self._show_all_point_labels: bool = False

        # label position overrides (cluster_id -> (x,y))
        self._label_pos_override: Dict[int, Tuple[float, float]] = {}

    def set_edit_mode_active(self, active: bool):
        """True: 점 편집 모드 (Pan 불가), False: 화면 이동 모드 (Pan 가능)"""
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
        # Simple repulsion to reduce overlap in view coordinates
        if not self._label_items:
            return
        clusters = sorted(self._label_items.keys())
        if len(clusters) <= 1:
            return

        xr, yr = self.getPlotItem().vb.viewRange()
        scale = max((xr[1] - xr[0]), (yr[1] - yr[0]))
        if not np.isfinite(scale) or scale <= 0:
            scale = 10.0
        min_dist = max(0.06 * scale, 0.6)  # tunable

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

        # ✅ Demand Space "스케일"은 자동으로 보기 좋게 맞춤
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

        # ✅ 점 라벨 기본은 "선택만 표시" (겹침 지옥 방지)
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
            # ✅ 점 자유이동 ON/OFF
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
            # merge 안 하면 라벨 위치만 유지
            self.remember_label_position(src_cluster, drop_xy)
            self._draw_hulls_and_labels()
            return

        dst = int(best)
        self._cluster[self._cluster == int(src_cluster)] = dst

        # src label override 제거(사라짐), dst는 유지
        if int(src_cluster) in self._label_pos_override:
            self._label_pos_override.pop(int(src_cluster), None)

        self.redraw_all()
        self.sigClustersChanged.emit()


# =============================
# Main Window UI + Logic
# =============================

class IntegratedApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto Segment Tool v4.8")
        self.resize(1600, 940)

        pg.setConfigOptions(antialias=True)

        self.state = AppState()

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # ✅ 최종 탭 순서 (요청 그대로)
        # 1. Recode mapping
        self._build_tab_recode()
        
        # 2. Group & Compose
        self._build_tab_grouping()
        
        # 3. Data loading
        self._build_tab_data()
        
        # 4. PCA
        self._build_tab_pca()
        
        # 5. Decision Tree
        self._build_tab_tree_hclust()
        
        # 6. Decision Tree Split
        self._build_tab_split_results()
        
        # 7. Segmentation setting (New Split)
        self._build_tab_seg_setting()
        
        # 8. Segmentation editing (New Split)
        self._build_tab_seg_editing()
        
        # 9. Export
        self._build_tab_export()

        self._apply_tab_styles()
        self._set_status("준비 완료.")

    # ---------------- UI helpers
    def _apply_tab_styles(self):
        # 하늘색 테마
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

        # 특수 기능 탭(앞쪽 2개) 텍스트 강조
        try:
            for i in range(self.tabs.count()):
                name = self.tabs.tabText(i)
                if name in ["RECODE 매핑", "GROUP/COMPOSE"]:
                    self.tabs.tabBar().setTabTextColor(i, QtGui.QColor("#1565c0"))
                else:
                    self.tabs.tabBar().setTabTextColor(i, QtGui.QColor("#263238"))
        except Exception:
            pass

    def _set_status(self, text: str):
        self.statusBar().showMessage(text)

    def _ensure_df(self):
        if self.state.df is None:
            raise RuntimeError("먼저 '데이터' 탭에서 엑셀을 불러와.")

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
        df = self.state.df
        if df is None:
            return
        cols = list(df.columns)

        # grouping
        self.cmb_group_source.clear()
        self.cmb_group_source.addItems(cols)

        # binary recode
        self.cmb_bin_col.clear()
        self.cmb_bin_col.addItems(cols)

        # pca
        self.lst_pca_cols.clear()
        for c in cols:
            it = QtWidgets.QListWidgetItem(c)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            it.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.lst_pca_cols.addItem(it)

        # decision tree deps
        self.cmb_dep_extra.clear()
        self.cmb_dep_extra.addItem("(없음)")
        self.cmb_dep_extra.addItems(cols)

        # decision tree exclude list
        self.lst_dt_exclude.clear()
        for c in cols:
            it = QtWidgets.QListWidgetItem(c)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            it.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.lst_dt_exclude.addItem(it)

        # demand space variables-as-points
        self.lst_demand_vars.clear()
        for c in cols:
            it = QtWidgets.QListWidgetItem(c)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            it.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.lst_demand_vars.addItem(it)

        # *_seg columns
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
        self.cmb_demand_target.addItem("(없음)")
        self.cmb_demand_target.addItems(cols)

        # Split Results / full-tree dropdowns는 Decision Tree 실행 이후에 채움
        self.cmb_dt_full_dep.clear()
        self.cmb_dt_full_ind.clear()
        self.cmb_split_dep.clear()
        self.cmb_split_ind.clear()

    # ---------------- Tab: RECODE Mapping
    def _build_tab_recode(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Recode mapping")

        layout = QtWidgets.QVBoxLayout(tab)
        self.lbl_recode = QtWidgets.QLabel("엑셀의 RECODE 시트(QUESTION / CODE / NAME)를 그대로 보여줌.")
        self.lbl_recode.setWordWrap(True)
        layout.addWidget(self.lbl_recode)

        self.tbl_recode = DataFrameTable(float_decimals=2)
        layout.addWidget(self.tbl_recode, 1)

    def _update_recode_tab(self):
        self.tbl_recode.set_df(self.state.recode_df)

    # ---------------- Tab: GROUP/COMPOSE (+ 빠른 이진 RECODE)
    def _build_tab_grouping(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Group & Compose")

        layout = QtWidgets.QVBoxLayout(tab)

        # ✅ 성별 블록 제거 → 빠른 이진 RECODE(2값)
        box_bin = QtWidgets.QGroupBox("빠른 이진 RECODE (2값)  →  *_seg 생성")
        bin_layout = QtWidgets.QHBoxLayout(box_bin)

        self.cmb_bin_col = QtWidgets.QComboBox()
        self.txt_bin_val1 = QtWidgets.QLineEdit("")   # 비우면 자동 top2
        self.txt_bin_lab1 = QtWidgets.QLineEdit("A")
        self.txt_bin_val2 = QtWidgets.QLineEdit("")
        self.txt_bin_lab2 = QtWidgets.QLineEdit("B")
        self.chk_bin_else_other = QtWidgets.QCheckBox("else=기타")
        self.txt_bin_else_lab = QtWidgets.QLineEdit("기타")
        self.txt_bin_else_lab.setMaximumWidth(90)

        self.txt_bin_newcol = QtWidgets.QLineEdit("")  # 비우면 {col}_seg
        self.btn_bin_apply = QtWidgets.QPushButton("이진 RECODE 적용")
        style_button(self.btn_bin_apply, level=2)
        self.btn_bin_apply.clicked.connect(self._apply_binary_recode)

        bin_layout.addWidget(QtWidgets.QLabel("대상 컬럼"))
        bin_layout.addWidget(self.cmb_bin_col, 2)
        bin_layout.addSpacing(10)
        bin_layout.addWidget(QtWidgets.QLabel("값1"))
        bin_layout.addWidget(self.txt_bin_val1)
        bin_layout.addWidget(QtWidgets.QLabel("라벨1"))
        bin_layout.addWidget(self.txt_bin_lab1)
        bin_layout.addSpacing(10)
        bin_layout.addWidget(QtWidgets.QLabel("값2"))
        bin_layout.addWidget(self.txt_bin_val2)
        bin_layout.addWidget(QtWidgets.QLabel("라벨2"))
        bin_layout.addWidget(self.txt_bin_lab2)
        bin_layout.addSpacing(10)
        bin_layout.addWidget(self.chk_bin_else_other)
        bin_layout.addWidget(self.txt_bin_else_lab)
        bin_layout.addSpacing(10)
        bin_layout.addWidget(QtWidgets.QLabel("새 컬럼(비우면 자동)"))
        bin_layout.addWidget(self.txt_bin_newcol, 2)
        bin_layout.addWidget(self.btn_bin_apply)

        layout.addWidget(box_bin)

        # General grouping mapping
        box = QtWidgets.QGroupBox("그룹핑: 원본 값 → 새 *_seg (매핑 테이블)")
        b = QtWidgets.QVBoxLayout(box)

        r1 = QtWidgets.QHBoxLayout()
        self.cmb_group_source = QtWidgets.QComboBox()
        self.txt_group_newcol = QtWidgets.QLineEdit("custom_seg")
        self.btn_group_build = QtWidgets.QPushButton("매핑표 만들기")
        style_button(self.btn_group_build, level=1)
        self.btn_group_build.clicked.connect(self._build_group_mapping)
        self.btn_group_apply = QtWidgets.QPushButton("매핑 적용 → 세그 생성")
        style_button(self.btn_group_apply, level=2)
        self.btn_group_apply.clicked.connect(self._apply_group_mapping)

        r1.addWidget(QtWidgets.QLabel("원본 컬럼"))
        r1.addWidget(self.cmb_group_source)
        r1.addSpacing(10)
        r1.addWidget(QtWidgets.QLabel("새 컬럼 이름"))
        r1.addWidget(self.txt_group_newcol)
        r1.addWidget(self.btn_group_build)
        r1.addWidget(self.btn_group_apply)
        b.addLayout(r1)

        self.tbl_group_map = DataFrameTable(editable=True, float_decimals=2)
        self.tbl_group_map.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked)
        b.addWidget(self.tbl_group_map, 1)

        # ✅ 선택한 값들(예: 10대/20대)을 하나로 묶어서 라벨 재네이밍
        merge_row = QtWidgets.QHBoxLayout()
        self.txt_group_merge_label = QtWidgets.QLineEdit("10-20대")
        self.btn_group_merge_apply = QtWidgets.QPushButton("선택 값 묶기(라벨 적용)")
        style_button(self.btn_group_merge_apply, level=1)
        self.btn_group_merge_apply.clicked.connect(self._merge_group_mapping_selected)

        merge_row.addWidget(QtWidgets.QLabel("선택한 원본값들을 같은 segment_label로 묶기"))
        merge_row.addWidget(self.txt_group_merge_label, 2)
        merge_row.addWidget(self.btn_group_merge_apply)
        b.addLayout(merge_row)

        layout.addWidget(box, 1)

        # Compose
        box2 = QtWidgets.QGroupBox("세그 조합: 여러 *_seg → 하나의 조합 세그")
        c = QtWidgets.QHBoxLayout(box2)

        self.lst_compose_segs = QtWidgets.QListWidget()
        self.lst_compose_segs.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        right = QtWidgets.QVBoxLayout()
        self.txt_compose_newcol = QtWidgets.QLineEdit("combo_seg")
        self.txt_compose_sep = QtWidgets.QLineEdit("|")
        self.btn_compose = QtWidgets.QPushButton("조합 세그 만들기")
        style_button(self.btn_compose, level=2)
        self.btn_compose.clicked.connect(self._compose_segs)

        right.addWidget(QtWidgets.QLabel("조합에 사용할 *_seg 컬럼들"))
        right.addWidget(QtWidgets.QLabel("새 컬럼 이름"))
        right.addWidget(self.txt_compose_newcol)
        right.addWidget(QtWidgets.QLabel("조합 구분자"))
        right.addWidget(self.txt_compose_sep)
        right.addWidget(self.btn_compose)
        right.addStretch(1)

        c.addWidget(self.lst_compose_segs, 2)
        c.addLayout(right, 1)
        layout.addWidget(box2, 1)

    def _apply_binary_recode(self):
        try:
            self._ensure_df()
            df = self.state.df
            col = self.cmb_bin_col.currentText().strip()
            if not col:
                raise RuntimeError("대상 컬럼을 선택해.")

            s = df[col]
            s_str = s.astype(str).str.strip()

            v1 = self.txt_bin_val1.text().strip()
            v2 = self.txt_bin_val2.text().strip()

            # 값1/값2를 비우면 자동으로 top2 유니크로 잡음
            if v1 == "" or v2 == "":
                top = s_str[s_str.notna()].value_counts().index.tolist()
                top = [t for t in top if t != "nan"]
                if len(top) < 2:
                    raise RuntimeError("유니크 값이 2개 이하라서 이진 RECODE가 어려움.")
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

            # ✅ else=기타 옵션
            if self.chk_bin_else_other.isChecked():
                other_lab = self.txt_bin_else_lab.text().strip() or "기타"
            else:
                other_lab = "NA"

            mapped = pd.Series(np.where(s_str == v1, lab1, np.where(s_str == v2, lab2, other_lab)), index=df.index)
            df[newcol] = mapped

            self.state.df = df
            self.tbl_preview.set_df(df)
            self._refresh_all_column_lists()
            self._set_status(f"{newcol} 생성 완료 (이진 RECODE).")
        except Exception as e:
            show_error(self, "Binary RECODE error", e)

    def _merge_group_mapping_selected(self):
        try:
            label = self.txt_group_merge_label.text().strip()
            if not label:
                raise RuntimeError("묶을 라벨 이름을 입력해.")
            sel = self.tbl_group_map.selectedItems()
            if not sel:
                raise RuntimeError("매핑표에서 묶을 행(원본값들)을 선택해.")

            rows = sorted(set([it.row() for it in sel]))
            # segment_label = col 2 (source_value, recode_name, segment_label)
            if self.tbl_group_map.columnCount() < 3:
                raise RuntimeError("매핑표 컬럼 구조가 부족함(3컬럼 필요).")

            for r in rows:
                item = self.tbl_group_map.item(r, 2)
                if item is None:
                    item = QtWidgets.QTableWidgetItem("")
                    self.tbl_group_map.setItem(r, 2, item)
                item.setText(label)

            self._set_status("선택 값 묶기 완료(선택된 행들의 segment_label 갱신).")
        except Exception as e:
            show_error(self, "Merge mapping error", e)

    def _build_group_mapping(self):
        try:
            self._ensure_df()
            df = self.state.df
            src = self.cmb_group_source.currentText().strip()
            if not src:
                raise RuntimeError("원본 컬럼 선택해.")

            vals = pd.Series(df[src].dropna().unique()).astype(str)
            # 보기 좋게 정렬
            try:
                vv = vals.astype(float)
                order = np.argsort(vv.values)
                vals = vals.iloc[order]
            except Exception:
                vals = vals.sort_values()

            # ✅ RECODE에서 NAME 끌어오기 (QUESTION/CODE/NAME)
            rec_name = {}
            if self.state.recode_df is not None and {"QUESTION", "CODE", "NAME"}.issubset(set(self.state.recode_df.columns)):
                r = self.state.recode_df.copy()
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
            self._set_status("매핑표 생성 완료 (segment_label 더블클릭해서 수정).")
        except Exception as e:
            show_error(self, "Mapping build error", e)

    def _apply_group_mapping(self):
        try:
            self._ensure_df()
            df = self.state.df
            src = self.cmb_group_source.currentText().strip()
            newcol = self.txt_group_newcol.text().strip()
            if not newcol:
                raise RuntimeError("새 컬럼 이름 입력해.")
            if not newcol.endswith("_seg"):
                newcol = newcol + "_seg"

            m = {}
            # ✅ 3컬럼(source_value, recode_name, segment_label) 구조 지원
            for r in range(self.tbl_group_map.rowCount()):
                k = self.tbl_group_map.item(r, 0).text() if self.tbl_group_map.item(r, 0) else ""
                if self.tbl_group_map.columnCount() >= 3:
                    v = self.tbl_group_map.item(r, 2).text() if self.tbl_group_map.item(r, 2) else ""
                else:
                    v = self.tbl_group_map.item(r, 1).text() if self.tbl_group_map.item(r, 1) else ""
                if k != "":
                    m[k] = v

            st = df[src].astype(str).str.strip()
            df[newcol] = st.map(m).fillna("NA")

            self.state.df = df
            self.tbl_preview.set_df(df)
            self._refresh_all_column_lists()
            self._set_status(f"{newcol} 생성 완료.")
        except Exception as e:
            show_error(self, "Apply mapping error", e)

    def _compose_segs(self):
        try:
            self._ensure_df()
            df = self.state.df
            cols = [it.text() for it in self.lst_compose_segs.selectedItems()]
            if len(cols) < 2:
                raise RuntimeError("2개 이상의 *_seg 컬럼 선택해.")
            newcol = self.txt_compose_newcol.text().strip()
            if not newcol:
                raise RuntimeError("새 컬럼 이름 입력해.")
            if not newcol.endswith("_seg"):
                newcol = newcol + "_seg"
            sep = self.txt_compose_sep.text()
            if sep is None or sep == "":
                sep = "|"

            df[newcol] = df[cols].astype(str).apply(lambda r: sep.join(r.values.tolist()), axis=1)
            self.state.df = df
            self.tbl_preview.set_df(df)
            self._refresh_all_column_lists()
            self._set_status(f"{newcol} (조합 세그) 생성 완료.")
        except Exception as e:
            show_error(self, "Compose error", e)

    # ---------------- Tab: Data Load
    def _build_tab_data(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Data loading")

        layout = QtWidgets.QVBoxLayout(tab)

        row1 = QtWidgets.QHBoxLayout()
        self.txt_path = QtWidgets.QLineEdit()
        self.btn_browse = QtWidgets.QPushButton("엑셀 파일 찾기…")
        style_button(self.btn_browse, level=1)
        self.btn_browse.clicked.connect(self._browse_excel)

        self.cmb_sheet = QtWidgets.QComboBox()
        self.btn_load = QtWidgets.QPushButton("불러오기")
        style_button(self.btn_load, level=2)
        self.btn_load.clicked.connect(self._load_excel)

        row1.addWidget(QtWidgets.QLabel("파일 경로:"))
        row1.addWidget(self.txt_path, 3)
        row1.addWidget(self.btn_browse)
        row1.addSpacing(10)
        row1.addWidget(QtWidgets.QLabel("시트:"))
        row1.addWidget(self.cmb_sheet, 1)
        row1.addWidget(self.btn_load)
        layout.addLayout(row1)

        self.lbl_data_info = QtWidgets.QLabel("아직 데이터가 없어.")
        self.lbl_data_info.setWordWrap(True)
        layout.addWidget(self.lbl_data_info)

        self.tbl_preview = DataFrameTable(float_decimals=2)
        layout.addWidget(self.tbl_preview, 1)

    def _browse_excel(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "엑셀 파일 선택", "", "Excel (*.xlsx *.xls)"
        )
        if not path:
            return
        self.txt_path.setText(path)
        self._populate_sheets(path)

    def _populate_sheets(self, path: str):
        try:
            xls = pd.ExcelFile(path, engine="openpyxl")
            self.cmb_sheet.clear()
            self.cmb_sheet.addItems(list(xls.sheet_names))
        except Exception as e:
            show_error(self, "시트 읽기 오류", e)

    def _load_excel(self):
        try:
            path = self.txt_path.text().strip()
            if not path or not os.path.exists(path):
                raise RuntimeError("유효한 엑셀 파일 경로가 아님.")
            if self.cmb_sheet.count() == 0:
                self._populate_sheets(path)
            sheet = self.cmb_sheet.currentText().strip()
            if not sheet:
                raise RuntimeError("시트 선택해.")

            df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
            self.state.df = df
            self.state.path = path
            self.state.sheet = sheet

            xls = pd.ExcelFile(path, engine="openpyxl")
            if "RECODE" in xls.sheet_names:
                rec = pd.read_excel(path, sheet_name="RECODE", engine="openpyxl")
                self.state.recode_df = normalize_recode_df(rec)
            else:
                self.state.recode_df = None

            self.tbl_preview.set_df(df)
            self.lbl_data_info.setText(
                f"불러옴: {os.path.basename(path)} / sheet={sheet} / rows={len(df):,} cols={df.shape[1]:,}"
            )
            self._set_status("데이터 로드 완료.")

            self._update_recode_tab()
            self._refresh_all_column_lists()
            self._reset_downstream_state()

        except Exception as e:
            show_error(self, "Load error", e)

    def _reset_downstream_state(self):
        self.state.pca_model = None
        self.state.pca_cols = None
        self.state.pca_scores = None
        self.state.pca_loadings = None

        self.state.dt_improve_pivot = None
        self.state.dt_split_best = None
        self.state.dt_full_nodes = None
        self.state.dt_full_split_groups = None
        self.state.dt_full_split_branches = None
        self.state.dt_full_path_info = None
        self.state.dt_full_condition_freq = None
        self.state.dt_full_selected = (None, None)

        self.state.hclust_assign = None

        self.state.demand_xy = None
        self.state.cluster_assign = None
        self.state.cluster_names = {}
        self.state.demand_seg_profile = None
        self.state.demand_seg_components = None
        self.state.demand_features_used = None
        self.state.manual_dirty = False

        # 아래는 이미 생성된 위젯들만 안전하게 초기화
        if hasattr(self, "tbl_pca_loadings"):
            self.tbl_pca_loadings.set_df(None)
        if hasattr(self, "tbl_dt_pivot"):
            self.tbl_dt_pivot.set_df(None)
        if hasattr(self, "tbl_dt_bestsplit"):
            self.tbl_dt_bestsplit.set_df(None)
        if hasattr(self, "tbl_hclust"):
            self.tbl_hclust.set_df(None)

        if hasattr(self, "tbl_dt_full_nodes"):
            self.tbl_dt_full_nodes.set_df(None)
        if hasattr(self, "tbl_dt_full_groups"):
            self.tbl_dt_full_groups.set_df(None)
        if hasattr(self, "tbl_dt_full_paths"):
            self.tbl_dt_full_paths.set_df(None)
        if hasattr(self, "tbl_dt_full_condfreq"):
            self.tbl_dt_full_condfreq.set_df(None)

        if hasattr(self, "tbl_split_groups_view"):
            self.tbl_split_groups_view.set_df(None)
        if hasattr(self, "tbl_split_detail"):
            self.tbl_split_detail.set_df(None)

        if hasattr(self, "_clear_demand_view"):
            self._clear_demand_view()

    # ---------------- Tab: PCA
    def _build_tab_pca(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "PCA")

        layout = QtWidgets.QHBoxLayout(tab)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("PCA Target에 넣을 변수 선택 (응답자 레벨):"))

        self.lst_pca_cols = QtWidgets.QListWidget()
        self.lst_pca_cols.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        left.addWidget(self.lst_pca_cols, 1)

        btnrow = QtWidgets.QHBoxLayout()
        self.btn_pca_check_sel = QtWidgets.QPushButton("선택 항목 체크")
        style_button(self.btn_pca_check_sel, level=1)
        self.btn_pca_uncheck_sel = QtWidgets.QPushButton("선택 항목 해제")
        style_button(self.btn_pca_uncheck_sel, level=1)
        self.btn_pca_check_all = QtWidgets.QPushButton("전체 체크")
        style_button(self.btn_pca_check_all, level=1)
        self.btn_pca_uncheck_all = QtWidgets.QPushButton("전체 해제")
        style_button(self.btn_pca_uncheck_all, level=1)
        self.btn_pca_check_sel.clicked.connect(lambda: self._set_checked_for_selected(self.lst_pca_cols, True))
        self.btn_pca_uncheck_sel.clicked.connect(lambda: self._set_checked_for_selected(self.lst_pca_cols, False))
        self.btn_pca_check_all.clicked.connect(lambda: self._set_all_checks(self.lst_pca_cols, True))
        self.btn_pca_uncheck_all.clicked.connect(lambda: self._set_all_checks(self.lst_pca_cols, False))
        btnrow.addWidget(self.btn_pca_check_sel)
        btnrow.addWidget(self.btn_pca_uncheck_sel)
        btnrow.addWidget(self.btn_pca_check_all)
        btnrow.addWidget(self.btn_pca_uncheck_all)
        left.addLayout(btnrow)

        ctrl = QtWidgets.QHBoxLayout()
        self.spin_pca_k = QtWidgets.QSpinBox()
        self.spin_pca_k.setRange(2, 30)
        self.spin_pca_k.setValue(12)
        self.btn_run_pca = QtWidgets.QPushButton("PCA 실행 → PCA1..k 생성")
        style_button(self.btn_run_pca, level=2)
        self.btn_run_pca.clicked.connect(self._run_pca_target)
        ctrl.addWidget(QtWidgets.QLabel("요인 개수 k"))
        ctrl.addWidget(self.spin_pca_k)
        ctrl.addWidget(self.btn_run_pca)
        left.addLayout(ctrl)

        self.lbl_pca_info = QtWidgets.QLabel("아직 PCA를 실행하지 않았음.")
        self.lbl_pca_info.setWordWrap(True)
        left.addWidget(self.lbl_pca_info)

        layout.addLayout(left, 2)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("PCA 적재값(Loadings) 미리보기:"))
        self.tbl_pca_loadings = DataFrameTable(float_decimals=2)
        right.addWidget(self.tbl_pca_loadings, 1)
        layout.addLayout(right, 3)

    def _run_pca_target(self):
        try:
            self._ensure_df()
            df = self.state.df
            cols = self._selected_checked_items(self.lst_pca_cols)
            if len(cols) < 2:
                raise RuntimeError("PCA에 넣을 변수를 2개 이상 체크해.")

            X = to_numeric_df(df, cols)
            X = X.dropna(axis=0, how="all")
            if len(X) < 10:
                raise RuntimeError("유효한 응답(row)이 너무 적음(NA 제외 후).")

            X_f = X.copy()
            for c in X_f.columns:
                m = X_f[c].mean()
                X_f[c] = X_f[c].fillna(m)

            k = int(self.spin_pca_k.value())
            k = min(k, X_f.shape[1])

            pca = PCA(n_components=k, random_state=42)
            scores = pca.fit_transform(X_f.values)

            score_cols = [f"PCA{i+1}" for i in range(k)]
            scores_df = pd.DataFrame(scores, index=X_f.index, columns=score_cols)

            for c in score_cols:
                df[c] = np.nan
                df.loc[scores_df.index, c] = scores_df[c].values

            loadings = pd.DataFrame(pca.components_.T, index=cols, columns=score_cols)

            # ✅ 미리보기는 절댓값 큰 변수부터 위로 정렬(“가운데 딱 보이게”)
            disp = loadings.copy()
            disp["_maxabs_"] = disp.abs().max(axis=1)
            disp = disp.sort_values("_maxabs_", ascending=False).drop(columns=["_maxabs_"])
            disp = disp.reset_index().rename(columns={"index": "variable"})

            evr = pca.explained_variance_ratio_
            self.lbl_pca_info.setText(
                "Created {k} PCs. 설명분산(앞 5개): ".format(k=k)
                + ", ".join([f"{v:.3f}" for v in evr[:5]])
            )
            self.tbl_pca_loadings.set_df(disp)

            self.state.df = df
            self.state.pca_model = pca
            self.state.pca_cols = cols
            self.state.pca_scores = scores_df
            self.state.pca_loadings = loadings  # 원본은 그대로 유지

            self.tbl_preview.set_df(df)
            self._refresh_all_column_lists()
            self._set_status("PCA Target 생성 완료 (PCA1..k 추가).")
        except Exception as e:
            show_error(self, "PCA error", e)

    # ---------------- Tab: Decision Tree & HCLUST
    def _build_tab_tree_hclust(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Decision Tree")

        layout = QtWidgets.QVBoxLayout(tab)

        head = QtWidgets.QLabel(
            "입력 → 처리 → 출력\n"
            "- 입력: (dep) PCA1..k + (optional) target dep (예: C2)\n"
            "- 처리A: dep×ind 단변량 '루트' split → Improve Pivot(Rel) + Best Split\n"
            "- 처리B: dep/ind 선택 → 단변량 트리 전체(루트~말단) 분기/경로 출력\n"
            "- 처리C: Improve Pivot 기반 HCLUST\n"
            "※ Exclude 체크된 변수는 ind에서 제외됨."
        )
        head.setWordWrap(True)
        layout.addWidget(head)

        row = QtWidgets.QHBoxLayout()
        self.chk_use_all_pca_as_dep = QtWidgets.QCheckBox("모든 PCA[1..k]를 dep로 사용")
        self.chk_use_all_pca_as_dep.setChecked(True)
        self.cmb_dep_extra = QtWidgets.QComboBox()
        self.btn_run_tree = QtWidgets.QPushButton("Decision Tree Outputs 실행")
        style_button(self.btn_run_tree, level=2)
        self.btn_run_tree.clicked.connect(self._run_decision_tree_outputs)
        row.addWidget(self.chk_use_all_pca_as_dep)
        row.addSpacing(14)
        row.addWidget(QtWidgets.QLabel("추가 dep(선택):"))
        row.addWidget(self.cmb_dep_extra)
        row.addWidget(self.btn_run_tree)
        layout.addLayout(row)

        # exclude vars
        ex_box = QtWidgets.QGroupBox("분석에서 제외할 변수 선택 (체크 = 제외)")
        ex_layout = QtWidgets.QVBoxLayout(ex_box)
        ex_row = QtWidgets.QHBoxLayout()
        self.txt_dt_exclude_filter = QtWidgets.QLineEdit()
        self.txt_dt_exclude_filter.setPlaceholderText("컬럼명 필터...")
        self.txt_dt_exclude_filter.textChanged.connect(self._filter_dt_exclude_list)
        self.btn_dt_excl_check_sel = QtWidgets.QPushButton("선택 제외")
        style_button(self.btn_dt_excl_check_sel, level=1)
        self.btn_dt_excl_uncheck_sel = QtWidgets.QPushButton("선택 제외 해제")
        style_button(self.btn_dt_excl_uncheck_sel, level=1)
        self.btn_dt_excl_check_sel.clicked.connect(lambda: self._set_checked_for_selected(self.lst_dt_exclude, True))
        self.btn_dt_excl_uncheck_sel.clicked.connect(lambda: self._set_checked_for_selected(self.lst_dt_exclude, False))
        ex_row.addWidget(self.txt_dt_exclude_filter, 2)
        ex_row.addWidget(self.btn_dt_excl_check_sel)
        ex_row.addWidget(self.btn_dt_excl_uncheck_sel)
        ex_layout.addLayout(ex_row)

        self.lst_dt_exclude = QtWidgets.QListWidget()
        self.lst_dt_exclude.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        ex_layout.addWidget(self.lst_dt_exclude, 1)
        layout.addWidget(ex_box)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # A) Pivot
        w1 = QtWidgets.QWidget()
        l1 = QtWidgets.QVBoxLayout(w1)
        l1.addWidget(QtWidgets.QLabel("Improve Pivot (Rel)  [dep 열 / ind 행]"))
        self.tbl_dt_pivot = DataFrameTable(float_decimals=2)
        l1.addWidget(self.tbl_dt_pivot, 1)

        # B) Best split
        w2 = QtWidgets.QWidget()
        l2 = QtWidgets.QVBoxLayout(w2)
        l2.addWidget(QtWidgets.QLabel("Best Split (root) per dep×ind"))
        self.tbl_dt_bestsplit = DataFrameTable(float_decimals=2)
        l2.addWidget(self.tbl_dt_bestsplit, 1)

        # C) Full tree viewer
        w3 = QtWidgets.QWidget()
        l3 = QtWidgets.QVBoxLayout(w3)

        selrow = QtWidgets.QHBoxLayout()
        self.cmb_dt_full_dep = QtWidgets.QComboBox()
        self.cmb_dt_full_ind = QtWidgets.QComboBox()
        self.spin_dt_full_depth = QtWidgets.QSpinBox()
        self.spin_dt_full_depth.setRange(1, 30)
        self.spin_dt_full_depth.setValue(6)
        self.spin_dt_full_minleaf = QtWidgets.QSpinBox()
        self.spin_dt_full_minleaf.setRange(1, 999999)
        self.spin_dt_full_minleaf.setValue(1)
        self.spin_dt_full_minsplit = QtWidgets.QSpinBox()
        self.spin_dt_full_minsplit.setRange(2, 999999)
        self.spin_dt_full_minsplit.setValue(2)
        self.spin_dt_full_minimpr = QtWidgets.QDoubleSpinBox()
        self.spin_dt_full_minimpr.setDecimals(3)
        self.spin_dt_full_minimpr.setSingleStep(0.01)
        self.spin_dt_full_minimpr.setValue(0.0)

        self.btn_dt_full_run = QtWidgets.QPushButton("선택 dep×ind 전체 분기 보기")
        style_button(self.btn_dt_full_run, level=2)
        self.btn_dt_full_run.clicked.connect(self._run_dt_full_for_selected)

        selrow.addWidget(QtWidgets.QLabel("dep"))
        selrow.addWidget(self.cmb_dt_full_dep, 2)
        selrow.addWidget(QtWidgets.QLabel("ind"))
        selrow.addWidget(self.cmb_dt_full_ind, 3)
        selrow.addWidget(QtWidgets.QLabel("max_depth"))
        selrow.addWidget(self.spin_dt_full_depth)
        selrow.addWidget(QtWidgets.QLabel("min_leaf"))
        selrow.addWidget(self.spin_dt_full_minleaf)
        selrow.addWidget(QtWidgets.QLabel("min_split"))
        selrow.addWidget(self.spin_dt_full_minsplit)
        selrow.addWidget(QtWidgets.QLabel("min_impr_rel"))
        selrow.addWidget(self.spin_dt_full_minimpr)
        selrow.addWidget(self.btn_dt_full_run)
        l3.addLayout(selrow)

        fullsplit = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        a = QtWidgets.QWidget()
        la = QtWidgets.QVBoxLayout(a)
        la.addWidget(QtWidgets.QLabel("Nodes (전체 노드)"))
        self.tbl_dt_full_nodes = DataFrameTable(float_decimals=2)
        la.addWidget(self.tbl_dt_full_nodes, 1)

        b = QtWidgets.QWidget()
        lb = QtWidgets.QVBoxLayout(b)
        lb.addWidget(QtWidgets.QLabel("Split_Groups (내부 분기 정보)"))
        self.tbl_dt_full_groups = DataFrameTable(float_decimals=2)
        lb.addWidget(self.tbl_dt_full_groups, 1)

        c = QtWidgets.QWidget()
        lc = QtWidgets.QVBoxLayout(c)
        lc.addWidget(QtWidgets.QLabel("Path_Info (leaf path)"))
        self.tbl_dt_full_paths = DataFrameTable(float_decimals=2)
        lc.addWidget(self.tbl_dt_full_paths, 1)

        d = QtWidgets.QWidget()
        ld = QtWidgets.QVBoxLayout(d)
        ld.addWidget(QtWidgets.QLabel("Condition_Freq (조건 등장 빈도)"))
        self.tbl_dt_full_condfreq = DataFrameTable(float_decimals=2)
        ld.addWidget(self.tbl_dt_full_condfreq, 1)

        fullsplit.addWidget(a)
        fullsplit.addWidget(b)
        fullsplit.addWidget(c)
        fullsplit.addWidget(d)
        fullsplit.setSizes([420, 420, 420, 320])
        l3.addWidget(fullsplit, 1)

        splitter.addWidget(w1)
        splitter.addWidget(w2)
        splitter.addWidget(w3)
        splitter.setSizes([220, 220, 480])
        layout.addWidget(splitter, 1)

        # HCLUST stage
        box = QtWidgets.QGroupBox("Tree → HCLUST (Agglomerative on Improve Pivot)")
        bl = QtWidgets.QHBoxLayout(box)
        self.spin_hclust_k = QtWidgets.QSpinBox()
        self.spin_hclust_k.setRange(2, 30)
        self.spin_hclust_k.setValue(6)
        self.btn_run_hclust = QtWidgets.QPushButton("HCLUST 실행")
        style_button(self.btn_run_hclust, level=2)
        self.btn_run_hclust.clicked.connect(self._run_hclust_from_pivot)
        bl.addWidget(QtWidgets.QLabel("클러스터 개수 k"))
        bl.addWidget(self.spin_hclust_k)
        bl.addWidget(self.btn_run_hclust)

        self.tbl_hclust = DataFrameTable(float_decimals=2)
        box_layout = QtWidgets.QVBoxLayout()
        box_layout.addLayout(bl)
        box_layout.addWidget(self.tbl_hclust, 1)
        box.setLayout(box_layout)
        layout.addWidget(box)

    def _filter_dt_exclude_list(self):
        term = self.txt_dt_exclude_filter.text().strip().lower()
        for i in range(self.lst_dt_exclude.count()):
            it = self.lst_dt_exclude.item(i)
            it.setHidden(term not in it.text().lower())

    def _get_excluded_vars_set(self) -> set:
        exc = set()
        for i in range(self.lst_dt_exclude.count()):
            it = self.lst_dt_exclude.item(i)
            if it.checkState() == QtCore.Qt.CheckState.Checked:
                exc.add(it.text())
        return exc

    def _run_decision_tree_outputs(self):
        try:
            self._ensure_df()
            df = self.state.df

            pca_cols = [c for c in df.columns if str(c).startswith("PCA") and str(c)[3:].isdigit()]
            pca_cols = sorted(pca_cols, key=lambda x: int(str(x)[3:]))

            deps: List[str] = []
            if self.chk_use_all_pca_as_dep.isChecked() and pca_cols:
                deps.extend(pca_cols)

            extra = self.cmb_dep_extra.currentText().strip()
            if extra and extra != "(없음)" and extra not in deps:
                deps.append(extra)

            if not deps:
                raise RuntimeError("Dependent targets가 없음. PCA 만들거나 extra dep 선택해.")

            excluded = self._get_excluded_vars_set()
            all_vars = list(df.columns)
            ind_vars = [c for c in all_vars if c not in deps and c != "resp_id" and c not in excluded]

            if len(ind_vars) == 0:
                raise RuntimeError("Independent variables가 없음(Exclude가 너무 많거나 dep만 남음).")

            best_rows = []
            pivot = pd.DataFrame(index=ind_vars, columns=deps, dtype=float)

            for dep in deps:
                y = df[dep]
                task = "reg"
                if dep == extra and extra != "(없음)":
                    task = "class" if is_categorical_series(y) else "reg"

                for ind in ind_vars:
                    x = df[ind]
                    best, _ = univariate_best_split(y, x, task=task)
                    if best is None:
                        pivot.loc[ind, dep] = np.nan
                        continue
                    pivot.loc[ind, dep] = best["improve_rel"]
                    row = {"dep": dep, "ind": ind}
                    row.update(best)
                    best_rows.append(row)

            best_df = pd.DataFrame(best_rows)
            pivot_reset = pivot.reset_index().rename(columns={"index": "ind"})

            self.tbl_dt_pivot.set_df(pivot_reset)
            self.tbl_dt_bestsplit.set_df(best_df)

            self.state.dt_improve_pivot = pivot_reset
            self.state.dt_split_best = best_df

            self.cmb_dt_full_dep.clear()
            self.cmb_dt_full_dep.addItems(deps)
            self.cmb_dt_full_ind.clear()
            self.cmb_dt_full_ind.addItems(ind_vars)

            # Split Results 탭용 dep/ind 콤보도 채움
            self.cmb_split_dep.clear()
            self.cmb_split_dep.addItems(deps)
            self.cmb_split_ind.clear()
            self.cmb_split_ind.addItems(ind_vars)

            self._set_status("Decision Tree outputs 생성 완료 (Pivot + Best Split).")

        except Exception as e:
            show_error(self, "Decision Tree error", e)

    def _compute_full_tree_internal(self, dep: str, ind: str):
        df = self.state.df
        if df is None:
            raise RuntimeError("데이터 없음.")

        if dep not in df.columns or ind not in df.columns:
            raise RuntimeError("dep/ind 컬럼이 데이터에 없음.")

        task = "class" if is_categorical_series(df[dep]) and not (str(dep).startswith("PCA") and str(dep)[3:].isdigit()) else "reg"

        nodes_df, split_groups, branches, path_info, cond_freq = build_univariate_tree_full(
            df=df, dep=dep, ind=ind, task=task,
            max_depth=int(self.spin_dt_full_depth.value()),
            min_leaf=int(self.spin_dt_full_minleaf.value()),
            min_split=int(self.spin_dt_full_minsplit.value()),
            max_unique_cat=50,
            min_improve_rel=float(self.spin_dt_full_minimpr.value()),
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
                raise RuntimeError("dep/ind 선택해.")

            task, nodes_df, split_groups, path_info, cond_freq = self._compute_full_tree_internal(dep, ind)

            self.tbl_dt_full_nodes.set_df(nodes_df)
            self.tbl_dt_full_groups.set_df(split_groups)
            self.tbl_dt_full_paths.set_df(path_info)
            self.tbl_dt_full_condfreq.set_df(cond_freq)

            self._set_status(f"Full branches ready: dep={dep}, ind={ind} (task={task})")
        except Exception as e:
            show_error(self, "Full branch error", e)

    def _run_hclust_from_pivot(self):
        try:
            if self.state.dt_improve_pivot is None:
                raise RuntimeError("먼저 Decision Tree Outputs 실행해.")
            piv = self.state.dt_improve_pivot.copy()
            if "ind" not in piv.columns:
                raise RuntimeError("Pivot 형식이 이상함.")

            X = piv.set_index("ind").fillna(0.0).values
            if X.shape[0] < 2:
                raise RuntimeError("클러스터링할 변수가 부족함.")

            k = int(self.spin_hclust_k.value())
            k = max(2, min(k, X.shape[0]))

            model = AgglomerativeClustering(n_clusters=k, linkage="ward") if X.shape[1] > 1 else AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(X) + 1

            out = pd.DataFrame({
                "variable(ind)": piv["ind"].values,
                "hclust_cluster": labels
            }).sort_values(["hclust_cluster", "variable(ind)"]).reset_index(drop=True)

            self.tbl_hclust.set_df(out)
            self.state.hclust_assign = out
            self._set_status("HCLUST 완료.")
        except Exception as e:
            show_error(self, "HCLUST error", e)

    # ---------------- Tab: Split Results
    def _build_tab_split_results(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Split Results")

        layout = QtWidgets.QVBoxLayout(tab)

        info = QtWidgets.QLabel(
            "의사결정트리 split 탐색 탭\n"
            "- dep/ind 선택 → 전체 트리 계산\n"
            "- 스플릿 선택 → 좌/우 가지 impurity 비교\n"
            "※ 불순도: 회귀=MSE, 분류=Gini"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        ctrl = QtWidgets.QHBoxLayout()
        self.cmb_split_dep = QtWidgets.QComboBox()
        self.cmb_split_ind = QtWidgets.QComboBox()
        self.btn_split_compute = QtWidgets.QPushButton("선택 dep×ind로 트리 계산")
        style_button(self.btn_split_compute, level=2)
        self.btn_split_compute.clicked.connect(self._split_compute_tree)

        ctrl.addWidget(QtWidgets.QLabel("dep"))
        ctrl.addWidget(self.cmb_split_dep, 2)
        ctrl.addWidget(QtWidgets.QLabel("ind"))
        ctrl.addWidget(self.cmb_split_ind, 3)
        ctrl.addWidget(self.btn_split_compute)
        layout.addLayout(ctrl)

        srow = QtWidgets.QHBoxLayout()
        self.cmb_split_select = QtWidgets.QComboBox()
        self.cmb_split_select.currentIndexChanged.connect(self._split_update_detail)
        srow.addWidget(QtWidgets.QLabel("Split 선택"))
        srow.addWidget(self.cmb_split_select, 4)
        layout.addLayout(srow)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        topw = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(topw)
        tl.addWidget(QtWidgets.QLabel("Split_Groups (내부 분기 전체)"))
        self.tbl_split_groups_view = DataFrameTable(float_decimals=2)
        tl.addWidget(self.tbl_split_groups_view, 1)

        botw = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(botw)
        self.lbl_split_imp = QtWidgets.QLabel("아직 split 선택 안 함.")
        self.lbl_split_imp.setWordWrap(True)
        bl.addWidget(self.lbl_split_imp)
        bl.addWidget(QtWidgets.QLabel("선택 split 좌/우 상세:"))
        self.tbl_split_detail = DataFrameTable(float_decimals=2)
        bl.addWidget(self.tbl_split_detail, 1)

        splitter.addWidget(topw)
        splitter.addWidget(botw)
        splitter.setSizes([360, 360])
        layout.addWidget(splitter, 1)

    def _split_compute_tree(self):
        try:
            self._ensure_df()
            dep = self.cmb_split_dep.currentText().strip()
            ind = self.cmb_split_ind.currentText().strip()
            if not dep or not ind:
                raise RuntimeError("dep/ind 선택해.")

            _, nodes_df, split_groups, path_info, cond_freq = self._compute_full_tree_internal(dep, ind)

            self.tbl_split_groups_view.set_df(split_groups)
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

            self._set_status(f"Split 결과 계산 완료 (dep={dep}, ind={ind}).")
        except Exception as e:
            show_error(self, "Split compute error", e)

    def _parse_condition_to_mask(self, df: pd.DataFrame, ind: str, cond: str) -> np.ndarray:
        """Split_Groups의 left/right 문자열에서 mask 생성.
           - 숫자 원본인데 '== 1' 같은 조건이 들어올 때 타입 꼬임 방지
        """
        s = df[ind]
        cond = str(cond).strip()

        def _strip_quotes(v: str) -> str:
            vv = v.strip()
            if (vv.startswith("'") and vv.endswith("'")) or (vv.startswith('"') and vv.endswith('"')):
                vv = vv[1:-1].strip()
            return vv

        def _try_numeric_compare(op: str, val_str: str) -> Optional[np.ndarray]:
            try:
                vv = float(val_str)
            except Exception:
                return None
            sn = pd.to_numeric(s, errors="coerce").values.astype(float)
            ok = np.isfinite(sn)
            if ok.sum() == 0:
                return None
            if op == "==":
                m = (sn == vv)
            elif op == "!=":
                m = (sn != vv)
            elif op == "<=":
                m = (sn <= vv)
            elif op == ">":
                m = (sn > vv)
            else:
                return None
            m[~ok] = False
            return m

        if "==" in cond:
            _, val = cond.split("==", 1)
            val = _strip_quotes(val.strip())
            m = _try_numeric_compare("==", val)
            if m is not None:
                return m
            return (s.astype(str).str.strip() == val).values

        if "!=" in cond:
            _, val = cond.split("!=", 1)
            val = _strip_quotes(val.strip())
            m = _try_numeric_compare("!=", val)
            if m is not None:
                return m
            return (s.astype(str).str.strip() != val).values

        if "<=" in cond:
            _, val = cond.split("<=", 1)
            val = _strip_quotes(val.strip())
            m = _try_numeric_compare("<=", val)
            if m is not None:
                return m
            try:
                thr = float(val)
                return pd.to_numeric(s, errors="coerce").values <= thr
            except Exception:
                return np.zeros(len(df), dtype=bool)

        if ">" in cond:
            _, val = cond.split(">", 1)
            val = _strip_quotes(val.strip())
            m = _try_numeric_compare(">", val)
            if m is not None:
                return m
            try:
                thr = float(val)
                return pd.to_numeric(s, errors="coerce").values > thr
            except Exception:
                return np.zeros(len(df), dtype=bool)

        return np.zeros(len(df), dtype=bool)

    def _split_update_detail(self):
        try:
            self._ensure_df()
            df = self.state.df
            if df is None:
                return
            split_groups = self.state.dt_full_split_groups
            dep, ind = self.state.dt_full_selected
            if split_groups is None or split_groups.empty or dep is None or ind is None:
                self.tbl_split_detail.set_df(None)
                self.lbl_split_imp.setText("먼저 dep×ind로 트리 계산해.")
                return

            idx = self.cmb_split_select.currentIndex()
            if idx < 0:
                return
            split_num = self.cmb_split_select.currentData()
            if split_num is None:
                return

            row = split_groups[split_groups["split_num"] == split_num].iloc[0]
            left_cond = row["left_group"]
            right_cond = row["right_group"]

            y = df[dep]
            task = "class" if is_categorical_series(y) and not (str(dep).startswith("PCA") and str(dep)[3:].isdigit()) else "reg"

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

            if np.isfinite(impL) and np.isfinite(impR):
                if impL < impR:
                    better = "왼쪽 가지가 더 깨끗함 (impurity 낮음)"
                elif impR < impL:
                    better = "오른쪽 가지가 더 깨끗함 (impurity 낮음)"
                else:
                    better = "좌/우 impurity 거의 동일"
            else:
                better = "NA가 많아서 impurity 비교가 애매함."

            base_text = f"Split {int(split_num)}: {left_cond}  /  {right_cond}\n"
            base_text += f"[task={task}] left_imp={fmt_float(impL,2)}, right_imp={fmt_float(impR,2)}  →  {better}"
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
            show_error(self, "Split detail error", e)

    # ---------------- Demand Space
    # NEW: Demand Space -> Segmentation setting & Segmentation editing
    # ----------------------------------------------------------------

    # 7. Segmentation setting (New Split)
    def _build_tab_seg_setting(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Segmentation setting")

        layout = QtWidgets.QHBoxLayout(tab)
        
        # Left: Settings
        left = QtWidgets.QVBoxLayout()

        mode_box = QtWidgets.QGroupBox("Mode")
        mlay = QtWidgets.QHBoxLayout(mode_box)
        self.cmb_demand_mode = QtWidgets.QComboBox()
        self.cmb_demand_mode.addItems([
            "Segments-as-points (여자|1020 ...)",
            "Variables-as-points (컬럼명 점찍기)"
        ])
        self.cmb_demand_mode.currentTextChanged.connect(self._on_demand_mode_changed)
        mlay.addWidget(QtWidgets.QLabel("Type"))
        mlay.addWidget(self.cmb_demand_mode, 1)
        left.addWidget(mode_box)

        seg_box = QtWidgets.QGroupBox("Segments-as-points input")
        seg_l = QtWidgets.QVBoxLayout(seg_box)

        seg_l.addWidget(QtWidgets.QLabel("세그 라벨로 쓸 *_seg 컬럼 체크:"))
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
        r.addWidget(QtWidgets.QLabel("라벨 구분자"))
        r.addWidget(self.txt_demand_seg_sep)
        r.addSpacing(12)
        r.addWidget(QtWidgets.QLabel("Target 변수"))
        r.addWidget(self.cmb_demand_target, 2)
        r.addSpacing(12)
        r.addWidget(QtWidgets.QLabel("세그 최소 n"))
        r.addWidget(self.spin_demand_min_n)
        seg_l.addLayout(r)

        feat = QtWidgets.QHBoxLayout()
        self.chk_demand_use_pca12 = QtWidgets.QCheckBox("PCA1~PCAk를 프로파일 feature로 사용")
        self.chk_demand_use_pca12.setChecked(True)
        self.spin_demand_pca_k = QtWidgets.QSpinBox()
        self.spin_demand_pca_k.setRange(2, 30)
        self.spin_demand_pca_k.setValue(12)
        feat.addWidget(self.chk_demand_use_pca12)
        feat.addSpacing(12)
        feat.addWidget(QtWidgets.QLabel("사용할 PCA 개수 k"))
        feat.addWidget(self.spin_demand_pca_k)
        seg_l.addLayout(feat)

        left.addWidget(seg_box)

        var_box = QtWidgets.QGroupBox("Variables-as-points input")
        var_l = QtWidgets.QVBoxLayout(var_box)
        var_l.addWidget(QtWidgets.QLabel("점으로 찍을 변수(컬럼) 체크:"))
        self.lst_demand_vars = QtWidgets.QListWidget()
        self.lst_demand_vars.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        var_l.addWidget(self.lst_demand_vars, 2)

        vbtn = QtWidgets.QHBoxLayout()
        self.btn_demand_check_sel = QtWidgets.QPushButton("선택 체크")
        style_button(self.btn_demand_check_sel, level=1)
        self.btn_demand_uncheck_sel = QtWidgets.QPushButton("선택 해제")
        style_button(self.btn_demand_uncheck_sel, level=1)
        self.btn_demand_check_sel.clicked.connect(lambda: self._set_checked_for_selected(self.lst_demand_vars, True))
        self.btn_demand_uncheck_sel.clicked.connect(lambda: self._set_checked_for_selected(self.lst_demand_vars, False))
        vbtn.addWidget(self.btn_demand_check_sel)
        vbtn.addWidget(self.btn_demand_uncheck_sel)
        var_l.addLayout(vbtn)

        left.addWidget(var_box)

        row = QtWidgets.QHBoxLayout()
        self.cmb_demand_coord = QtWidgets.QComboBox()
        self.cmb_demand_coord.addItems(["PCA (PC1/PC2)", "MDS (1-corr distance)"])
        self.spin_demand_k = QtWidgets.QSpinBox()
        self.spin_demand_k.setRange(2, 30)
        self.spin_demand_k.setValue(6)
        self.btn_run_demand = QtWidgets.QPushButton("Demand Space 실행")
        style_button(self.btn_run_demand, level=2)
        self.btn_run_demand.clicked.connect(self._run_demand_space)

        row.addWidget(QtWidgets.QLabel("좌표 방식"))
        row.addWidget(self.cmb_demand_coord)
        row.addWidget(QtWidgets.QLabel("k (kmeans 클러스터 수)"))
        row.addWidget(self.spin_demand_k)
        row.addWidget(self.btn_run_demand)
        left.addLayout(row)

        layout.addLayout(left, 2)

        # Right: Preview Plot (Read-onlyish)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("Preview (수정은 'Segmentation editing' 탭에서)"))
        self.plot_preview = DemandClusterPlot(editable=False) # Non-editable preview
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
        self.chk_demand_use_pca12.setEnabled(seg_mode)
        self.spin_demand_pca_k.setEnabled(seg_mode)

        self.lst_demand_vars.setEnabled(not seg_mode)
        self.btn_demand_check_sel.setEnabled(not seg_mode)
        self.btn_demand_uncheck_sel.setEnabled(not seg_mode)

    # 8. Segmentation editing (New Split)
    def _build_tab_seg_editing(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Segmentation editing")
        layout = QtWidgets.QHBoxLayout(tab)

        # Left: Controls & Summary
        left = QtWidgets.QVBoxLayout()
        
        # Mode Toggle (Exclusive)
        toggle_group = QtWidgets.QGroupBox("편집 모드 선택 (Toggle)")
        tgl_lay = QtWidgets.QVBoxLayout(toggle_group)
        self.radio_edit_points = QtWidgets.QRadioButton("점 편집 모드 (이동/병합)")
        self.radio_edit_points.setToolTip("점을 드래그하여 이동하거나 다른 세그먼트에 병합합니다. 배경 이동(Pan)은 잠깁니다.")
        self.radio_edit_view = QtWidgets.QRadioButton("화면 이동 모드 (Pan)")
        self.radio_edit_view.setToolTip("배경을 드래그하여 화면을 이동합니다. 점 편집은 잠깁니다.")
        self.radio_edit_view.setChecked(True) # Default to View mode for safety

        self.radio_edit_points.toggled.connect(self._on_edit_mode_toggled)
        self.radio_edit_view.toggled.connect(self._on_edit_mode_toggled)
        
        tgl_lay.addWidget(self.radio_edit_points)
        tgl_lay.addWidget(self.radio_edit_view)
        left.addWidget(toggle_group)

        # Point Move Option
        opt_group = QtWidgets.QGroupBox("점 편집 옵션")
        olay = QtWidgets.QVBoxLayout(opt_group)
        self.chk_free_move_points = QtWidgets.QCheckBox("점 자유이동 (ON: 위치이동 / OFF: 병합)")
        self.chk_free_move_points.toggled.connect(lambda v: self.plot_edit.set_free_move_points(v))
        self.chk_show_all_point_labels = QtWidgets.QCheckBox("점 라벨 모두 표시")
        self.chk_show_all_point_labels.toggled.connect(lambda v: self.plot_edit.set_show_all_point_labels(v))
        self.btn_auto_labels = QtWidgets.QPushButton("라벨 자동정리")
        style_button(self.btn_auto_labels, level=1)
        self.btn_auto_labels.clicked.connect(lambda: self.plot_edit.auto_arrange_labels())
        self.btn_reset_label_pos = QtWidgets.QPushButton("라벨 위치 초기화")
        style_button(self.btn_reset_label_pos, level=1)
        self.btn_reset_label_pos.clicked.connect(lambda: self.plot_edit.reset_label_positions())
        
        olay.addWidget(self.chk_free_move_points)
        olay.addWidget(self.chk_show_all_point_labels)
        olay.addWidget(self.btn_auto_labels)
        olay.addWidget(self.btn_reset_label_pos)
        left.addWidget(opt_group)

        # Summary Table
        left.addWidget(QtWidgets.QLabel("Cluster Summary (n수 동적 변경)"))
        self.tbl_cluster_summary = DataFrameTable(float_decimals=2)
        left.addWidget(self.tbl_cluster_summary, 1)

        # Rename Cluster
        rename_box = QtWidgets.QHBoxLayout()
        self.spin_rename_cluster_id = QtWidgets.QSpinBox()
        self.spin_rename_cluster_id.setRange(1, 999)
        self.txt_rename_cluster = QtWidgets.QLineEdit("Cluster name")
        self.btn_rename_cluster = QtWidgets.QPushButton("클러스터 이름 변경")
        style_button(self.btn_rename_cluster, level=1)
        self.btn_rename_cluster.clicked.connect(self._rename_cluster)
        rename_box.addWidget(QtWidgets.QLabel("클러스터 id"))
        rename_box.addWidget(self.spin_rename_cluster_id)
        rename_box.addWidget(self.txt_rename_cluster)
        rename_box.addWidget(self.btn_rename_cluster)
        left.addLayout(rename_box)

        layout.addLayout(left, 1)

        # Right: Interactive Plot
        right = QtWidgets.QVBoxLayout()
        self.plot_edit = DemandClusterPlot(editable=True)
        self.plot_edit.sigClustersChanged.connect(self._on_manual_clusters_changed)
        self.plot_edit.sigCoordsChanged.connect(self._on_manual_coords_changed)
        right.addWidget(self.plot_edit, 1)
        layout.addLayout(right, 3)

        # Initial Mode Sync
        self._on_edit_mode_toggled()

    def _on_edit_mode_toggled(self):
        # Radio button logic
        is_point_edit = self.radio_edit_points.isChecked()
        self.plot_edit.set_edit_mode_active(is_point_edit)
        
        if is_point_edit:
            self._set_status("점 편집 모드: 점/라벨 드래그 가능 (배경 이동 잠김)")
        else:
            self._set_status("화면 이동 모드: 배경 드래그로 이동 (점 편집 잠김)")

    def _clear_demand_view(self):
        empty_args = ([], [], np.zeros((0, 2)), np.zeros((0,), dtype=int), {})
        self.plot_preview.set_data(*empty_args)
        self.plot_edit.set_data(*empty_args)
        self.tbl_cluster_summary.set_df(None)
        self.lbl_demand_status.setText("아직 Demand Space 실행 안 함.")

    def _variables_as_matrix(self, cols: List[str]) -> Tuple[np.ndarray, List[str]]:
        df = self.state.df
        X = to_numeric_df(df, cols)

        Xf = X.copy()
        for c in Xf.columns:
            Xf[c] = Xf[c].fillna(Xf[c].mean())

        M = Xf.values
        V = M.T
        mu = V.mean(axis=1, keepdims=True)
        sd = V.std(axis=1, keepdims=True) + 1e-9
        Vz = (V - mu) / sd
        return Vz, cols

    def _build_segment_profiles(self, seg_cols: List[str], sep: str, use_pca: bool, pca_k: int,
                                target: Optional[str], min_n: int) -> Tuple[pd.DataFrame, List[str]]:
        df = self.state.df.copy()

        seg_df = df[seg_cols].astype(str).fillna("NA")
        label = seg_df.apply(lambda r: sep.join(r.values.tolist()), axis=1)

        feat_cols: List[str] = []
        base_feat = pd.DataFrame(index=df.index)

        if use_pca:
            pca_cols = [c for c in df.columns if str(c).startswith("PCA") and str(c)[3:].isdigit()]
            pca_cols = sorted(pca_cols, key=lambda x: int(str(x)[3:]))
            pca_cols = pca_cols[:max(1, int(pca_k))]
            if not pca_cols:
                raise RuntimeError("PCA1..k 컬럼 없음. 먼저 PCA 생성해.")
            base_feat[pca_cols] = df[pca_cols]
            feat_cols.extend(pca_cols)

        if target and target != "(없음)":
            if target not in df.columns:
                raise RuntimeError("Target 컬럼이 데이터에 없음.")
            t = df[target]
            if is_categorical_series(t):
                tt = t.astype(str).fillna("NA")
                top_levels = tt.value_counts().index.tolist()
                if len(top_levels) > 20:
                    keep = set(top_levels[:20])
                    tt = tt.apply(lambda x: x if x in keep else "OTHER")
                dummies = pd.get_dummies(tt, prefix=f"{target}")
                base_feat = base_feat.join(dummies)
                feat_cols.extend(list(dummies.columns))
            else:
                base_feat[target] = pd.to_numeric(t, errors="coerce")
                feat_cols.append(target)

        if not feat_cols:
            raise RuntimeError("Profile feature가 없음(PCA 또는 target 중 최소 1개 필요).")

        base_feat["_SEG_LABEL_"] = label.values

        grp = base_feat.groupby("_SEG_LABEL_", dropna=False)
        prof = grp[feat_cols].mean(numeric_only=True)
        prof["n"] = grp.size().astype(int)

        prof = prof[prof["n"] >= int(min_n)].copy()
        if prof.shape[0] < 3:
            raise RuntimeError("세그 개수가 너무 적음. Min n 낮추거나 seg 구성 바꿔.")

        return prof, feat_cols

    def _run_demand_space(self):
        try:
            self._ensure_df()

            seg_mode = self.cmb_demand_mode.currentText().startswith("Segments-as-points")
            mode = self.cmb_demand_coord.currentText()
            k = int(self.spin_demand_k.value())

            if seg_mode:
                seg_cols = self._selected_checked_items(self.lst_demand_segcols)
                if len(seg_cols) < 1:
                    raise RuntimeError("Segments-as-points 모드에서는 *_seg 컬럼 1개 이상 체크해.")
                sep = self.txt_demand_seg_sep.text().strip() or "|"
                use_pca = bool(self.chk_demand_use_pca12.isChecked())
                pca_k = int(self.spin_demand_pca_k.value())
                target = self.cmb_demand_target.currentText().strip()
                if target == "":
                    target = "(없음)"
                min_n = int(self.spin_demand_min_n.value())

                prof, feat_cols = self._build_segment_profiles(seg_cols, sep, use_pca, pca_k, target, min_n)
                X = prof[feat_cols].copy()
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.fillna(X.mean())
                Xz = (X - X.mean()) / (X.std() + 1e-9)

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

                xy_df = pd.DataFrame({
                    "id": ids,
                    "label": labels,
                    "x": xy[:, 0],
                    "y": xy[:, 1],
                    "n": prof["n"].values
                })
                cl_s = pd.Series(cl, index=ids)

                self.state.demand_mode = "Segments-as-points"
                self.state.demand_xy = xy_df
                self.state.cluster_assign = cl_s
                self.state.cluster_names = {i + 1: f"Cluster {i + 1}" for i in range(k)}
                self.state.demand_seg_profile = prof
                self.state.demand_seg_components = seg_cols
                self.state.demand_features_used = feat_cols
                self.state.manual_dirty = False

                # Update both plots
                args = (ids, labels, xy, cl, self.state.cluster_names)
                self.plot_preview.set_data(*args)
                self.plot_edit.set_data(*args)
                self._update_cluster_summary()

                self.lbl_demand_status.setText(
                    f"Done: {coord_name}, segments={len(ids)}, k={k}. "
                    f"(Shift+Click 다중 선택, 점 drag=클러스터 이동/좌표 이동, 라벨 drag=병합, Shift+라벨=이동)"
                )
                self._set_status("Demand Space 실행 완료. 'Segmentation editing' 탭에서 수정하세요.")

            else:
                cols = self._selected_checked_items(self.lst_demand_vars)
                if len(cols) < 3:
                    raise RuntimeError("Variables-as-points 모드에서는 변수를 3개 이상 체크해.")

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
                self.state.demand_seg_components = None
                self.state.demand_features_used = None
                self.state.manual_dirty = False

                # Update both plots
                args = (ids, labels, xy, cl, self.state.cluster_names)
                self.plot_preview.set_data(*args)
                self.plot_edit.set_data(*args)
                self._update_cluster_summary()

                self.lbl_demand_status.setText(
                    f"Done: {coord_name}, points={len(ids)}, k={k}. "
                    f"(Shift+Click 다중 선택, 점 drag=클러스터 이동/좌표 이동, 라벨 drag=병합, Shift+라벨=이동)"
                )
                self._set_status("Demand Space ready (Variables-as-points).")

        except Exception as e:
            show_error(self, "Demand Space error", e)

    def _update_cluster_summary(self):
        if self.state.demand_xy is None or self.state.cluster_assign is None:
            self.tbl_cluster_summary.set_df(None)
            return

        cl = self.state.cluster_assign.copy()
        names = self.state.cluster_names or {}

        rows = []
        n_map = {}
        if self.state.demand_mode == "Segments-as-points" and "n" in self.state.demand_xy.columns:
            n_map = dict(zip(self.state.demand_xy["id"], self.state.demand_xy["n"]))

        for cid in sorted(cl.unique()):
            items = cl[cl == cid].index.tolist()
            n_sum = None
            if n_map:
                n_sum = int(sum(int(n_map.get(x, 0)) for x in items))
            rows.append({
                "cluster_id": int(cid),
                "cluster_name": names.get(int(cid), f"Cluster {int(cid)}"),
                "n_points": len(items),
                "n_sum(segments)" if n_map else "": n_sum if n_map else "",
                "items": ", ".join(items)
            })
        out = pd.DataFrame(rows).sort_values(["cluster_id"]).reset_index(drop=True)
        self.tbl_cluster_summary.set_df(out, max_rows=500)

    def _on_manual_clusters_changed(self):
        if self.state.demand_xy is None:
            return
        s = self.plot_edit.get_cluster_series()
        self.state.cluster_assign = s
        self.state.manual_dirty = True
        self._update_cluster_summary()
        self._set_status("Manual clusters updated.")

    def _on_manual_coords_changed(self):
        if self.state.demand_xy is None:
            return
        xy_map = self.plot_edit.get_xy_map()
        try:
            df = self.state.demand_xy.copy()
            if "id" in df.columns and "x" in df.columns and "y" in df.columns:
                df["x"] = df["id"].astype(str).map(lambda k: xy_map.get(str(k), (np.nan, np.nan))[0])
                df["y"] = df["id"].astype(str).map(lambda k: xy_map.get(str(k), (np.nan, np.nan))[1])
                self.state.demand_xy = df
                self.state.manual_dirty = True
                self._set_status("Manual coords updated.")
        except Exception:
            # 좌표 업데이트 실패해도 앱은 계속
            self._set_status("Manual coords updated (partial).")

    def _rename_cluster(self):
        try:
            cid = int(self.spin_rename_cluster_id.value())
            name = self.txt_rename_cluster.text().strip()
            if not name:
                raise RuntimeError("클러스터 이름 입력해.")
            self.state.cluster_names[cid] = name
            
            # Update both plots
            self.plot_edit.set_cluster_names(self.state.cluster_names)
            self.plot_preview.set_cluster_names(self.state.cluster_names)
            
            self._update_cluster_summary()
            self.state.manual_dirty = True
        except Exception as e:
            show_error(self, "Rename error", e)

    # 9. Export
    def _build_tab_export(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Export")

        layout = QtWidgets.QVBoxLayout(tab)

        self.lbl_export = QtWidgets.QLabel(
            "Export to Excel (고정 sheet 이름):\n"
            "01_Data, 02_RECODE, 03_PCA_Loadings, 04_PCA_Scores,\n"
            "05_DT_ImprovePivot, 06_DT_BestSplit, 07_DT_Full_Nodes, 08_DT_Full_SplitGroups,\n"
            "09_DT_Full_Paths, 10_DT_Full_CondFreq, 11_HCLUST,\n"
            "12_Demand_Coords, 13_Demand_Clusters, 14_Demand_Summary, 15_Demand_SegProfile"
        )
        self.lbl_export.setWordWrap(True)
        layout.addWidget(self.lbl_export)

        row = QtWidgets.QHBoxLayout()
        self.txt_export_path = QtWidgets.QLineEdit()
        self.btn_export_browse = QtWidgets.QPushButton("경로 선택…")
        style_button(self.btn_export_browse, level=1)
        self.btn_export_browse.clicked.connect(self._browse_export_path)
        self.btn_export = QtWidgets.QPushButton("엑셀로 내보내기")
        style_button(self.btn_export, level=2)
        self.btn_export.clicked.connect(self._export_excel)

        row.addWidget(QtWidgets.QLabel("Output xlsx"))
        row.addWidget(self.txt_export_path, 3)
        row.addWidget(self.btn_export_browse)
        row.addWidget(self.btn_export)
        layout.addLayout(row)

        self.lbl_export_status = QtWidgets.QLabel("")
        self.lbl_export_status.setWordWrap(True)
        layout.addWidget(self.lbl_export_status)

    def _browse_export_path(self):
        default = "analysis_output.xlsx"
        if self.state.path:
            base = os.path.splitext(os.path.basename(self.state.path))[0]
            default = f"{base}_AutoSegmentTool.xlsx"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "엑셀 파일 저장", default, "Excel (*.xlsx)"
        )
        if path:
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            self.txt_export_path.setText(path)

    def _export_excel(self):
        try:
            self._ensure_df()
            out = self.txt_export_path.text().strip()
            if not out:
                raise RuntimeError("저장 경로 지정해.")
            if not out.lower().endswith(".xlsx"):
                out += ".xlsx"

            with pd.ExcelWriter(out, engine="openpyxl") as w:
                self.state.df.to_excel(w, sheet_name="01_Data", index=False)

                if self.state.recode_df is not None:
                    # ✅ QUESTION/CODE/NAME 기준 시트 저장
                    self.state.recode_df[["QUESTION", "CODE", "NAME"]].to_excel(w, sheet_name="02_RECODE", index=False)

                if self.state.pca_loadings is not None:
                    self.state.pca_loadings.reset_index().rename(columns={"index": "variable"}).to_excel(
                        w, sheet_name="03_PCA_Loadings", index=False
                    )

                if self.state.pca_scores is not None:
                    self.state.pca_scores.reset_index().rename(columns={"index": "row_index"}).to_excel(
                        w, sheet_name="04_PCA_Scores", index=False
                    )

                if self.state.dt_improve_pivot is not None:
                    self.state.dt_improve_pivot.to_excel(w, sheet_name="05_DT_ImprovePivot", index=False)

                if self.state.dt_split_best is not None:
                    self.state.dt_split_best.to_excel(w, sheet_name="06_DT_BestSplit", index=False)

                if self.state.dt_full_nodes is not None and not self.state.dt_full_nodes.empty:
                    self.state.dt_full_nodes.to_excel(w, sheet_name="07_DT_Full_Nodes", index=False)

                if self.state.dt_full_split_groups is not None and not self.state.dt_full_split_groups.empty:
                    self.state.dt_full_split_groups.to_excel(w, sheet_name="08_DT_Full_SplitGroups", index=False)

                if self.state.dt_full_path_info is not None and not self.state.dt_full_path_info.empty:
                    self.state.dt_full_path_info.to_excel(w, sheet_name="09_DT_Full_Paths", index=False)

                if self.state.dt_full_condition_freq is not None and not self.state.dt_full_condition_freq.empty:
                    self.state.dt_full_condition_freq.to_excel(w, sheet_name="10_DT_Full_CondFreq", index=False)

                if self.state.hclust_assign is not None:
                    self.state.hclust_assign.to_excel(w, sheet_name="11_HCLUST", index=False)

                if self.state.demand_xy is not None:
                    self.state.demand_xy.to_excel(w, sheet_name="12_Demand_Coords", index=False)

                if self.state.cluster_assign is not None:
                    cl_df = self.state.cluster_assign.reset_index()
                    cl_df.columns = ["id", "cluster_id"]
                    cl_df["cluster_name"] = cl_df["cluster_id"].map(self.state.cluster_names).fillna("")
                    cl_df["mode"] = self.state.demand_mode
                    cl_df.to_excel(w, sheet_name="13_Demand_Clusters", index=False)

                if self.state.cluster_assign is not None:
                    cl = self.state.cluster_assign.copy()
                    names = self.state.cluster_names or {}
                    n_map = {}
                    if self.state.demand_xy is not None and "n" in self.state.demand_xy.columns:
                        n_map = dict(zip(self.state.demand_xy["id"], self.state.demand_xy["n"]))

                    rows = []
                    for cid in sorted(cl.unique()):
                        items = cl[cl == cid].index.tolist()
                        n_sum = int(sum(int(n_map.get(x, 0)) for x in items)) if n_map else ""
                        rows.append({
                            "cluster_id": int(cid),
                            "cluster_name": names.get(int(cid), f"Cluster {int(cid)}"),
                            "n_points": len(items),
                            "n_sum(segments)" if n_map else "": n_sum,
                            "items": ", ".join(items)
                        })
                    pd.DataFrame(rows).to_excel(w, sheet_name="14_Demand_Summary", index=False)

                if self.state.demand_seg_profile is not None and not self.state.demand_seg_profile.empty:
                    self.state.demand_seg_profile.reset_index().rename(columns={"_SEG_LABEL_": "segment"}).to_excel(
                        w, sheet_name="15_Demand_SegProfile", index=False
                    )

            self.lbl_export_status.setText(f"✅ Exported: {out}\n(manual changes included: {self.state.manual_dirty})")
            self._set_status("Export done.")
        except Exception as e:
            show_error(self, "Export error", e)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = IntegratedApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
