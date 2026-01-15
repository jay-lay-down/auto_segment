# =============================================================================
# app.py (Part 1/8)
# Auto Segment Tool v8.1 - Imports, Helpers, AppState
# =============================================================================
# -*- coding: utf-8 -*-

"""
Auto Segment Tool v8.1 (Enhanced Features)

[v8.0 Features Preserved]
- Smart Data Cleaning Wizard
- Project Save/Load
- AI Factor Naming
- Visual Decision Tree
- Segment Profiler

[v8.1 New Features]
1. Variable Type Manager: SPSS-style Numeric/Categorical designation
2. API Rate Limit Handling: Retry logic with exponential backoff
3. Enhanced Demand Space Explanations
"""

from __future__ import annotations

import os
import sys
import json
import traceback
import pickle
import math
import ast
import time  # Added for retry delays
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import requests
from scipy.stats import zscore
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------
logger = logging.getLogger(__name__)

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


def resource_path(rel_path: str) -> str:
    base = getattr(sys, "_MEIPASS", str(Path(__file__).resolve().parent))
    return str(Path(base) / rel_path)


def is_categorical_series(s: pd.Series, max_unique_numeric_as_cat: int = 20) -> bool:
    """Auto-detect if series is categorical based on dtype and unique count."""
    if s is None:
        return False
    if pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
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
            if n in col_lut:
                return col_lut[n]
        return None

    q = pick("question", "문항", "문항명", "q", "item", "variable")
    qk = pick("question_kr", "questionkr", "문항(한글)", "문항한글", "question (kr)")
    c = pick("code", "코드", "값", "value", "val")
    n = pick("name", "라벨", "label", "명", "설명", "text")
    nk = pick("name_kr", "label_kr", "라벨한글", "name (kr)")

    if q is None or c is None or n is None:
        if len(cols) >= 3:
            q, c, n = cols[0], cols[1], cols[2]

    out = df.copy()
    rename_map = {}
    if q:
        rename_map[q] = "QUESTION"
    if qk:
        rename_map[qk] = "QUESTION_KR"
    if c:
        rename_map[c] = "CODE"
    if n:
        rename_map[n] = "NAME"
    if nk:
        rename_map[nk] = "NAME_KR"
    out = out.rename(columns=rename_map)

    for cc in ["QUESTION", "CODE", "NAME"]:
        if cc not in out.columns:
            out[cc] = np.nan

    out["QUESTION"] = out["QUESTION"].astype(str).str.strip()
    out["CODE"] = out["CODE"].astype(str).str.strip()
    out["NAME"] = out["NAME"].astype(str).str.strip()
    if "QUESTION_KR" in out.columns:
        out["QUESTION_KR"] = out["QUESTION_KR"].astype(str).str.strip()
    if "NAME_KR" in out.columns:
        out["NAME_KR"] = out["NAME_KR"].astype(str).str.strip()
    return out[["QUESTION", "CODE", "NAME"] + [c for c in out.columns if c not in ["QUESTION", "CODE", "NAME"]]]


# -----------------------------------------------------------------------------
# [v8.1 NEW] Variable Type Constants
# -----------------------------------------------------------------------------
VAR_TYPE_AUTO = "Auto"
VAR_TYPE_NUMERIC = "Numeric"
VAR_TYPE_CATEGORICAL = "Categorical"


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
    recode_sources: List[str] = field(default_factory=list)
    recode_label_mode: str = "original"  # "original" or "korean"

    # [v8.1 NEW] Variable Type Overrides: {column_name: VAR_TYPE_NUMERIC | VAR_TYPE_CATEGORICAL | VAR_TYPE_AUTO}
    var_types: Dict[str, str] = field(default_factory=dict)

    # Factor Analysis Data (PCA or EFA)
    factor_model: Any = None
    factor_cols: Optional[List[str]] = None
    factor_scores: Optional[pd.DataFrame] = None
    factor_score_cols: List[str] = field(default_factory=list)
    factor_loadings: Optional[pd.DataFrame] = None
    factor_loadings_order: Optional[List[str]] = None
    ui_lang: str = "ko"
    factor_mode: str = "PCA"
    factor_ai_names: Dict[str, str] = field(default_factory=dict)
    factor_ai_suggestions: Dict[str, str] = field(default_factory=dict)

    # Decision tree outputs (Setting Tab)
    dt_improve_pivot: Optional[pd.DataFrame] = None
    dt_split_best: Optional[pd.DataFrame] = None
    dt_importance_summary: Optional[pd.DataFrame] = None

    # Decision tree full Analysis (Results Tab)
    dt_full_nodes: Optional[pd.DataFrame] = None
    dt_full_split_groups: Optional[pd.DataFrame] = None
    dt_full_split_branches: Optional[pd.DataFrame] = None
    dt_full_path_info: Optional[pd.DataFrame] = None
    dt_full_split_paths: Optional[pd.DataFrame] = None
    dt_full_condition_freq: Optional[pd.DataFrame] = None
    dt_full_selected: Tuple[Optional[str], Optional[str]] = (None, None)
    dt_full_split_view: Optional[pd.DataFrame] = None  # formatted split summary for the All Splits table
    dt_full_split_pivot: Optional[pd.DataFrame] = None  # pivot: rows=split condition, cols=dep, val=improve
    dt_selected_deps: Optional[List[str]] = None
    dt_selected_inds: Optional[List[str]] = None
    dt_edit_group_map: Dict[str, Dict[str, str]] = field(default_factory=dict)
    dt_edit_view_mode: str = "split"  # "split" (좌/우+개선도) or "combo" (모든 조합 개선도)

    # Demand Space Data
    demand_mode: str = "Segments-as-points"
    demand_xy: Optional[pd.DataFrame] = None
    cluster_assign: Optional[pd.Series] = None
    cluster_names: Dict[int, str] = field(default_factory=dict)
    cluster_colors: Dict[int, str] = field(default_factory=dict)

    # Demand Space Profile Data
    demand_seg_profile: Optional[pd.DataFrame] = None
    demand_seg_components: Optional[List[str]] = None
    demand_targets: Optional[List[str]] = None
    demand_targets_used: Optional[List[str]] = None
    demand_features_used: Optional[List[str]] = None
    demand_seg_labels: Optional[pd.Series] = None
    demand_seg_sep: str = "|"
    demand_seg_cluster_map: Dict[str, int] = field(default_factory=dict)

    manual_dirty: bool = False
    label_pos_override: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    # System Logs (for RAG)
    last_error: str = ""

    def get_var_type(self, col: str) -> str:
        """Returns the user-specified variable type, or 'Auto' if not set."""
        return self.var_types.get(col, VAR_TYPE_AUTO)

    def set_var_type(self, col: str, vtype: str):
        """Sets the variable type for a column."""
        if vtype == VAR_TYPE_AUTO and col in self.var_types:
            del self.var_types[col]
        else:
            self.var_types[col] = vtype

    def is_categorical(self, col: str, series: Optional[pd.Series] = None) -> bool:
        """
        Determines if a column should be treated as categorical.
        Uses user override if set, otherwise auto-detects.
        """
        vtype = self.get_var_type(col)
        if vtype == VAR_TYPE_CATEGORICAL:
            return True
        if vtype == VAR_TYPE_NUMERIC:
            return False
        # Auto mode: use auto-detection
        if series is not None:
            return is_categorical_series(series)
        if self.df is not None and col in self.df.columns:
            return is_categorical_series(self.df[col])
        return False


# -----------------------------------------------------------------------------
# [v8.1 NEW] API Call Helper with Retry Logic
# -----------------------------------------------------------------------------
def _flatten_messages(messages: List[dict]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def call_openai_api(
    api_key: str,
    messages: List[dict],
    model: str = "gpt-4o-mini",
    max_retries: int = 4,
    initial_delay: float = 2.0,
    timeout: int = 30,
) -> Tuple[bool, str]:
    """
    Calls OpenAI API with exponential backoff retry logic and clearer
    rate-limit handling.

    Returns:
        (success: bool, result: str)
        - If success: result is the AI response content
        - If failure: result is the error message
    """
    if not api_key or not str(api_key).strip():
        return False, "API key is missing. Please enter a valid key."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, "temperature": 0.7}

    delay = float(initial_delay)
    last_error = ""
    suggested_wait = 0.0

    for attempt in range(max_retries):
        resp: Optional[requests.Response] = None
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # Check for rate limit (429)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                # Some gateways return reset hints
                reset_hint = resp.headers.get("x-ratelimit-reset-requests")
                wait_time = delay
                for candidate in [retry_after, reset_hint]:
                    try:
                        if candidate:
                            wait_time = max(wait_time, float(candidate))
                    except Exception:
                        pass

                suggested_wait = max(suggested_wait, wait_time)
                last_error = (
                    f"Rate limit hit. Waiting {wait_time:.1f}s before retry "
                    f"({attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
                delay = min(delay * 2, 60)
                continue

            resp.raise_for_status()
            try:
                data = resp.json()
                answer = data["choices"][0]["message"]["content"]
            except (ValueError, KeyError, TypeError) as parse_err:
                return False, f"Invalid response from server: {parse_err}"
            return True, answer

        except requests.exceptions.Timeout:
            last_error = (
                f"Request timeout after {timeout}s (Attempt {attempt + 1}/{max_retries})"
            )
            suggested_wait = max(suggested_wait, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60)

        except requests.exceptions.HTTPError as e:
            # Use server-provided message if available
            if resp is not None:
                if resp.status_code == 401:
                    return False, "Invalid API key or unauthorized. Please check your key."
                if resp.status_code == 429:
                    suggested_wait = max(suggested_wait, delay)
                    last_error = "Rate limit exceeded. Please wait and try again."
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                try:
                    err_json = resp.json()
                    msg = err_json.get("error", {}).get("message")
                    if msg:
                        return False, msg
                except Exception:
                    pass

            if "429" in str(e):
                last_error = "Rate limit exceeded. Please wait and try again."
                suggested_wait = max(suggested_wait, delay)
                time.sleep(delay)
                delay = min(delay * 2, 60)
            else:
                return False, f"HTTP Error: {str(e)}"

        except requests.exceptions.RequestException as e:
            return False, f"Network Error: {str(e)}"

        except Exception as e:
            return False, f"Unexpected Error: {str(e)}"

    if suggested_wait > 0:
        return False, (
            f"Rate limit hit repeatedly. Please wait about {suggested_wait:.0f}s "
            "and try again."
        )

    return False, f"Failed after {max_retries} attempts. Last error: {last_error}"


def _normalize_gemini_model(model: Optional[str]) -> str:
    """Ensure Gemini model names align with deployable versions."""
    name = (model or "").strip()
    if not name:
        return "gemini-3-pro-preview"

    name = name.lower()
    if name.startswith("models/"):
        name = name.split("/", 1)[1]

    alias = {
        "gemini": "gemini-3-pro-preview",
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-1.5-flash-latest": "gemini-1.5-flash",
        "gemini-1.5-flash-001": "gemini-1.5-flash-001",
        "gemini-1.5-pro": "gemini-1.5-pro",
        "gemini-1.5-pro-001": "gemini-1.5-pro-001",
        "gemini-1.5-flash-8b": "gemini-1.5-flash-8b",
        "gemini-3-pro": "gemini-3-pro-preview",
        "gemini-3-pro-preview": "gemini-3-pro-preview",
        "gemini-3.5-pro": "gemini-3.5-pro-preview",
        "gemini-3.5-pro-preview": "gemini-3.5-pro-preview",
        "gemini-3.5-pro-preview-0409": "gemini-3.5-pro-preview-0409",
        "gemini (1.5-flash)": "gemini-1.5-flash",
        "gemini (1.5-flash-latest)": "gemini-1.5-flash",
        "gemini (3-pro-preview)": "gemini-3-pro-preview",
        "gemini (3.5-pro-preview)": "gemini-3.5-pro-preview",
        "gemini (3.5-pro-preview-0409)": "gemini-3.5-pro-preview-0409",
    }

    return alias.get(name, name)


def call_gemini_api(
    api_key: str,
    messages: List[dict],
    model: str = "gemini-3-pro-preview",
    max_retries: int = 4,
    initial_delay: float = 2.0,
    timeout: int = 30,
) -> Tuple[bool, str]:
    """Calls Gemini Generative Language API with retry logic similar to OpenAI."""
    if not api_key or not str(api_key).strip():
        return False, "API key is missing. Please enter a valid key."

    model = _normalize_gemini_model(model)
    fallback_by_model = {
        "gemini-3-pro-preview": [
            "gemini-3.5-pro-preview",
            "gemini-3.5-pro-preview-0409",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
        "gemini-3.5-pro-preview": [
            "gemini-3-pro-preview",
            "gemini-3.5-pro-preview-0409",
            "gemini-1.5-pro",
        ],
        "gemini-3.5-pro-preview-0409": [
            "gemini-3.5-pro-preview",
            "gemini-3-pro-preview",
            "gemini-1.5-pro",
        ],
        "gemini-1.5-flash": ["gemini-1.5-flash-001", "gemini-1.5-pro"],
        "gemini-1.5-flash-latest": ["gemini-1.5-flash", "gemini-1.5-flash-001"],
        "gemini-1.5-flash-001": ["gemini-1.5-flash", "gemini-1.5-pro"],
        "gemini-1.5-flash-8b": ["gemini-1.5-flash", "gemini-1.5-flash-001"],
        "gemini-1.5-pro": ["gemini-1.5-pro-001", "gemini-1.5-flash"],
        "gemini-1.5-pro-001": ["gemini-1.5-pro", "gemini-1.5-flash"],
    }
    candidate_models: List[str] = [model]
    for alt in fallback_by_model.get(model, ["gemini-3-pro-preview", "gemini-1.5-flash"]):
        if alt not in candidate_models:
            candidate_models.append(alt)

    model_idx = 0
    delay = float(initial_delay)
    last_error = ""

    for attempt in range(max_retries):
        current_model = candidate_models[min(model_idx, len(candidate_models) - 1)]
        path_model = (
            current_model if current_model.startswith("models/") else f"models/{current_model}"
        )
        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/{path_model}:generateContent"
                f"?key={api_key}"
            )
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": _flatten_messages(messages)}],
                    }
                ]
            }
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )

            if resp.status_code == 429:
                last_error = (
                    f"Rate limit hit. Waiting {delay:.1f}s before retry "
                    f"({attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue

            resp.raise_for_status()
            try:
                data = resp.json()
                candidates = data.get("candidates") or []
                first = candidates[0]
                parts = first.get("content", {}).get("parts", [])
                text = parts[0].get("text") if parts else None
                if not text:
                    raise KeyError("No text in Gemini response")
            except (ValueError, KeyError, IndexError, AttributeError) as parse_err:
                return False, f"Invalid response from Gemini: {parse_err}"
            return True, text

        except requests.exceptions.HTTPError as http_err:
            status_code = getattr(getattr(http_err, "response", None), "status_code", None)
            if status_code == 404:
                last_error = (
                    f"Model not found (404) for '{current_model}'. "
                    "Trying alternate Gemini model names."
                )
                if model_idx + 1 < len(candidate_models):
                    model_idx += 1
                    continue
                tried = ", ".join(candidate_models)
                recommended = ", ".join(
                    [
                        "gemini-3-pro-preview",
                        "gemini-3.5-pro-preview",
                        "gemini-3.5-pro-preview-0409",
                        "gemini-1.5-pro",
                        "gemini-1.5-flash",
                    ]
                )
                msg = (
                    "Model not found (404). Please verify the Gemini model name "
                    f"and access. Tried: {tried}. Recommended models: {recommended}."
                )
            else:
                msg = f"HTTP error: {http_err}"
            return False, msg
        except requests.exceptions.Timeout:
            last_error = (
                f"Request timeout after {timeout}s (Attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(delay)
            delay = min(delay * 2, 60)
        except requests.exceptions.RequestException as e:
            last_error = f"Network error: {str(e)}"
            time.sleep(delay)
            delay = min(delay * 2, 60)

    msg = f"Failed after {max_retries} attempts. Last error: {last_error}"
    return False, msg


def call_ai_chat(
    provider: str,
    api_key: str,
    messages: List[dict],
    model: Optional[str] = None,
    max_retries: int = 4,
    initial_delay: float = 2.0,
    timeout: int = 30,
) -> Tuple[bool, str]:
    provider = (provider or "openai").lower()
    if provider == "gemini":
        model = _normalize_gemini_model(model or "gemini-3-pro-preview")
        return call_gemini_api(
            api_key,
            messages,
            model=model,
            max_retries=max_retries,
            initial_delay=initial_delay,
            timeout=timeout,
        )

    model = model or "gpt-4o-mini"
    return call_openai_api(
        api_key,
        messages,
        model=model,
        max_retries=max_retries,
        initial_delay=initial_delay,
        timeout=timeout,
    )

# =============================================================================
# app.py (Part 2/8)
# Variable Type Manager Dialog & DataFrameTable
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


# -----------------------------------------------------------------------------
# [v8.1 NEW] Variable Type Manager Dialog (SPSS-Style)
# -----------------------------------------------------------------------------
class VariableTypeManagerDialog(QtWidgets.QDialog):
    """
    Dialog to manage variable types (Numeric/Categorical) for each column.
    Similar to SPSS Variable View.
    """
    def __init__(self, df: pd.DataFrame, var_types: Dict[str, str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Variable Type Manager")
        self.resize(900, 600)
        self.df = df
        self.var_types = var_types.copy()  # Work on a copy
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Instructions
        info = QtWidgets.QLabel(
            "<b>Variable Type Settings</b><br>"
            "Set each variable as <b>Numeric</b> (continuous) or <b>Categorical</b> (discrete/nominal).<br>"
            "• <b>Auto</b>: System auto-detects based on dtype and unique values<br>"
            "• <b>Numeric</b>: Treated as continuous (mean, regression splits)<br>"
            "• <b>Categorical</b>: Treated as discrete (mode, subset splits)<br><br>"
            "<i>This affects Factor Analysis (numeric only) and Decision Tree split logic.</i>"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Filter row
        filter_row = QtWidgets.QHBoxLayout()
        self.txt_filter = QtWidgets.QLineEdit()
        self.txt_filter.setPlaceholderText("Filter variables by name...")
        self.txt_filter.textChanged.connect(self._apply_filter)
        
        self.btn_all_numeric = QtWidgets.QPushButton("Set All → Numeric")
        self.btn_all_categorical = QtWidgets.QPushButton("Set All → Categorical")
        self.btn_all_auto = QtWidgets.QPushButton("Set All → Auto")
        
        style_button(self.btn_all_numeric, 1)
        style_button(self.btn_all_categorical, 1)
        style_button(self.btn_all_auto, 1)
        
        self.btn_all_numeric.clicked.connect(lambda: self._set_all_types(VAR_TYPE_NUMERIC))
        self.btn_all_categorical.clicked.connect(lambda: self._set_all_types(VAR_TYPE_CATEGORICAL))
        self.btn_all_auto.clicked.connect(lambda: self._set_all_types(VAR_TYPE_AUTO))
        
        filter_row.addWidget(QtWidgets.QLabel("Filter:"))
        filter_row.addWidget(self.txt_filter, 2)
        filter_row.addWidget(self.btn_all_numeric)
        filter_row.addWidget(self.btn_all_categorical)
        filter_row.addWidget(self.btn_all_auto)
        layout.addLayout(filter_row)

        # Main Table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Variable Name", "Current Type", "Auto-Detected", 
            "Dtype", "Unique Values", "Sample Values"
        ])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.Stretch)
        
        self.table.setColumnWidth(1, 130)
        self.table.setColumnWidth(2, 100)
        self.table.setColumnWidth(3, 100)
        self.table.setColumnWidth(4, 90)
        
        layout.addWidget(self.table, 1)

        # Batch operations for selected
        batch_row = QtWidgets.QHBoxLayout()
        self.btn_sel_numeric = QtWidgets.QPushButton("Selected → Numeric")
        self.btn_sel_categorical = QtWidgets.QPushButton("Selected → Categorical")
        self.btn_sel_auto = QtWidgets.QPushButton("Selected → Auto")
        
        style_button(self.btn_sel_numeric, 2)
        style_button(self.btn_sel_categorical, 2)
        style_button(self.btn_sel_auto, 2)
        
        self.btn_sel_numeric.clicked.connect(lambda: self._set_selected_types(VAR_TYPE_NUMERIC))
        self.btn_sel_categorical.clicked.connect(lambda: self._set_selected_types(VAR_TYPE_CATEGORICAL))
        self.btn_sel_auto.clicked.connect(lambda: self._set_selected_types(VAR_TYPE_AUTO))
        
        batch_row.addWidget(QtWidgets.QLabel("Batch (Selected Rows):"))
        batch_row.addWidget(self.btn_sel_numeric)
        batch_row.addWidget(self.btn_sel_categorical)
        batch_row.addWidget(self.btn_sel_auto)
        batch_row.addStretch(1)
        layout.addLayout(batch_row)

        # Dialog buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Populate table
        self._populate_table()

    def _populate_table(self):
        """Fill table with variable information."""
        cols = list(self.df.columns)
        self.table.setRowCount(len(cols))
        
        self._combo_widgets = {}  # Store combo references

        for i, col in enumerate(cols):
            series = self.df[col]
            
            # Column 0: Variable Name
            name_item = QtWidgets.QTableWidgetItem(str(col))
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 0, name_item)
            
            # Column 1: Current Type (ComboBox)
            combo = QtWidgets.QComboBox()
            combo.addItems([VAR_TYPE_AUTO, VAR_TYPE_NUMERIC, VAR_TYPE_CATEGORICAL])
            current_type = self.var_types.get(col, VAR_TYPE_AUTO)
            combo.setCurrentText(current_type)
            combo.currentTextChanged.connect(lambda text, c=col: self._on_type_changed(c, text))
            self.table.setCellWidget(i, 1, combo)
            self._combo_widgets[col] = combo
            
            # Column 2: Auto-Detected Type
            auto_type = "Categorical" if is_categorical_series(series) else "Numeric"
            auto_item = QtWidgets.QTableWidgetItem(auto_type)
            auto_item.setFlags(auto_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            if auto_type == "Categorical":
                auto_item.setBackground(QtGui.QColor("#fff3e0"))
            else:
                auto_item.setBackground(QtGui.QColor("#e3f2fd"))
            self.table.setItem(i, 2, auto_item)
            
            # Column 3: Dtype
            dtype_item = QtWidgets.QTableWidgetItem(str(series.dtype))
            dtype_item.setFlags(dtype_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 3, dtype_item)
            
            # Column 4: Unique Values
            nunique = series.nunique()
            unique_item = QtWidgets.QTableWidgetItem(f"{nunique:,}")
            unique_item.setFlags(unique_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 4, unique_item)
            
            # Column 5: Sample Values
            samples = series.dropna().head(5).astype(str).tolist()
            sample_txt = ", ".join(samples)
            if len(sample_txt) > 60:
                sample_txt = sample_txt[:60] + "..."
            sample_item = QtWidgets.QTableWidgetItem(sample_txt)
            sample_item.setFlags(sample_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 5, sample_item)

    def _on_type_changed(self, col: str, new_type: str):
        """Handle combo box change."""
        if new_type == VAR_TYPE_AUTO:
            if col in self.var_types:
                del self.var_types[col]
        else:
            self.var_types[col] = new_type

    def _apply_filter(self):
        """Filter table rows by variable name."""
        term = self.txt_filter.text().strip().lower()
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item:
                visible = term in item.text().lower() if term else True
                self.table.setRowHidden(i, not visible)

    def _set_all_types(self, vtype: str):
        """Set all variables to a specific type."""
        for col, combo in self._combo_widgets.items():
            combo.setCurrentText(vtype)

    def _set_selected_types(self, vtype: str):
        """Set selected variables to a specific type."""
        selected_rows = set()
        for item in self.table.selectedItems():
            selected_rows.add(item.row())
        
        cols = list(self.df.columns)
        for row in selected_rows:
            if row < len(cols):
                col = cols[row]
                if col in self._combo_widgets:
                    self._combo_widgets[col].setCurrentText(vtype)

    def get_var_types(self) -> Dict[str, str]:
        """Return the final variable type settings."""
        return self.var_types.copy()


# -----------------------------------------------------------------------------
# [v8.0] Smart Data Cleaning Wizard
# -----------------------------------------------------------------------------
class SmartCleanerDialog(QtWidgets.QDialog):
    """
    Wizard to diagnose and clean data upon loading.
    Handles NaN values and Duplicates.
    """
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Smart Data Cleaning Wizard")
        self.resize(500, 400)
        self.df_in = df
        self.df_out = df.copy()
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        n_rows = len(self.df_in)
        n_dupes = self.df_in.duplicated().sum()
        n_nans = self.df_in.isna().sum().sum()
        cols_with_nan = self.df_in.columns[self.df_in.isna().any()].tolist()

        info_txt = f"<b>Data Diagnosis Report:</b><br>"
        info_txt += f"Total Rows: {n_rows:,}<br>"
        info_txt += f"Duplicates Found: <font color='red'>{n_dupes:,}</font><br>"
        info_txt += f"Total Missing Values (NaN): <font color='red'>{n_nans:,}</font><br>"
        if cols_with_nan:
            info_txt += f"Columns with Missing Data: {', '.join(cols_with_nan[:5])}"
            if len(cols_with_nan) > 5:
                info_txt += f"... (+{len(cols_with_nan)-5} more)"

        layout.addWidget(QtWidgets.QLabel(info_txt))

        box = QtWidgets.QGroupBox("Recommended Actions")
        bl = QtWidgets.QVBoxLayout(box)

        self.chk_drop_dupes = QtWidgets.QCheckBox(f"Drop Duplicate Rows ({n_dupes})")
        self.chk_drop_dupes.setChecked(bool(n_dupes > 0))
        self.chk_drop_dupes.setEnabled(bool(n_dupes > 0))

        self.chk_fill_numeric = QtWidgets.QCheckBox("Fill Missing Numeric Values (with Mean)")
        self.chk_fill_numeric.setChecked(True)

        self.chk_fill_cat = QtWidgets.QCheckBox("Fill Missing Categorical Values (with 'Unknown')")
        self.chk_fill_cat.setChecked(True)

        bl.addWidget(self.chk_drop_dupes)
        bl.addWidget(self.chk_fill_numeric)
        bl.addWidget(self.chk_fill_cat)
        layout.addWidget(box)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def accept(self):
        df = self.df_out

        if self.chk_drop_dupes.isChecked():
            df = df.drop_duplicates()

        if self.chk_fill_numeric.isChecked():
            num_cols = df.select_dtypes(include=[np.number]).columns
            for c in num_cols:
                if df[c].isna().any():
                    df[c] = df[c].fillna(df[c].mean())

        if self.chk_fill_cat.isChecked():
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            for c in cat_cols:
                if df[c].isna().any():
                    df[c] = df[c].fillna("Unknown")

        self.df_out = df
        super().accept()

# =============================================================================
# app.py (Part 3/8)
# Decision Tree Logic (Impurity, Optimal Subset Split, Recursive Build)
# =============================================================================

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
    max_unique_cat: int = 50,
    force_categorical: Optional[bool] = None  # [v8.1] None=auto, True=force cat, False=force numeric
) -> Tuple[Optional[dict], List[dict]]:
    """
    Finds the single best split for a target (y) and predictor (x).
    
    [v8.1 UPDATE] Added force_categorical parameter to override auto-detection.
    
    [v7.0 Features Preserved]
    - Optimal Subset Split logic for categorical variables.
    - Sorts categories by Target Mean (or Class Prob) and finds best cut.
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

    # [v8.1] Determine if predictor should be treated as categorical
    if force_categorical is None:
        treat_as_categorical = is_categorical_series(xv)
    else:
        treat_as_categorical = force_categorical

    # 2. Case A: Categorical Predictor (OPTIMAL SUBSET SPLIT)
    if treat_as_categorical:
        xv_str = xv.astype(str)
        cats = xv_str.unique()

        if len(cats) < 2 or len(cats) > max_unique_cat:
            return None, []

        # Step 2-1: Calculate sorting metric for each category
        if task == "reg":
            cat_stats = pd.DataFrame({'x': xv_str, 'y': yv})
            agg = cat_stats.groupby('x')['y'].mean().sort_values()
            sorted_cats = agg.index.tolist()
        else:
            u, c = np.unique(yv, return_counts=True)
            target_cls = u[np.argmax(c)]

            cat_stats = pd.DataFrame({'x': xv_str, 'y': yv})
            cat_stats['is_target'] = (cat_stats['y'] == target_cls).astype(int)
            agg = cat_stats.groupby('x')['is_target'].mean().sort_values()
            sorted_cats = agg.index.tolist()

        # Step 2-2: Iterate through split points in the sorted list
        rank_map = {cat: i for i, cat in enumerate(sorted_cats)}
        x_rank = np.array([rank_map[v] for v in xv_str])

        for i in range(len(sorted_cats) - 1):
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

            left_items = sorted_cats[:i + 1]
            right_items = sorted_cats[i + 1:]

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
                "cutpoint": f"Rank {i} (Sorted)",
                "left_group": l_txt,
                "right_group": r_txt,
                "left_items": left_items,
                "right_items": right_items,
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


# -----------------------------------------------------------------------------
# Tree Building Logic (Recursive)
# -----------------------------------------------------------------------------

@dataclass
class UniNode:
    """Represents a node in the Univariate Decision Tree."""
    node_id: int
    depth: int
    n: int
    condition: str
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
    max_unique_cat: int = 50,
    force_categorical: Optional[bool] = None  # [v8.1]
) -> Tuple[Optional[dict], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Helper to find the best split on a subset of data defined by `idx`.
    Returns: (best_split_dict, left_indices, right_indices)
    """
    ys = y.iloc[idx]
    xs = x.iloc[idx]

    best, _rows = univariate_best_split(ys, xs, task=task, max_unique_cat=max_unique_cat, 
                                         force_categorical=force_categorical)
    if best is None:
        return None, None, None

    mask = pd.notna(ys) & pd.notna(xs)
    valid_idx = idx[mask.values]

    xs_v = xs[mask]

    if len(valid_idx) < 5:
        return None, None, None

    if best["split_type"] == "categorical(subset)":
        left_items = best["left_items"]
        left_mask = xs_v.astype(str).isin(set(map(str, left_items))).values
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
    task: str,
    max_depth: int = 30,
    min_leaf: int = 1,
    min_split: int = 2,
    max_unique_cat: int = 50,
    min_improve_rel: float = 0.0,
    force_categorical: Optional[bool] = None  # [v8.1]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds a full univariate decision tree for a single pair of variables (dep, ind).
    Returns tables: nodes, split_groups, branches, path_info, cond_freq.
    
    [v8.1] Added force_categorical parameter.
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
            y, x, idx, task=task, max_unique_cat=max_unique_cat,
            force_categorical=force_categorical
        )

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

        nid = next_id
        next_id += 1

        subset_items = None

        if best["split_type"] == "categorical(subset)":
            l_items = best["left_items"]
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
            left_condition=left_cond, right_condition=right_cond,
            left_id=left_id, right_id=right_id,
            pred=_pred_text(yy, task=task),
            subset_items=subset_items
        ))
        return nid

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

# =============================================================================
# app.py (Part 4/8)
# Visualization Components: DemandClusterPlot & VisualTreeWidget
# =============================================================================

# -----------------------------------------------------------------------------
# Demand Space Interactive Plot Components
# -----------------------------------------------------------------------------

class DraggableClusterLabel(pg.TextItem):
    """
    A text label for a cluster centroid that can be dragged.
    - Dragging은 라벨 위치만 옮깁니다(클러스터 병합 없음).
    - Shift + Drag 역시 라벨 위치만 조정합니다.
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
        if not self.plot.is_edit_mode_active():
            ev.ignore()
            return

        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return

        vb = self.plot.getPlotItem().vb
        pos_view = vb.mapSceneToView(ev.scenePos())
        ev.accept()

        shift = False
        try:
            mods = ev.modifiers()
            shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        except Exception:
            shift = False

        if ev.isFinish():
            self.plot.remember_label_position(self.cluster_id, (float(pos_view.x()), float(pos_view.y())))
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
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                pos = self.mapSceneToView(ev.scenePos())
                self.plot.drag_event((float(pos.x()), float(pos.y())), ev)
                return
            ev.ignore()
        else:
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
    sigSelectionChanged = QtCore.pyqtSignal(object)

    def __init__(self, parent=None, editable: bool = True):
        self._vb = ClusterViewBox(self)
        super().__init__(parent=parent, viewBox=self._vb)
        self._editable_widget = editable
        self._edit_mode_active = False

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
        self._cluster_custom_colors: Dict[int, QtGui.QColor] = {}

        self._scatter = pg.ScatterPlotItem(size=11, pxMode=True)
        self.addItem(self._scatter)

        self._point_text_items: List[pg.TextItem] = []
        self._hull_items: Dict[int, QtWidgets.QGraphicsPathItem] = {}
        self._label_items: Dict[int, DraggableClusterLabel] = {}

        self._selected: set[int] = set()
        self._dragging = False
        self._drag_temp_positions: Optional[np.ndarray] = None
        self._drag_anchor_xy: Optional[Tuple[float, float]] = None
        self._drag_committed: bool = False

        self._free_move_points: bool = False
        self._show_all_point_labels: bool = False

        self._label_pos_override: Dict[int, Tuple[float, float]] = {}

        # New cluster creation template (user-defined name/color via UI)
        self._new_cluster_enabled: bool = False
        self._new_cluster_name: str = ""
        self._new_cluster_color: Optional[QtGui.QColor] = None
        self._new_clusters_created: List[Tuple[int, str, QtGui.QColor]] = []
        self._new_cluster_history: List[Tuple[int, List[int], List[int]]] = []

    def set_edit_mode_active(self, active: bool):
        self._edit_mode_active = active
        self.reset_interaction_state(clear_selection=True)

    def is_edit_mode_active(self) -> bool:
        return self._edit_mode_active and self._editable_widget

    def set_free_move_points(self, on: bool):
        self._free_move_points = bool(on)

    def set_show_all_point_labels(self, on: bool):
        self._show_all_point_labels = bool(on)
        self._draw_scatter()

    def set_new_cluster_mode(self, on: bool):
        self._new_cluster_enabled = bool(on)
        self.reset_interaction_state(clear_selection=True)

    def set_new_cluster_template(self, name: str, color: Optional[str]):
        self._new_cluster_name = name or ""
        if color:
            try:
                self._new_cluster_color = qcolor(str(color))
            except Exception:
                self._new_cluster_color = None
        else:
            self._new_cluster_color = None

    def reset_interaction_state(self, clear_selection: bool = False):
        self._dragging = False
        self._drag_committed = False
        self._drag_temp_positions = None
        self._drag_anchor_xy = None
        if clear_selection:
            self._selected.clear()
            self._draw_scatter()
            self._emit_selection_changed()

    def _emit_selection_changed(self):
        if not self._selected:
            self.sigSelectionChanged.emit(None)
            return
        cluster_ids = {int(self._cluster[i]) for i in self._selected}
        if len(cluster_ids) == 1:
            self.sigSelectionChanged.emit(cluster_ids.pop())
        else:
            self.sigSelectionChanged.emit(None)

    def select_cluster(self, cid: Optional[int]):
        if cid is None:
            self._selected.clear()
            self._draw_scatter()
            self.sigSelectionChanged.emit(None)
            return
        indices = set(np.where(self._cluster == int(cid))[0].tolist())
        self._selected = indices
        self._draw_scatter()
        self.sigSelectionChanged.emit(int(cid))

    def consume_new_clusters(self) -> List[Tuple[int, str, str]]:
        items: List[Tuple[int, str, str]] = []
        for cid, name, col in self._new_clusters_created:
            try:
                items.append((int(cid), name, QtGui.QColor(col).name()))
            except Exception:
                continue
        self._new_clusters_created.clear()
        return items

    def undo_last_new_cluster(self) -> Optional[int]:
        if not self._new_cluster_history:
            return None
        cid, indices, prev_clusters = self._new_cluster_history.pop()
        for idx, prev in zip(indices, prev_clusters):
            self._cluster[idx] = int(prev)
        self._cluster_names.pop(int(cid), None)
        self._cluster_custom_colors.pop(int(cid), None)
        self._new_clusters_created = [
            item for item in self._new_clusters_created if int(item[0]) != int(cid)
        ]
        self.redraw_all()
        self.sigClustersChanged.emit()
        self._emit_selection_changed()
        return int(cid)

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
        if int(cid) in self._cluster_custom_colors:
            c = QtGui.QColor(self._cluster_custom_colors[int(cid)])
            c.setAlpha(int(alpha))
            return c
        base = self._hex[(int(cid) - 1) % len(self._hex)]
        return qcolor(base, alpha=alpha)

    def set_data(
        self,
        ids: List[str],
        labels: List[str],
        xy: np.ndarray,
        clusters: np.ndarray,
        cluster_names: Optional[Dict[int, str]] = None,
        cluster_colors: Optional[Dict[int, str]] = None,
    ):
        self._ids = list(map(str, ids))
        self._labels = list(map(lambda x: "" if x is None else str(x), labels))
        self._xy = np.asarray(xy, dtype=float)
        self._cluster = np.asarray(clusters, dtype=int)
        self._cluster_names = dict(cluster_names or {})
        self._cluster_custom_colors = {}
        for k, v in (cluster_colors or {}).items():
            try:
                self._cluster_custom_colors[int(k)] = qcolor(str(v))
            except Exception:
                continue

        self._selected.clear()
        self._dragging = False
        self._drag_temp_positions = None
        self._drag_anchor_xy = None

        self._label_pos_override.clear()

        self.redraw_all()

        try:
            self.getPlotItem().enableAutoRange()
        except Exception:
            pass

    def get_cluster_series(self) -> pd.Series:
        return pd.Series(self._cluster.copy(), index=self._ids)

    def get_cluster_names(self) -> Dict[int, str]:
        return dict(self._cluster_names)

    def get_cluster_colors(self) -> Dict[int, str]:
        out: Dict[int, str] = {}
        for k, v in self._cluster_custom_colors.items():
            try:
                out[int(k)] = QtGui.QColor(v).name()
            except Exception:
                continue
        return out

    def get_xy_map(self) -> Dict[str, Tuple[float, float]]:
        return {self._ids[i]: (float(self._xy[i, 0]), float(self._xy[i, 1])) for i in range(len(self._ids))}

    def set_cluster_names(self, names: Dict[int, str]):
        self._cluster_names = dict(names or {})
        self._draw_hulls_and_labels()

    def set_cluster_colors(self, colors: Dict[int, str]):
        self._cluster_custom_colors = {}
        for k, v in (colors or {}).items():
            try:
                self._cluster_custom_colors[int(k)] = qcolor(str(v))
            except Exception:
                continue
        self._draw_scatter()
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

        show_idx = list(range(len(self._ids))) if self._show_all_point_labels else sorted(list(self._selected))
        for i in show_idx:
            cid = int(self._cluster[i])
            col = self._cluster_color(cid, alpha=255)
            t = pg.TextItem(text=self._labels[i], anchor=(0, 1))
            t.setColor(col)
            t.setPos(float(xy[i, 0]), float(xy[i, 1]))
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
                self._emit_selection_changed()
            return
        if shift:
            if i in self._selected:
                self._selected.remove(i)
            else:
                self._selected.add(i)
        else:
            self._selected = {i}
        self._draw_scatter()
        self._emit_selection_changed()

    def _cluster_centroids(self) -> Dict[int, Tuple[float, float]]:
        out = {}
        for cid in sorted(set(map(int, self._cluster.tolist()))):
            idx = np.where(self._cluster == cid)[0]
            if len(idx) == 0:
                continue
            pts = self._xy[idx]
            out[cid] = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        return out

    def _cluster_scale(self) -> float:
        if self._xy.shape[0] == 0:
            return 0.0
        dx = float(np.max(self._xy[:, 0]) - np.min(self._xy[:, 0])) if self._xy.size else 0.0
        dy = float(np.max(self._xy[:, 1]) - np.min(self._xy[:, 1])) if self._xy.size else 0.0
        return max(dx, dy)

    def _cluster_at_position(self, drop_xy: Tuple[float, float], allow_far_new: bool = False) -> Optional[int]:
        """
        Return cluster id under drop_xy, preferring hull containment then nearest centroid.
        When allow_far_new=True, drop positions far from any centroid will return None
        to signal new-cluster creation.
        """
        pt = QtCore.QPointF(drop_xy[0], drop_xy[1])

        # 1) If the drop point is inside a drawn hull, use that cluster.
        for cid, hull_item in self._hull_items.items():
            try:
                if hull_item.path().contains(pt):
                    return int(cid)
            except Exception:
                continue

        # 2) Fallback to nearest centroid.
        cent = self._cluster_centroids()
        if not cent:
            return None
        d2 = {cid: (drop_xy[0] - c[0]) ** 2 + (drop_xy[1] - c[1]) ** 2 for cid, c in cent.items()}
        nearest = int(min(d2, key=d2.get))
        if allow_far_new:
            min_dist = float(np.sqrt(d2[nearest]))
            scale = self._cluster_scale()
            thresh = max(0.22 * scale, 0.0)
            if scale <= 0:
                thresh = 0.0
            if thresh > 0 and min_dist > thresh:
                return None
        return nearest

    def _inside_any_hull(self, drop_xy: Tuple[float, float]) -> bool:
        pt = QtCore.QPointF(drop_xy[0], drop_xy[1])
        for hull_item in self._hull_items.values():
            try:
                if hull_item.path().contains(pt):
                    return True
            except Exception:
                continue
        return False

    def _min_drag_distance(self) -> float:
        scale = self._cluster_scale()
        return max(0.03 * scale, 0.35)

    def _drop_too_close_to_points(self, drop_xy: Tuple[float, float], ignore_selected: bool = False) -> bool:
        if self._xy.shape[0] == 0:
            return False
        xy = self._drag_temp_positions if self._drag_temp_positions is not None else self._xy
        scale = self._cluster_scale()
        thr = max(0.08 * scale, 0.8)
        mask = None
        if ignore_selected and self._selected:
            mask = np.ones(len(xy), dtype=bool)
            for i in self._selected:
                if 0 <= i < len(mask):
                    mask[i] = False
        dx = xy[:, 0] - drop_xy[0]
        dy = xy[:, 1] - drop_xy[1]
        d2 = dx * dx + dy * dy
        if mask is not None:
            d2 = d2[mask]
        if d2.size == 0:
            return False
        return bool(d2.min() <= thr * thr)

    def _resolve_new_cluster_color(self, color: QtGui.QColor, new_cid: int) -> Tuple[QtGui.QColor, bool]:
        used_colors = set()
        for cid in sorted(set(map(int, self._cluster.tolist()))):
            try:
                used_colors.add(self._cluster_color(cid, alpha=255).name().lower())
            except Exception:
                continue
        original = QtGui.QColor(color).name().lower()
        if original not in used_colors:
            return color, False

        hsv = QtGui.QColor(color).toHsv()
        hue = hsv.hue()
        if hue < 0:
            hue = 0
        for step in range(1, 13):
            new_hue = (hue + step * 30) % 360
            candidate = QtGui.QColor.fromHsv(new_hue, hsv.saturation(), hsv.value())
            if candidate.name().lower() not in used_colors:
                return candidate, True

        fallback = self._cluster_color(new_cid, alpha=255)
        return fallback, True

    def _assign_selected_to_cluster(self, dst: Optional[int]):
        if dst is None or not self._selected:
            self._drag_temp_positions = None
            self._drag_anchor_xy = None
            self._draw_scatter()
            return
        for i in self._selected:
            self._cluster[i] = int(dst)
        self._drag_temp_positions = None
        self._drag_anchor_xy = None
        self.redraw_all()
        self.sigClustersChanged.emit()
        self._emit_selection_changed()

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

    def _create_cluster_from_drop(self, drop_xy: Tuple[float, float]):
        new_cid = int(max(self._cluster) if len(self._cluster) else 0) + 1
        default_name = f"Cluster {new_cid}"
        base_name = self._new_cluster_name.strip() or default_name
        existing_names = {str(v).strip().lower() for v in self._cluster_names.values()}
        if base_name.strip().lower() in existing_names:
            base_name = default_name
        name = base_name
        suffix = 2
        while name.strip().lower() in existing_names:
            name = f"{base_name} {suffix}"
            suffix += 1

        color = self._new_cluster_color or self._cluster_color(new_cid, alpha=255)
        color, replaced = self._resolve_new_cluster_color(QtGui.QColor(color), new_cid)
        if replaced:
            QtWidgets.QToolTip.showText(
                QtGui.QCursor.pos(),
                "이미 사용 중인 색입니다. 비슷한 다른 색으로 자동 변경했습니다."
            )

        if self._drag_temp_positions is not None:
            self._xy = self._drag_temp_positions

        selected_indices = sorted(self._selected)
        prev_clusters = [int(self._cluster[i]) for i in selected_indices]
        for i in selected_indices:
            self._cluster[i] = new_cid

        self._cluster_names[new_cid] = name
        self._cluster_custom_colors[new_cid] = QtGui.QColor(color)
        self._new_clusters_created.append((new_cid, name, QtGui.QColor(color)))
        self._new_cluster_history.append((new_cid, selected_indices, prev_clusters))

        self._drag_temp_positions = None
        self._drag_anchor_xy = None
        self.redraw_all()
        self.select_cluster(new_cid)
        self.sigClustersChanged.emit()
        self._emit_selection_changed()

    def _drop_with_snap(self, drop_xy: Tuple[float, float]):
        inside_hull = self._inside_any_hull(drop_xy)
        dst = self._cluster_at_position(drop_xy, allow_far_new=self._new_cluster_enabled)
        if self._new_cluster_enabled and not inside_hull:
            dst = None
        if self._new_cluster_enabled and dst is None:
            if self._drag_anchor_xy is not None:
                dist = float(np.hypot(drop_xy[0] - self._drag_anchor_xy[0], drop_xy[1] - self._drag_anchor_xy[1]))
                if dist < self._min_drag_distance():
                    QtWidgets.QToolTip.showText(
                        QtGui.QCursor.pos(),
                        "드래그 거리가 너무 짧아 새 클러스터 생성이 취소되었습니다."
                    )
                    return
            if self._drop_too_close_to_points(drop_xy, ignore_selected=True):
                QtWidgets.QToolTip.showText(
                    QtGui.QCursor.pos(),
                    "기존 포인트/클러스터 근처에서는 새 클러스터를 만들 수 없습니다."
                )
                return
            if not self._selected:
                QtWidgets.QToolTip.showText(
                    QtGui.QCursor.pos(),
                    "새 클러스터 생성은 포인트 선택이 필요합니다."
                )
                return
            self._create_cluster_from_drop(drop_xy)
        else:
            self._assign_selected_to_cluster(dst)

    def drag_event(self, pos: Tuple[float, float], ev):
        if not self._editable_widget:
            ev.ignore()
            return

        if ev.isStart():
            logger.debug(
                "drag_start edit=%s new_cluster=%s free_move=%s selected=%s pos=%s",
                self._edit_mode_active,
                self._new_cluster_enabled,
                self._free_move_points,
                len(self._selected),
                pos,
            )
            i = self._nearest_point(pos[0], pos[1])
            if i is None:
                if self._selected and (self._new_cluster_enabled or self._free_move_points):
                    self._dragging = True
                    self._drag_committed = False
                    self._drag_temp_positions = self._xy.copy()
                    self._drag_anchor_xy = (pos[0], pos[1])
                    ev.accept()
                    return
                ev.ignore()
                return
            if i not in self._selected:
                self._selected = {i}
            self._dragging = True
            self._drag_committed = False
            self._drag_temp_positions = self._xy.copy()
            self._drag_anchor_xy = (pos[0], pos[1])
            ev.accept()
            self._draw_scatter()
            self._emit_selection_changed()
            return

        if not self._dragging:
            return

        ev.accept()

        if ev.isFinish():
            logger.debug(
                "drag_finish free_move=%s new_cluster=%s selected=%s pos=%s",
                self._free_move_points,
                self._new_cluster_enabled,
                len(self._selected),
                pos,
            )
            if self._drag_committed:
                return
            self._drag_committed = True
            self._dragging = False
            if self._free_move_points:
                self._commit_selected_move(pos)
            else:
                self._drop_with_snap(pos)
            return

        logger.debug(
            "drag_move free_move=%s new_cluster=%s selected=%s pos=%s",
            self._free_move_points,
            self._new_cluster_enabled,
            len(self._selected),
            pos,
        )
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
        """(No-op merge) Keep existing clusters; only remember label position."""
        self.remember_label_position(src_cluster, drop_xy)
        self._draw_hulls_and_labels()


# -----------------------------------------------------------------------------
# [v8.0] Visual Decision Tree Widget
# -----------------------------------------------------------------------------
class VisualTreeWidget(QtWidgets.QGraphicsView):
    """Draws a node-link diagram of the decision tree."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.nodes_df = None

    def set_tree_data(self, nodes_df: pd.DataFrame):
        self.nodes_df = nodes_df
        self.draw_tree()

    def draw_tree(self):
        self.scene.clear()
        if self.nodes_df is None or self.nodes_df.empty:
            return

        node_map = {row['node_id']: row for _, row in self.nodes_df.iterrows()}
        levels = {}
        for nid, row in node_map.items():
            d = row['depth']
            if d not in levels:
                levels[d] = []
            levels[d].append(nid)

        w, h = 140, 70
        gap_x, gap_y = 30, 100

        pos = {}
        for d in sorted(levels.keys()):
            cnt = len(levels[d])
            total_w = cnt * (w + gap_x)
            start_x = -total_w / 2
            for i, nid in enumerate(sorted(levels[d])):
                pos[nid] = (start_x + i * (w + gap_x), d * (h + gap_y))

        pen = QtGui.QPen(QtCore.Qt.GlobalColor.gray, 2)
        for nid, p in pos.items():
            row = node_map[nid]
            if not row['is_leaf']:
                lid, rid = row['left_id'], row['right_id']
                if pd.notna(lid) and int(lid) in pos:
                    self.scene.addLine(p[0] + w / 2, p[1] + h, pos[int(lid)][0] + w / 2, pos[int(lid)][1], pen)
                if pd.notna(rid) and int(rid) in pos:
                    self.scene.addLine(p[0] + w / 2, p[1] + h, pos[int(rid)][0] + w / 2, pos[int(rid)][1], pen)

        for nid, p in pos.items():
            row = node_map[nid]
            rect = QtWidgets.QGraphicsRectItem(p[0], p[1], w, h)
            color = QtGui.QColor("#e3f2fd") if not row['is_leaf'] else QtGui.QColor("#c8e6c9")
            rect.setBrush(QtGui.QBrush(color))
            rect.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.black))
            self.scene.addItem(rect)

            cond = row.get('condition', '')
            if '&' in str(cond):
                cond = str(cond).split('&')[-1].strip()
            if cond == "(root)":
                cond = "Root"

            info = f"{cond}\nn={row['n']}"
            if row['is_leaf']:
                info += f"\nPred: {row['pred']}"

            txt = QtWidgets.QGraphicsTextItem(info)
            txt.setTextWidth(w)
            txt.setPos(p[0], p[1])
            self.scene.addItem(txt)


# -----------------------------------------------------------------------------
# Decision Tree Setting Splitter (with double-click reset)
# -----------------------------------------------------------------------------

class DecisionTreeSplitter(QtWidgets.QSplitter):
    sigHandleDoubleClicked = QtCore.pyqtSignal()

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setHandleWidth(8)
        self.setChildrenCollapsible(False)

    def createHandle(self):
        handle = super().createHandle()
        handle.setCursor(QtCore.Qt.CursorShape.SplitVCursor)
        return handle

    def mouseDoubleClickEvent(self, event):
        self.sigHandleDoubleClicked.emit()
        super().mouseDoubleClickEvent(event)

# =============================================================================
# app.py (Part 5/8)
# MainWindow, Data Loading, Project Save/Load, Variable Type Manager
# =============================================================================

class IntegratedApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto Segment Tool v8.1")
        self.resize(1600, 950)

        pg.setConfigOptions(antialias=True)

        self.state = AppState()
        self._settings = QtCore.QSettings("AutoSegment", "AutoSegmentTool")

        # Compatibility flags for optional/legacy tab builders
        self._dt_results_built = False
        self._active_cluster_id: Optional[int] = None
        self._suppress_cluster_name_update = False
        self._suppress_summary_selection = False
        self._dt_setting_splitter_sizes: Optional[List[int]] = None

        # Menu Bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        save_act = QtGui.QAction("Save Project...", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._save_project)
        file_menu.addAction(save_act)

        load_act = QtGui.QAction("Load Project...", self)
        load_act.setShortcut("Ctrl+O")
        load_act.triggered.connect(self._load_project)
        file_menu.addAction(load_act)

        file_menu.addSeparator()

        # [v8.1] Variable Type Manager Menu
        vartype_act = QtGui.QAction("Variable Type Manager...", self)
        vartype_act.setShortcut("Ctrl+T")
        vartype_act.triggered.connect(self._open_variable_type_manager)
        file_menu.addAction(vartype_act)

        self._i18n_texts: List[Tuple[Any, str, str]] = []
        self._i18n_tab_labels: Dict[int, Tuple[str, str]] = {}

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab Initialization
        self._build_tab_data()
        self._build_tab_recode()
        self._build_tab_factor()
        self._build_tab_dt_setting()
        self._build_tab_dt_results()
        self._build_tab_dt_editing()
        self._build_tab_grouping()
        self._build_tab_seg_setting()
        self._build_tab_seg_editing()
        self._build_tab_export()
        self._build_tab_rag()

        self._apply_tab_styles()
        self._apply_ui_language(self.state.ui_lang)

        # Footer credit
        footer = QtWidgets.QLabel(
            'Made by <b>jihee.cho</b> | '
            '<a href="https://github.com/jay-lay-down/">github.com/jay-lay-down</a>'
        )
        footer.setTextFormat(QtCore.Qt.TextFormat.RichText)
        footer.setOpenExternalLinks(True)
        self.statusBar().addPermanentWidget(footer)
        self._add_language_toggle()

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

    def _add_language_toggle(self):
        label = QtWidgets.QLabel("언어:")
        combo = QtWidgets.QComboBox()
        combo.addItem("한국어", "ko")
        combo.addItem("English", "en")
        combo.setCurrentIndex(0 if self.state.ui_lang == "ko" else 1)
        combo.currentIndexChanged.connect(lambda _: self._apply_ui_language(combo.currentData()))
        self.statusBar().addPermanentWidget(label)
        self.statusBar().addPermanentWidget(combo)

    def _register_text(self, widget: Any, ko: str, en: str):
        if widget is None:
            return
        self._i18n_texts.append((widget, ko, en))

    def _register_tab_label(self, tab: QtWidgets.QWidget, ko: str, en: str):
        idx = self.tabs.indexOf(tab)
        if idx >= 0:
            self._i18n_tab_labels[idx] = (ko, en)

    def _apply_ui_language(self, lang: str):
        self.state.ui_lang = "en" if lang == "en" else "ko"
        for widget, ko, en in self._i18n_texts:
            text = en if self.state.ui_lang == "en" else ko
            try:
                widget.setText(text)
            except Exception:
                continue
        for idx, (ko, en) in self._i18n_tab_labels.items():
            self.tabs.setTabText(idx, en if self.state.ui_lang == "en" else ko)

    def _ensure_df(self):
        if self.state.df is None:
            raise RuntimeError("No data loaded.")

    def _selected_checked_items(self, widget: QtWidgets.QListWidget) -> List[str]:
        out = []
        for i in range(widget.count()):
            it = widget.item(i)
            if it.checkState() == QtCore.Qt.CheckState.Checked:
                data = it.data(QtCore.Qt.ItemDataRole.UserRole)
                out.append(data if data is not None else it.text())
        return out

    def _checked_or_selected_items(self, widget: QtWidgets.QListWidget) -> List[str]:
        """Returns checked items, or falls back to selected items if none are checked."""

        checked = self._selected_checked_items(widget)
        if checked:
            return checked
        out = []
        for it in widget.selectedItems():
            data = it.data(QtCore.Qt.ItemDataRole.UserRole)
            out.append(data if data is not None else it.text())
        return out

    def _set_checked_for_selected(self, widget: QtWidgets.QListWidget, checked: bool):
        st = QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked
        for it in widget.selectedItems():
            it.setCheckState(st)

    def _set_all_checks(self, widget: QtWidgets.QListWidget, checked: bool):
        st = QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked
        for i in range(widget.count()):
            widget.item(i).setCheckState(st)

    def _set_checked_items(self, widget: QtWidgets.QListWidget, names: List[str]):
        block_prev = widget.blockSignals(True)
        try:
            want = set(names)
            for i in range(widget.count()):
                it = widget.item(i)
                data = it.data(QtCore.Qt.ItemDataRole.UserRole)
                key = data if data is not None else it.text()
                st = QtCore.Qt.CheckState.Checked if key in want else QtCore.Qt.CheckState.Unchecked
                it.setCheckState(st)
        finally:
            widget.blockSignals(block_prev)

    def _sync_demand_target_master_checkbox(self):
        if not hasattr(self, "chk_demand_targets_all"):
            return
        total = self.lst_demand_targets.count()
        checked = len(self._selected_checked_items(self.lst_demand_targets))
        if total == 0:
            state = QtCore.Qt.CheckState.Unchecked
        elif checked == 0:
            state = QtCore.Qt.CheckState.Unchecked
        elif checked == total:
            state = QtCore.Qt.CheckState.Checked
        else:
            state = QtCore.Qt.CheckState.PartiallyChecked

        block_prev = self.chk_demand_targets_all.blockSignals(True)
        try:
            self.chk_demand_targets_all.setCheckState(state)
        finally:
            self.chk_demand_targets_all.blockSignals(block_prev)

    def _restore_demand_target_checks(self):
        if self.state.demand_targets:
            self._set_checked_items(self.lst_demand_targets, self.state.demand_targets)
        self._sync_demand_target_master_checkbox()

    def _on_demand_target_item_changed(self, _item):
        self.state.demand_targets = self._selected_checked_items(self.lst_demand_targets)
        self._sync_demand_target_master_checkbox()

    def _on_toggle_all_demand_targets(self, state: int):
        check = state == QtCore.Qt.CheckState.Checked
        self._set_all_checks(self.lst_demand_targets, check)
        self.state.demand_targets = self._selected_checked_items(self.lst_demand_targets)
        self._sync_demand_target_master_checkbox()

    def _refresh_all_column_lists(self):
        """Updates all ComboBoxes and ListWidgets when new data is loaded."""
        df = self.state.df
        if df is None:
            return
        cols = list(df.columns)

        # Factor Tab
        self.lst_factor_cols.clear()
        for c in cols:
            display = self._resolve_question_label(c, include_code=True)
            it = QtWidgets.QListWidgetItem(display)
            it.setData(QtCore.Qt.ItemDataRole.UserRole, c)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            it.setCheckState(QtCore.Qt.CheckState.Unchecked)
            # [v8.1] Mark categorical variables with different color
            if self.state.is_categorical(c):
                it.setBackground(QtGui.QColor("#fff3e0"))  # Light orange for categorical
            self.lst_factor_cols.addItem(it)

        # DT Tab
        self.lst_dt_predictors.clear()
        for c in cols:
            it = QtWidgets.QListWidgetItem(c)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            it.setCheckState(QtCore.Qt.CheckState.Unchecked)
            # [v8.1] Mark categorical variables
            if self.state.is_categorical(c):
                it.setBackground(QtGui.QColor("#fff3e0"))
            self.lst_dt_predictors.addItem(it)

        self.lst_dep_extra.clear()
        for c in cols:
            it = QtWidgets.QListWidgetItem(c)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            it.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.lst_dep_extra.addItem(it)

        self.cmb_dt_full_dep.clear()
        self.cmb_dt_full_dep.addItems(cols)
        self.cmb_dt_full_ind.clear()
        self.cmb_dt_full_ind.addItems(cols)

        self.cmb_split_dep.clear()
        self.cmb_split_dep.addItems(cols)
        self.cmb_split_ind.clear()
        self.cmb_split_ind.addItems(cols)

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

        self.lst_demand_targets.blockSignals(True)
        self.lst_demand_targets.clear()
        for c in cols:
            it = QtWidgets.QListWidgetItem(c)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            it.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.lst_demand_targets.addItem(it)
        self.lst_demand_targets.blockSignals(False)
        self._restore_demand_target_checks()

    # -------------------------------------------------------------------------
    # [v8.1] Variable Type Manager
    # -------------------------------------------------------------------------
    def _open_variable_type_manager(self):
        """Opens the Variable Type Manager dialog."""
        if self.state.df is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        dlg = VariableTypeManagerDialog(self.state.df, self.state.var_types, self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.state.var_types = dlg.get_var_types()
            self._refresh_all_column_lists()  # Update UI to reflect changes
            self._set_status(f"Variable types updated. {len(self.state.var_types)} custom settings applied.")

    # -------------------------------------------------------------------------
    # Project Save/Load
    # -------------------------------------------------------------------------
    def _save_project(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Project", "", "AST Project (*.ast)")
        if path:
            try:
                with open(path, "wb") as f:
                    pickle.dump(self.state, f)
                self.statusBar().showMessage(f"Project saved to {path}")
            except Exception as e:
                show_error(self, "Save Error", e)

    def _load_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Project", "", "AST Project (*.ast)")
        if path:
            try:
                with open(path, "rb") as f:
                    self.state = pickle.load(f)

                if not hasattr(self.state, "dt_importance_summary"):
                    self.state.dt_importance_summary = None
                if not hasattr(self.state, "demand_targets_used"):
                    self.state.demand_targets_used = getattr(self.state, "demand_targets", None)

                if self.state.df is not None:
                    self.tbl_preview.set_df(self.state.df)
                    self._refresh_all_column_lists()

                if self.state.recode_df is not None:
                    self._update_recode_tab()

                if self.state.factor_loadings is not None:
                    self.tbl_factor_loadings.set_df(self.state.factor_loadings.reset_index())

                if self.state.dt_improve_pivot is not None:
                    self.tbl_dt_pivot.set_df(self.state.dt_improve_pivot)
                if getattr(self.state, "dt_importance_summary", None) is not None:
                    self.tbl_dt_importance.set_df(self.state.dt_importance_summary)

                self.statusBar().showMessage(f"Project loaded from {path}")
            except Exception as e:
                show_error(self, "Load Error", e)

    # -------------------------------------------------------------------------
    # Tab 1: Data Loading
    # -------------------------------------------------------------------------
    def _build_tab_data(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "데이터 로딩")
        self._register_tab_label(tab, "데이터 로딩", "Data Loading")
        layout = QtWidgets.QVBoxLayout(tab)

        row1 = QtWidgets.QHBoxLayout()
        self.txt_path = QtWidgets.QLineEdit()
        self.btn_browse = QtWidgets.QPushButton("엑셀 찾기")
        style_button(self.btn_browse, level=1)
        self.btn_browse.clicked.connect(self._browse_excel)
        self._register_text(self.btn_browse, "엑셀 찾기", "Browse Excel")

        self.cmb_sheet = QtWidgets.QComboBox()
        self.btn_load = QtWidgets.QPushButton("데이터 불러오기")
        style_button(self.btn_load, level=2)
        self.btn_load.clicked.connect(self._load_excel)
        self._register_text(self.btn_load, "데이터 불러오기", "Load Data")

        row1.addWidget(QtWidgets.QLabel("File Path:"))
        row1.addWidget(self.txt_path, 3)
        row1.addWidget(self.btn_browse)
        row1.addSpacing(10)
        row1.addWidget(QtWidgets.QLabel("Sheet:"))
        row1.addWidget(self.cmb_sheet, 1)
        row1.addWidget(self.btn_load)
        layout.addLayout(row1)

        # [v8.1] Variable Type Manager Button in Data Tab
        row2 = QtWidgets.QHBoxLayout()
        self.btn_var_type_mgr = QtWidgets.QPushButton("📊 변수 유형 관리자 (연속/범주)")
        style_button(self.btn_var_type_mgr, level=3)
        self.btn_var_type_mgr.clicked.connect(self._open_variable_type_manager)
        self._register_text(self.btn_var_type_mgr, "📊 변수 유형 관리자 (연속/범주)", "📊 Variable Type Manager (Numeric/Categorical)")
        self.btn_var_type_mgr.setToolTip(
            "SPSS-style variable type settings.\n"
            "Set variables as Numeric (continuous) or Categorical (discrete).\n"
            "This affects Factor Analysis and Decision Tree logic."
        )
        
        self.lbl_var_type_status = QtWidgets.QLabel("")
        self.lbl_var_type_status.setStyleSheet("color: #1565c0; font-style: italic;")
        
        row2.addWidget(self.btn_var_type_mgr)
        row2.addWidget(self.lbl_var_type_status)
        row2.addStretch(1)
        layout.addLayout(row2)

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

            # Smart Data Cleaning Hook
            dlg = SmartCleanerDialog(df, self)
            if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                df = dlg.df_out
                self._set_status("Data Cleaned & Loaded.")

            self.state.df = df
            self.state.path = path
            self.state.sheet = sheet
            
            # [v8.1] Reset variable types on new data load
            self.state.var_types = {}

            # Load RECODE sheet(s) if they exist
            xls = pd.ExcelFile(path, engine="openpyxl")
            recode_sheets = [s for s in xls.sheet_names if s.lower().startswith("recode")]
            rec_frames: List[pd.DataFrame] = []
            for rs in recode_sheets:
                rec = pd.read_excel(path, sheet_name=rs, engine="openpyxl")
                rec["_SOURCE_SHEET"] = rs
                rec_frames.append(rec)
            if rec_frames:
                merged_recode = pd.concat(rec_frames, ignore_index=True)
                self.state.recode_df = normalize_recode_df(merged_recode)
                self.state.recode_sources = recode_sheets
            else:
                self.state.recode_df = None
                self.state.recode_sources = []

            self.tbl_preview.set_df(df)
            self.lbl_data_info.setText(f"Loaded: {os.path.basename(path)} / sheet={sheet} / rows={len(df):,} cols={df.shape[1]:,}")
            
            # [v8.1] Update variable type status
            n_cat = sum(1 for c in df.columns if is_categorical_series(df[c]))
            n_num = len(df.columns) - n_cat
            self.lbl_var_type_status.setText(f"Auto-detected: {n_num} Numeric, {n_cat} Categorical. Use Manager to customize.")
            
            self._set_status("Data Loaded Successfully.")

            self._update_recode_tab()
            self._refresh_all_column_lists()
            self._reset_downstream_state()

        except Exception as e:
            self.state.last_error = str(e)
            show_error(self, "Load Error", e)

    def _clear_factor_results(self):
        """Resets factor-related state and UI outputs."""
        self.state.factor_model = None
        self.state.factor_cols = None
        self.state.factor_scores = None
        self.state.factor_score_cols = []
        self.state.factor_loadings = None
        self.state.factor_mode = "PCA"
        self.state.factor_ai_names = {}
        self.state.factor_ai_suggestions = {}
        self.state.factor_loadings_order = None

        if hasattr(self, "tbl_factor_loadings"):
            self.tbl_factor_loadings.set_df(None)
        if hasattr(self, "lbl_factor_info"):
            self.lbl_factor_info.setText("Analysis not run.")

    def _clear_dt_outputs(self):
        """Resets decision-tree related state and UI outputs."""
        self.state.dt_improve_pivot = None
        self.state.dt_split_best = None
        self.state.dt_importance_summary = None
        self.state.dt_full_nodes = None
        self.state.dt_full_split_groups = None
        self.state.dt_full_path_info = None
        self.state.dt_full_split_branches = None
        self.state.dt_full_split_paths = None

        if hasattr(self, "tbl_dt_pivot"):
            self.tbl_dt_pivot.set_df(None)
        if hasattr(self, "tbl_dt_importance"):
            self.tbl_dt_importance.set_df(None)
        if hasattr(self, "tbl_dt_bestsplit"):
            self.tbl_dt_bestsplit.set_df(None)
        if hasattr(self, "tbl_split_detail"):
            self.tbl_split_detail.set_df(None)
        if hasattr(self, "tbl_split_paths"):
            self.tbl_split_paths.set_df(None)
        if hasattr(self, "tree_viz"):
            self.tree_viz.set_tree_data(pd.DataFrame())
        if hasattr(self, "lbl_split_imp"):
            self.lbl_split_imp.setText("No split selected.")

    def _reset_downstream_state(self):
        self._clear_factor_results()
        self._clear_dt_outputs()

    # -------------------------------------------------------------------------
    # Tab 2: Recode Mapping
    # -------------------------------------------------------------------------
    def _build_tab_recode(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "리코드 매핑")
        self._register_tab_label(tab, "리코드 매핑", "Recode Mapping")
        layout = QtWidgets.QVBoxLayout(tab)

        layout.addWidget(QtWidgets.QLabel("Load & edit RECODE sheets (QUESTION / CODE / NAME / *_KR)."))

        ctrl = QtWidgets.QHBoxLayout()
        self.btn_reload_recode = QtWidgets.QPushButton("RECODE 시트 새로고침")
        style_button(self.btn_reload_recode, level=1)
        self.btn_reload_recode.clicked.connect(self._reload_recode_from_source)
        self._register_text(self.btn_reload_recode, "RECODE 시트 새로고침", "Reload RECODE sheets from file")
        self.btn_save_recode = QtWidgets.QPushButton("RECODE 변경 적용")
        style_button(self.btn_save_recode, level=2)
        self.btn_save_recode.clicked.connect(self._save_recode_edits)
        self._register_text(self.btn_save_recode, "RECODE 변경 적용", "Apply grid edits to RECODE")
        ctrl.addWidget(self.btn_reload_recode)
        ctrl.addWidget(self.btn_save_recode)
        ctrl.addStretch(1)
        layout.addLayout(ctrl)

        mode_box = QtWidgets.QGroupBox("Question / Code Label Display")
        mode_lay = QtWidgets.QHBoxLayout(mode_box)
        self.radio_recode_original = QtWidgets.QRadioButton("Use original labels (QUESTION / NAME)")
        self.radio_recode_korean = QtWidgets.QRadioButton("Use Korean labels when available (QUESTION_KR / NAME_KR)")
        self.radio_recode_original.setChecked(True)
        self.radio_recode_original.toggled.connect(lambda v: v and self._on_recode_label_mode_changed("original"))
        self.radio_recode_korean.toggled.connect(lambda v: v and self._on_recode_label_mode_changed("korean"))
        mode_lay.addWidget(self.radio_recode_original)
        mode_lay.addWidget(self.radio_recode_korean)
        layout.addWidget(mode_box)

        self.lbl_recode_info = QtWidgets.QLabel("")
        self.lbl_recode_info.setStyleSheet("color:#546e7a;")
        layout.addWidget(self.lbl_recode_info)

        self.tbl_recode = DataFrameTable(editable=True, float_decimals=2)
        layout.addWidget(self.tbl_recode, 1)

    def _update_recode_tab(self):
        if not hasattr(self, "tbl_recode"):
            return
        self.tbl_recode.set_df(self.state.recode_df)
        df = self.state.recode_df

        has_korean = self._has_korean_labels(df)
        if hasattr(self, "radio_recode_korean"):
            self.radio_recode_korean.setEnabled(has_korean)
        if not has_korean and self.state.recode_label_mode == "korean":
            self.state.recode_label_mode = "original"

        if hasattr(self, "radio_recode_original"):
            self.radio_recode_original.setChecked(self.state.recode_label_mode == "original")
        if hasattr(self, "radio_recode_korean"):
            self.radio_recode_korean.setChecked(self.state.recode_label_mode == "korean")

        label_col = self._recode_label_column(df)
        q_col = self._recode_question_label_column(df)
        sheets = ", ".join(self.state.recode_sources) if getattr(self.state, "recode_sources", []) else "None"
        if hasattr(self, "lbl_recode_info"):
            self.lbl_recode_info.setText(
                f"Loaded RECODE sheets: {sheets} | Code label column: {label_col} | Question label column: {q_col}"
            )

    def _on_recode_label_mode_changed(self, mode: str):
        self.state.recode_label_mode = mode
        self._update_recode_tab()
        self._refresh_all_column_lists()
        self._render_factor_loadings_table()
        self._set_status(f"Recode label mode set to: {mode}")

    def _reload_recode_from_source(self):
        try:
            if not self.state.path:
                raise RuntimeError("데이터 파일을 먼저 불러온 뒤 RECODE 시트를 새로고침하세요.")
            xls = pd.ExcelFile(self.state.path, engine="openpyxl")
            recode_sheets = [s for s in xls.sheet_names if s.lower().startswith("recode")]
            if not recode_sheets:
                raise RuntimeError("RECODE 시트를 찾을 수 없습니다.")

            frames: List[pd.DataFrame] = []
            for rs in recode_sheets:
                rec = pd.read_excel(self.state.path, sheet_name=rs, engine="openpyxl")
                rec["_SOURCE_SHEET"] = rs
                frames.append(rec)
            merged = pd.concat(frames, ignore_index=True)
            self.state.recode_df = normalize_recode_df(merged)
            self.state.recode_sources = recode_sheets
            self._update_recode_tab()
            self._set_status(f"RECODE 새로고침 완료: {', '.join(recode_sheets)}")
        except Exception as e:
            self.state.last_error = str(e)
            show_error(self, "Reload RECODE Error", e)

    def _save_recode_edits(self):
        try:
            df = self._extract_table_to_df(self.tbl_recode)
            if df.empty:
                self.state.recode_df = None
                self.state.recode_sources = []
            else:
                self.state.recode_df = normalize_recode_df(df)
            self._update_recode_tab()
            self._set_status("RECODE 매핑 변경사항이 반영되었습니다.")
        except Exception as e:
            self.state.last_error = str(e)
            show_error(self, "Save RECODE Error", e)

    def _extract_table_to_df(self, table: QtWidgets.QTableWidget) -> pd.DataFrame:
        cols = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
        data = []
        for r in range(table.rowCount()):
            row = {}
            for c, col_name in enumerate(cols):
                item = table.item(r, c)
                row[col_name] = item.text() if item is not None else ""
            data.append(row)
        return pd.DataFrame(data)

    def _has_korean_labels(self, df: Optional[pd.DataFrame]) -> bool:
        return df is not None and any(col in df.columns for col in ["NAME_KR", "QUESTION_KR"])

    def _recode_label_column(self, df: Optional[pd.DataFrame]) -> str:
        if self.state.recode_label_mode == "korean" and df is not None:
            for c in ["NAME_KR", "LABEL_KR", "NAME_KO"]:
                if c in df.columns:
                    return c
        return "NAME"

    def _recode_question_label_column(self, df: Optional[pd.DataFrame]) -> str:
        if self.state.recode_label_mode == "korean" and df is not None and "QUESTION_KR" in df.columns:
            return "QUESTION_KR"
        return "QUESTION"

    def _get_recode_lookup(self, question: str) -> Dict[str, str]:
        if self.state.recode_df is None:
            return {}
        r = self.state.recode_df
        mask = r["QUESTION"].astype(str).str.strip() == str(question).strip()
        r = r[mask]
        if r.empty:
            return {}
        name_col = self._recode_label_column(r)
        if name_col not in r.columns:
            name_col = "NAME"
        return dict(zip(r["CODE"].astype(str).str.strip(), r[name_col].astype(str).str.strip()))

    def _resolve_question_label(self, question: str, include_code: bool = False) -> str:
        label = str(question)
        if self.state.recode_label_mode != "korean" or self.state.recode_df is None:
            return label
        r = self.state.recode_df
        if "QUESTION" not in r.columns:
            return label
        if "QUESTION_KR" not in r.columns:
            return label
        sub = r[r["QUESTION"].astype(str).str.strip() == str(question).strip()]
        if sub.empty:
            return label
        kr = sub["QUESTION_KR"].dropna().astype(str).str.strip()
        if kr.empty:
            return label
        resolved = kr.iloc[0]
        if include_code and resolved and resolved != label:
            return f"{resolved} ({label})"
        return resolved or label

# =============================================================================
# app.py (Part 6/8)
# Factor Analysis (AI Naming) & Decision Tree Setting Tabs
# =============================================================================

    # -------------------------------------------------------------------------
    # Tab 3: Factor Analysis (PCA / EFA) + AI Auto-Naming
    # -------------------------------------------------------------------------
    def _build_tab_factor(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "요인 분석")
        self._register_tab_label(tab, "요인 분석", "Factor Analysis")

        layout = QtWidgets.QHBoxLayout(tab)
        left = QtWidgets.QVBoxLayout()
        
        # [v8.1] Info about variable types
        info_label = QtWidgets.QLabel(
            "<b>Note:</b> Only <span style='color:#1565c0;'>Numeric</span> variables can be used for Factor Analysis. "
            "<span style='background-color:#fff3e0;'>Orange</span> highlighted = Categorical (excluded)."
        )
        info_label.setWordWrap(True)
        left.addWidget(info_label)
        
        left.addWidget(QtWidgets.QLabel("Select Variables for Analysis:"))

        self.lst_factor_cols = QtWidgets.QListWidget()
        self.lst_factor_cols.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_factor_cols.setMaximumHeight(220)
        size_pol = self.lst_factor_cols.sizePolicy()
        size_pol.setVerticalStretch(0)
        self.lst_factor_cols.setSizePolicy(size_pol)
        left.addWidget(self.lst_factor_cols)

        # Selection Buttons
        btnrow = QtWidgets.QHBoxLayout()
        self.btn_fac_check_sel = QtWidgets.QPushButton("선택 체크")
        style_button(self.btn_fac_check_sel, level=1)
        self.btn_fac_uncheck_sel = QtWidgets.QPushButton("선택 해제")
        style_button(self.btn_fac_uncheck_sel, level=1)
        self.btn_fac_check_all = QtWidgets.QPushButton("숫자 전체 체크")
        style_button(self.btn_fac_check_all, level=1)
        self.btn_fac_uncheck_all = QtWidgets.QPushButton("전체 해제")
        style_button(self.btn_fac_uncheck_all, level=1)

        self.btn_fac_check_sel.clicked.connect(lambda: self._set_checked_for_selected(self.lst_factor_cols, True))
        self.btn_fac_uncheck_sel.clicked.connect(lambda: self._set_checked_for_selected(self.lst_factor_cols, False))
        self.btn_fac_check_all.clicked.connect(self._check_all_numeric_factor)
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
        self.btn_run_factor = QtWidgets.QPushButton("분석 시작")
        style_button(self.btn_run_factor, level=4)
        self.btn_run_factor.clicked.connect(self._run_factor_analysis)
        self._register_text(self.btn_run_factor, "분석 시작", "Run Analysis")

        self.btn_ai_name = QtWidgets.QPushButton("AI 요인명 추천")
        self.btn_ai_name.clicked.connect(self._ai_name_factors)
        style_button(self.btn_ai_name, 3)
        self._register_text(self.btn_ai_name, "AI 요인명 추천", "AI Auto-Name Factors")

        self.btn_save_factor = QtWidgets.QPushButton("요인 결과 저장")
        self.btn_save_factor.clicked.connect(self._save_factor_results)
        style_button(self.btn_save_factor, 2)
        self._register_text(self.btn_save_factor, "요인 결과 저장", "Save Factor Results")

        ctrl.addWidget(QtWidgets.QLabel("Number of Factors (k):"))
        ctrl.addWidget(self.spin_factor_k)
        ctrl.addWidget(self.btn_run_factor)
        ctrl.addWidget(self.btn_ai_name)
        ctrl.addWidget(self.btn_save_factor)
        left.addLayout(ctrl)

        self.lbl_factor_info = QtWidgets.QLabel("Analysis not run.")
        self.lbl_factor_info.setWordWrap(True)
        left.addWidget(self.lbl_factor_info)

        # Factor name editor (AI suggestions + manual edits)
        name_grp = QtWidgets.QGroupBox("Factor Names (AI 제안/수정)")
        name_lay = QtWidgets.QVBoxLayout(name_grp)
        self.tbl_factor_name_editor = QtWidgets.QTableWidget()
        self.tbl_factor_name_editor.setColumnCount(3)
        self.tbl_factor_name_editor.setHorizontalHeaderLabels(["Factor", "Suggested", "Final Name"])
        self.tbl_factor_name_editor.horizontalHeader().setStretchLastSection(True)
        self.tbl_factor_name_editor.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.tbl_factor_name_editor.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.AllEditTriggers)
        self.tbl_factor_name_editor.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_factor_name_editor.setAlternatingRowColors(True)
        self.tbl_factor_name_editor.setMinimumHeight(140)
        name_lay.addWidget(self.tbl_factor_name_editor, 1)

        name_btn_row = QtWidgets.QHBoxLayout()
        self.btn_apply_factor_names = QtWidgets.QPushButton("요인명 적용")
        style_button(self.btn_apply_factor_names, level=2)
        self.btn_apply_factor_names.clicked.connect(self._apply_factor_names_from_editor)
        self.btn_reset_factor_names = QtWidgets.QPushButton("요인명 초기화")
        style_button(self.btn_reset_factor_names, level=1)
        self.btn_reset_factor_names.clicked.connect(self._reset_factor_names)
        self._register_text(self.btn_apply_factor_names, "요인명 적용", "Apply Edited Names")
        self._register_text(self.btn_reset_factor_names, "요인명 초기화", "Clear Names")
        name_btn_row.addWidget(self.btn_apply_factor_names)
        name_btn_row.addWidget(self.btn_reset_factor_names)
        name_lay.addLayout(name_btn_row)
        left.addWidget(name_grp, 1)

        layout.addLayout(left, 2)

        right = QtWidgets.QVBoxLayout()
        loadings_header = QtWidgets.QHBoxLayout()
        loadings_header.addWidget(QtWidgets.QLabel("Loadings Matrix (Preview):"))
        loadings_header.addStretch(1)
        self.btn_factor_apply_order = QtWidgets.QPushButton("행 순서 적용")
        style_button(self.btn_factor_apply_order, level=1)
        self.btn_factor_apply_order.clicked.connect(self._apply_factor_loadings_order)
        self.btn_factor_reset_order = QtWidgets.QPushButton("행 순서 초기화")
        style_button(self.btn_factor_reset_order, level=1)
        self.btn_factor_reset_order.clicked.connect(self._reset_factor_loadings_order)
        self._register_text(self.btn_factor_apply_order, "행 순서 적용", "Apply Row Order")
        self._register_text(self.btn_factor_reset_order, "행 순서 초기화", "Reset Order")
        loadings_header.addWidget(self.btn_factor_apply_order)
        loadings_header.addWidget(self.btn_factor_reset_order)
        loadings_header.addSpacing(12)
        loadings_header.addWidget(QtWidgets.QLabel("Min |loading|"))
        self.txt_factor_min_loading = QtWidgets.QLineEdit()
        self.txt_factor_min_loading.setFixedWidth(80)
        self.txt_factor_min_loading.setPlaceholderText("0.00")
        self.txt_factor_min_loading.setValidator(QtGui.QDoubleValidator(0.0, 10.0, 3))
        self.txt_factor_min_loading.textChanged.connect(self._render_factor_loadings_table)
        loadings_header.addWidget(self.txt_factor_min_loading)
        right.addLayout(loadings_header)
        self.tbl_factor_loadings = DataFrameTable(float_decimals=3)
        self.tbl_factor_loadings.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.tbl_factor_loadings.setDragEnabled(True)
        self.tbl_factor_loadings.setDropIndicatorShown(True)
        self.tbl_factor_loadings.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        right.addWidget(self.tbl_factor_loadings, 1)
        layout.addLayout(right, 3)

    def _check_all_numeric_factor(self):
        """[v8.1] Check only numeric variables for factor analysis."""
        if self.state.df is None:
            return
        for i in range(self.lst_factor_cols.count()):
            it = self.lst_factor_cols.item(i)
            col = it.text()
            # Only check if NOT categorical
            if not self.state.is_categorical(col):
                it.setCheckState(QtCore.Qt.CheckState.Checked)
            else:
                it.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def _run_factor_analysis(self):
        try:
            self._ensure_df()
            df = self.state.df
            cols = self._selected_checked_items(self.lst_factor_cols)
            
            if len(cols) < 2:
                raise RuntimeError("Please select at least 2 variables.")

            # [v8.1] Filter out categorical variables with warning
            numeric_cols = []
            skipped_cols = []
            for c in cols:
                if self.state.is_categorical(c):
                    skipped_cols.append(c)
                else:
                    numeric_cols.append(c)
            
            if skipped_cols:
                QtWidgets.QMessageBox.warning(
                    self, "Categorical Variables Skipped",
                    f"The following categorical variables were excluded from analysis:\n"
                    f"{', '.join(skipped_cols[:10])}{'...' if len(skipped_cols) > 10 else ''}\n\n"
                    f"Factor Analysis requires numeric variables only."
                )
            
            if len(numeric_cols) < 2:
                raise RuntimeError("Not enough numeric variables. Need at least 2.")
            
            cols = numeric_cols

            # Prepare Numeric Data
            X = to_numeric_df(df, cols)
            X = X.dropna(axis=0, how="all")
            if len(X) < 10:
                raise RuntimeError("Not enough valid rows (after removing all-NaNs).")

            X_f = X.copy()
            for c in X_f.columns:
                m = X_f[c].mean()
                X_f[c] = X_f[c].fillna(m)

            k = int(self.spin_factor_k.value())
            k = min(k, X_f.shape[1])

            is_pca = self.radio_pca.isChecked()
            mode_name = "PCA" if is_pca else "EFA"

            if is_pca:
                model = PCA(n_components=k, random_state=42)
                scores = model.fit_transform(X_f.values)
                components = model.components_
                expl_var = model.explained_variance_ratio_
                info_text = f"Method: PCA. Explained Variance (first 5): {', '.join([f'{v:.3f}' for v in expl_var[:5]])}"
            else:
                try:
                    model = FactorAnalysis(n_components=k, rotation='varimax', random_state=42)
                except Exception:
                    model = FactorAnalysis(n_components=k, random_state=42)

                scores = model.fit_transform(X_f.values)
                components = model.components_
                info_text = f"Method: Factor Analysis (EFA). Latent factors extracted."

            score_cols = [f"Factor{i+1}" for i in range(k)]
            scores_df = pd.DataFrame(scores, index=X_f.index, columns=score_cols)

            for c in score_cols:
                df[c] = np.nan
                df.loc[scores_df.index, c] = scores_df[c].values

            loadings = pd.DataFrame(components.T, index=cols, columns=score_cols)

            self.lbl_factor_info.setText(info_text)

            self.state.df = df
            self.state.factor_model = model
            self.state.factor_cols = cols
            self.state.factor_scores = scores_df
            self.state.factor_score_cols = score_cols
            self.state.factor_loadings = loadings
            self.state.factor_mode = mode_name
            self.state.factor_ai_names = {}
            self.state.factor_ai_suggestions = {}

            self._render_factor_loadings_table()
            self._sync_factor_name_editor()

            self.tbl_preview.set_df(df)
            self._refresh_all_column_lists()
            self._set_status(f"{mode_name} completed. Columns {score_cols} added.")

        except Exception as e:
            self.state.last_error = str(e)
            show_error(self, "Analysis Error", e)

    def _save_factor_results(self):
        if self.state.factor_loadings is None and self.state.factor_scores is None:
            QtWidgets.QMessageBox.warning(self, "No Factors", "Run Factor Analysis first.")
            return
        try:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Factor Results",
                "",
                "Excel Files (*.xlsx)"
            )
            if not path:
                return
            if not path.lower().endswith(".xlsx"):
                path = f"{path}.xlsx"
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                if self.state.factor_loadings is not None:
                    self.state.factor_loadings.reset_index().rename(
                        columns={"index": "variable"}
                    ).to_excel(writer, sheet_name="Factor_Loadings", index=False)
                if self.state.factor_scores is not None:
                    self.state.factor_scores.reset_index().rename(
                        columns={"index": "row_index"}
                    ).to_excel(writer, sheet_name="Factor_Scores", index=False)
            self._set_status(f"요인 결과 저장 완료: {path}")
        except Exception as e:
            show_error(self, "Save Factor Results Error", e)

    def _ai_name_factors(self):
        """Uses OpenAI API to rename factors with retry logic."""
        if self.state.factor_loadings is None:
            QtWidgets.QMessageBox.warning(self, "No Factors", "Run Factor Analysis first.")
            return
            
        provider, key = self._get_ai_provider_and_key()
        if not key:
            QtWidgets.QMessageBox.warning(
                self,
                "No Key",
                "Enter API Key in AI Tab for the selected provider (OpenAI/Gemini).",
            )
            return

        loadings = self.state.factor_loadings
        txt = "Based on top loadings, suggest short names (2-3 words) for each factor. Output as JSON {factor_col: name}.\n"
        for col in loadings.columns:
            top = loadings[col].abs().sort_values(ascending=False).head(5)
            vars_ = ", ".join([f"{i} ({v:.2f})" for i, v in zip(top.index, loadings.loc[top.index, col])])
            txt += f"{col}: {vars_}\n"

        try:
            self._set_status("Asking AI (with retry logic)...")
            QtWidgets.QApplication.processEvents()

            # [v8.1] Use new retry-enabled API call
            messages = [{"role": "user", "content": txt}]
            success, result = call_ai_chat(
                provider,
                key,
                messages,
                max_retries=3,
                initial_delay=2.0,
            )

            if success:
                suggestions = self._parse_ai_factor_suggestions(result)
                if suggestions:
                    self.state.factor_ai_suggestions = suggestions
                    self._sync_factor_name_editor(suggestions)
                    self._set_status("AI 제안 로드됨. 표에서 수정 후 적용하세요.")
                else:
                    self._set_status("AI Naming returned no usable mapping.")
            else:
                QtWidgets.QMessageBox.warning(self, "AI Error", result)
                self._set_status("AI Naming Failed.")

        except Exception as e:
            show_error(self, "AI Error", e)

    def _parse_ai_factor_suggestions(self, ai_text: str) -> Dict[str, str]:
        """Parse AI output into a mapping and push it into the editor (no popups)."""
        factors = []
        if self.state.factor_loadings is not None:
            factors = list(map(str, self.state.factor_loadings.columns))
        elif self.state.factor_scores is not None:
            factors = list(map(str, self.state.factor_scores.columns))

        available_cols = set()
        if self.state.factor_scores is not None:
            available_cols.update(map(str, self.state.factor_scores.columns))
        if self.state.factor_loadings is not None:
            available_cols.update(factors)

        def _norm(x: Any) -> str:
            return str(x).strip()

        parsed: Dict[str, str] = {}
        parsed_json: Any = None

        # 1) Try JSON dict/list directly
        try:
            parsed_json = json.loads(ai_text)
        except Exception:
            parsed_json = None

        if isinstance(parsed_json, dict):
            parsed = {str(k): _norm(v) for k, v in parsed_json.items() if _norm(v)}
        elif isinstance(parsed_json, list):
            seq = [_norm(v) for v in parsed_json if _norm(v)]
            parsed = {col: name for col, name in zip(factors or available_cols, seq)}

        # 2) Fallback: parse "factor: name" lines or plain lines
        if not parsed:
            lines = [_norm(l) for l in ai_text.splitlines() if _norm(l)]
            line_map = {}
            for line in lines:
                if ":" in line:
                    left, right = line.split(":", 1)
                    line_map[_norm(left)] = _norm(right)
            if line_map:
                parsed = line_map
            elif lines:
                parsed = {col: name for col, name in zip(factors or available_cols, lines)}

        # 3) Keep only columns we actually have; if keys are missing, align by order
        applied = {}
        for col, name in parsed.items():
            if col in available_cols:
                applied[col] = name

        if not applied and parsed:
            for col, name in zip(factors, parsed.values()):
                if col not in applied and name:
                    applied[col] = name

        if not applied:
            self._set_status("AI 제안 파싱 실패: 응답에서 이름을 찾지 못했습니다.")
            return {}

        return applied

    def _apply_factor_names_from_editor(self):
        """Apply edited factor names from the editor table to loadings/recode."""
        if self.state.factor_loadings is None:
            QtWidgets.QMessageBox.warning(self, "No Factors", "Run Factor Analysis first.")
            return

        rows = self.tbl_factor_name_editor.rowCount()
        applied: Dict[str, str] = {}
        for r in range(rows):
            factor_item = self.tbl_factor_name_editor.item(r, 0)
            sugg_item = self.tbl_factor_name_editor.item(r, 1)
            final_item = self.tbl_factor_name_editor.item(r, 2)
            factor = factor_item.text().strip() if factor_item else ""
            suggested = sugg_item.text().strip() if sugg_item else ""
            final = final_item.text().strip() if final_item else ""
            if not factor:
                continue
            name_to_use = final or suggested
            if name_to_use:
                applied[factor] = name_to_use

        if not applied:
            QtWidgets.QMessageBox.information(self, "No Names", "입력된 이름이 없습니다.")
            return

        self.state.factor_ai_names = applied
        self._apply_factor_column_renames(applied)
        self._render_factor_loadings_table()
        self._update_recode_with_ai_names(applied)
        self._set_status("수정된 요인 이름을 적용했습니다.")
        msg = json.dumps(applied, ensure_ascii=False, indent=2)
        QtWidgets.QMessageBox.information(self, "적용된 이름", msg)

    def _reset_factor_names(self):
        self.state.factor_ai_names = {}
        self.state.factor_ai_suggestions = {}
        self.state.factor_score_cols = list(self.state.factor_scores.columns) if self.state.factor_scores is not None else []
        self._sync_factor_name_editor()
        self._render_factor_loadings_table()
        self._set_status("요인 이름을 초기화했습니다.")

    def _update_recode_with_ai_names(self, ai_map: Dict[str, str]):
        if not ai_map:
            return

        rows = []
        for col, name in ai_map.items():
            rows.append({"QUESTION": str(col), "CODE": str(col), "NAME": str(name)})

        current = normalize_recode_df(self.state.recode_df)
        if current is None or current.empty:
            recode = pd.DataFrame(rows)
        else:
            recode = current.copy()
            for row in rows:
                mask = (
                    recode["QUESTION"].astype(str) == row["QUESTION"]
                ) & (recode["CODE"].astype(str) == row["CODE"])
                if mask.any():
                    recode.loc[mask, "NAME"] = row["NAME"]
                else:
                    recode = pd.concat([recode, pd.DataFrame([row])], ignore_index=True)

        self.state.recode_df = normalize_recode_df(recode)
        self._update_recode_tab()

    def _render_factor_loadings_table(self):
        if self.state.factor_loadings is None:
            self.tbl_factor_loadings.set_df(None)
            self._sync_factor_name_editor()
            return

        disp = self.state.factor_loadings.copy()
        min_cut = 0.0
        min_text = ""
        if hasattr(self, "txt_factor_min_loading"):
            min_text = self.txt_factor_min_loading.text().strip()
        if min_text:
            try:
                min_cut = float(min_text)
            except ValueError:
                min_cut = 0.0

        rename_map = {
            col: f"{self.state.factor_ai_names[col]} ({col})"
            for col in disp.columns
            if col in self.state.factor_ai_names and self.state.factor_ai_names[col]
        }
        if rename_map:
            disp = disp.rename(columns=rename_map)

        base = self.state.factor_loadings
        if self.state.factor_loadings_order:
            try:
                base = base.reindex(self.state.factor_loadings_order)
                disp = disp.reindex(self.state.factor_loadings_order)
            except Exception:
                pass
        abs_values = base.abs()
        dominant_idx = abs_values.values.argmax(axis=1)
        dominant_cols = pd.Index(base.columns).take(dominant_idx)
        dominant_vals = base.to_numpy()[np.arange(len(base)), dominant_idx]
        factor_order = {col: i for i, col in enumerate(base.columns)}
        dominant_series = pd.Series(dominant_cols, index=base.index)
        dominant_order = pd.Series([factor_order[col] for col in dominant_cols], index=base.index)
        dominant_abs = pd.Series(np.abs(dominant_vals), index=base.index)

        if min_cut > 0:
            disp = disp.mask(disp.abs() < min_cut)
        disp["_dominant_order_"] = dominant_order
        disp["_dominant_abs_"] = dominant_abs
        if not self.state.factor_loadings_order:
            disp = disp.sort_values(
                ["_dominant_order_", "_dominant_abs_"],
                ascending=[True, False],
            )
        disp = disp.drop(columns=["_dominant_order_", "_dominant_abs_"])
        disp = disp.reset_index().rename(columns={"index": "variable"})
        variable_keys = disp["variable"].copy()
        disp["variable"] = disp["variable"].map(lambda v: self._resolve_question_label(v, include_code=True))
        self.tbl_factor_loadings.set_df(disp)
        self._factor_loadings_var_keys = list(variable_keys)
        for row, key in enumerate(self._factor_loadings_var_keys):
            item = self.tbl_factor_loadings.item(row, 0)
            if item is not None:
                item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
        display_col_map = {k: rename_map.get(k, k) for k in base.columns}
        dominant_display = dominant_series.map(display_col_map).reindex(variable_keys).values
        self._apply_factor_dominant_highlight(dominant_display)
        self._sync_factor_name_editor()

    def _apply_factor_loadings_order(self):
        if self.state.factor_loadings is None:
            return
        if not hasattr(self, "_factor_loadings_var_keys"):
            return
        ordered_keys: List[str] = []
        for row in range(self.tbl_factor_loadings.rowCount()):
            item = self.tbl_factor_loadings.item(row, 0)
            if item is None:
                continue
            key = item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text()
            ordered_keys.append(str(key))
        if ordered_keys:
            self.state.factor_loadings_order = ordered_keys
            self._render_factor_loadings_table()
            self._set_status("요인 문항 순서를 저장했습니다.")

    def _reset_factor_loadings_order(self):
        self.state.factor_loadings_order = None
        self._render_factor_loadings_table()
        self._set_status("요인 문항 순서를 초기화했습니다.")

    def _apply_factor_dominant_highlight(self, dominant_cols: np.ndarray):
        if not hasattr(self, "tbl_factor_loadings") or dominant_cols.size == 0:
            return

        palette = [
            "#f3f7ff",
            "#f7f3ff",
            "#fff7f0",
            "#f0fff4",
            "#fff0f6",
            "#f0fbff",
            "#fffde8",
            "#f5f5f5",
        ]
        color_map: Dict[str, QtGui.QColor] = {}
        for col in dominant_cols:
            if col not in color_map:
                color_map[col] = QtGui.QColor(palette[len(color_map) % len(palette)])

        headers = [self.tbl_factor_loadings.horizontalHeaderItem(c).text() for c in range(self.tbl_factor_loadings.columnCount())]
        for row in range(self.tbl_factor_loadings.rowCount()):
            factor = dominant_cols[row]
            if factor not in color_map:
                continue
            try:
                col_idx = headers.index(factor)
            except ValueError:
                continue
            item = self.tbl_factor_loadings.item(row, col_idx)
            if item is not None:
                item.setBackground(QtGui.QBrush(color_map[factor]))

    def _sync_factor_name_editor(self, suggestions: Optional[Dict[str, str]] = None):
        """Refresh the editable factor-name table with current factors and suggestions."""
        tbl = getattr(self, "tbl_factor_name_editor", None)
        if tbl is None:
            return

        tbl.clearContents()

        if self.state.factor_loadings is None:
            tbl.setRowCount(0)
            return

        factors = list(self.state.factor_loadings.columns)
        tbl.setRowCount(len(factors))
        tbl.setColumnCount(3)
        tbl.setHorizontalHeaderLabels(["Factor", "Suggested", "Final Name"])

        sug_map = suggestions or self.state.factor_ai_suggestions or {}
        cur_map = self.state.factor_ai_names

        for r, factor in enumerate(factors):
            suggested = sug_map.get(factor, "")
            final = cur_map.get(factor, "") or suggested

            f_item = QtWidgets.QTableWidgetItem(str(factor))
            f_item.setFlags(f_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            s_item = QtWidgets.QTableWidgetItem(str(suggested))
            s_item.setFlags(s_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            f_final = QtWidgets.QTableWidgetItem(str(final))
            f_final.setFlags(f_final.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)

            tbl.setItem(r, 0, f_item)
            tbl.setItem(r, 1, s_item)
            tbl.setItem(r, 2, f_final)

        tbl.resizeColumnsToContents()

    def _apply_factor_column_renames(self, name_map: Dict[str, str]):
        """Rename factor score/loadings columns and propagate to df and cached outputs."""
        if not name_map:
            return

        def _rename_cols(df_obj: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df_obj is None:
                return None
            return df_obj.rename(columns=name_map)

        # 1) Rename factor scores/loadings
        if self.state.factor_scores is not None:
            self.state.factor_scores = _rename_cols(self.state.factor_scores)
            self.state.factor_score_cols = list(self.state.factor_scores.columns)
        if self.state.factor_loadings is not None:
            self.state.factor_loadings = _rename_cols(self.state.factor_loadings)

        # 2) Rename columns in the main dataframe
        if self.state.df is not None:
            self.state.df = self.state.df.rename(columns=name_map)
            self.tbl_preview.set_df(self.state.df)

        # 3) Rename decision tree outputs if present
        def _rename_dep_ind(df_obj: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df_obj is None:
                return None
            df_obj = df_obj.copy()
            df_obj = df_obj.rename(columns=name_map)
            for col in ["dep", "ind", "target", "split_feature", "feature"]:
                if col in df_obj.columns:
                    df_obj[col] = df_obj[col].replace(name_map)
            return df_obj

        self.state.dt_improve_pivot = _rename_cols(self.state.dt_improve_pivot)
        self.state.dt_split_best = _rename_dep_ind(self.state.dt_split_best)
        self.state.dt_importance_summary = _rename_dep_ind(self.state.dt_importance_summary)
        self.state.dt_full_nodes = _rename_dep_ind(self.state.dt_full_nodes)
        self.state.dt_full_split_groups = _rename_dep_ind(self.state.dt_full_split_groups)
        self.state.dt_full_split_branches = _rename_dep_ind(self.state.dt_full_split_branches)
        self.state.dt_full_path_info = _rename_dep_ind(self.state.dt_full_path_info)
        self.state.dt_full_split_paths = _rename_dep_ind(self.state.dt_full_split_paths)
        self.state.dt_full_condition_freq = _rename_dep_ind(self.state.dt_full_condition_freq)
        self.state.dt_full_split_view = _rename_dep_ind(self.state.dt_full_split_view)
        self.state.dt_full_split_pivot = _rename_cols(self.state.dt_full_split_pivot)

        # 4) Refresh UI lists
        self._refresh_all_column_lists()

    def _factor_score_columns(self) -> List[str]:
        """Return current factor score column names in order."""
        if self.state.factor_score_cols:
            return list(self.state.factor_score_cols)
        if self.state.factor_scores is not None:
            return list(self.state.factor_scores.columns)
        return []

    def _get_ai_provider_and_key(self) -> Tuple[str, str]:
        provider = "openai"
        key = ""

        if hasattr(self, "cmb_ai_provider"):
            sel = self.cmb_ai_provider.currentData()
            provider = sel or self.cmb_ai_provider.currentText() or "openai"

        provider = str(provider).lower().strip() or "openai"

        if provider == "gemini" and hasattr(self, "txt_gemini_key"):
            key = self.txt_gemini_key.text().strip()
        elif hasattr(self, "txt_openai_key"):
            key = self.txt_openai_key.text().strip()

        return provider, key

    # -------------------------------------------------------------------------
    # Tab 4: Decision Tree Setting
    # -------------------------------------------------------------------------
    def _build_tab_dt_setting(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "의사결정나무 설정")
        self._register_tab_label(tab, "의사결정나무 설정", "Decision Tree Setting")

        layout = QtWidgets.QVBoxLayout(tab)
        splitter = DecisionTreeSplitter(QtCore.Qt.Orientation.Vertical)
        layout.addWidget(splitter, 1)

        top_widget = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top_widget)
        top_layout.setContentsMargins(6, 6, 6, 6)
        top_layout.setSpacing(6)

        # [v8.1] Enhanced Header with Variable Type Info
        head_row = QtWidgets.QHBoxLayout()
        head_title = QtWidgets.QLabel("<b>Decision Tree Analysis</b>")
        self.lbl_dt_head_summary = QtWidgets.QLabel(
            "1) 타깃/예측변수 선택 → 2) 분석 시작 → 3) 결과 확인"
        )
        self.lbl_dt_head_summary.setWordWrap(True)
        self.btn_dt_head_toggle = QtWidgets.QToolButton()
        self.btn_dt_head_toggle.setText("자세히")
        self.btn_dt_head_toggle.setCheckable(True)
        self.btn_dt_head_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.btn_dt_head_toggle.toggled.connect(self._toggle_dt_head_details)
        head_row.addWidget(head_title)
        head_row.addStretch(1)
        head_row.addWidget(self.btn_dt_head_toggle)
        top_layout.addLayout(head_row)
        top_layout.addWidget(self.lbl_dt_head_summary)

        self.lbl_dt_head_details = QtWidgets.QLabel(
            "1. Select Dependent(Target) & Independent(Predictors) variables.<br>"
            "2. Click 'Run Analysis' to generate Improvement Pivot.<br>"
            "3. Select a cell in Pivot and click 'Recommend Grouping' to auto-create segments.<br>"
            "<i>Note: <span style='background-color:#fff3e0;'>Orange</span> = Categorical (Optimal Subset Split), "
            "White = Numeric (Threshold Split). Use Variable Type Manager to customize.</i>"
        )
        self.lbl_dt_head_details.setWordWrap(True)
        self.lbl_dt_head_details.setVisible(False)
        top_layout.addWidget(self.lbl_dt_head_details)

        # Controls: Targets
        row = QtWidgets.QHBoxLayout()
        self.chk_use_all_factors = QtWidgets.QCheckBox("모든 요인(Factor1..k)을 타깃으로 사용")
        self.chk_use_all_factors.setChecked(True)
        self.btn_run_tree = QtWidgets.QPushButton("분석 시작")
        style_button(self.btn_run_tree, level=4)
        self.btn_run_tree.clicked.connect(self._run_decision_tree_outputs)
        self._register_text(self.btn_run_tree, "분석 시작", "Run Decision Tree Analysis")

        row.addWidget(self.chk_use_all_factors)
        row.addStretch(1)
        row.addWidget(self.btn_run_tree)
        top_layout.addLayout(row)

        extra_box = QtWidgets.QGroupBox("추가 타깃(선택)")
        extra_layout = QtWidgets.QVBoxLayout(extra_box)
        extra_layout.addWidget(QtWidgets.QLabel("추가 타깃으로 사용할 컬럼을 선택하세요."))

        extra_row = QtWidgets.QHBoxLayout()
        self.lst_dep_extra = QtWidgets.QListWidget()
        self.lst_dep_extra.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_dep_extra.setMaximumHeight(90)
        extra_row.addWidget(self.lst_dep_extra, 1)

        self.btn_dep_extra_check_all = QtWidgets.QPushButton("전체 체크")
        style_button(self.btn_dep_extra_check_all, level=1)
        self.btn_dep_extra_uncheck_all = QtWidgets.QPushButton("전체 해제")
        style_button(self.btn_dep_extra_uncheck_all, level=1)

        self.btn_dep_extra_check_all.clicked.connect(lambda: self._set_all_checks(self.lst_dep_extra, True))
        self.btn_dep_extra_uncheck_all.clicked.connect(lambda: self._set_all_checks(self.lst_dep_extra, False))

        btn_col = QtWidgets.QVBoxLayout()
        btn_col.addWidget(self.btn_dep_extra_check_all)
        btn_col.addWidget(self.btn_dep_extra_uncheck_all)
        btn_col.addStretch(1)

        extra_row.addLayout(btn_col)
        extra_layout.addLayout(extra_row)
        top_layout.addWidget(extra_box)

        # Controls: Predictors (Whitelist)
        pred_box = QtWidgets.QGroupBox("독립변수(예측변수) 선택")
        pred_layout = QtWidgets.QVBoxLayout(pred_box)
        pred_layout.setContentsMargins(8, 8, 8, 8)
        pred_layout.setSpacing(6)

        p_row = QtWidgets.QHBoxLayout()
        self.txt_dt_pred_filter = QtWidgets.QLineEdit()
        self.txt_dt_pred_filter.setPlaceholderText("변수 필터...")
        self.txt_dt_pred_filter.textChanged.connect(self._filter_dt_pred_list)

        self.btn_dt_pred_check_sel = QtWidgets.QPushButton("선택 체크")
        style_button(self.btn_dt_pred_check_sel, level=1)
        self.btn_dt_pred_uncheck_sel = QtWidgets.QPushButton("선택 해제")
        style_button(self.btn_dt_pred_uncheck_sel, level=1)
        self.btn_dt_pred_check_all = QtWidgets.QPushButton("전체 체크")
        style_button(self.btn_dt_pred_check_all, level=1)
        self.btn_dt_pred_uncheck_all = QtWidgets.QPushButton("전체 해제")
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
        self.lst_dt_predictors.setMaximumHeight(90)
        pred_layout.addWidget(self.lst_dt_predictors, 1)
        top_layout.addWidget(pred_box)

        splitter.addWidget(top_widget)

        # Importance + Pivot (Main View)
        results_widget = QtWidgets.QWidget()
        results_layout = QtWidgets.QVBoxLayout(results_widget)
        results_layout.setContentsMargins(6, 4, 6, 6)
        results_layout.setSpacing(6)

        results_header = QtWidgets.QHBoxLayout()
        results_header.addWidget(QtWidgets.QLabel("<b>Results</b>"), 1)
        self.btn_dt_toggle_results = QtWidgets.QPushButton("결과 크게 보기")
        style_button(self.btn_dt_toggle_results, level=1)
        self.btn_dt_toggle_results.setCheckable(True)
        self.btn_dt_toggle_results.toggled.connect(self._toggle_dt_setting_results)
        self.btn_dt_reset_layout = QtWidgets.QPushButton("레이아웃 복원")
        style_button(self.btn_dt_reset_layout, level=1)
        self.btn_dt_reset_layout.clicked.connect(self._reset_dt_setting_splitter_layout)
        results_header.addWidget(self.btn_dt_toggle_results)
        results_header.addWidget(self.btn_dt_reset_layout)
        results_layout.addLayout(results_header)

        w1 = QtWidgets.QWidget()
        l1 = QtWidgets.QVBoxLayout(w1)
        l1.addWidget(QtWidgets.QLabel("Predictor Importance (sum of improve_rel, cumulative %)"))
        self.tbl_dt_importance = DataFrameTable(float_decimals=2)
        l1.addWidget(self.tbl_dt_importance, 1)
        l1.addWidget(QtWidgets.QLabel("Improvement Pivot (Rel. Impurity Drop) [Rows=Predictors, Cols=Targets]"))
        self.tbl_dt_pivot = DataFrameTable(float_decimals=2)
        l1.addWidget(self.tbl_dt_pivot, 1)

        rec_layout = QtWidgets.QHBoxLayout()
        self.btn_dt_recommend = QtWidgets.QPushButton("선택 기준 그룹 추천 → 그룹 탭 전송")
        style_button(self.btn_dt_recommend, level=2)
        self.btn_dt_recommend.setMinimumHeight(40)
        self.btn_dt_recommend.clicked.connect(self._recommend_grouping_transfer)
        self._register_text(self.btn_dt_recommend, "선택 기준 그룹 추천 → 그룹 탭 전송", "Recommend Grouping → Send to Group Tab")

        rec_layout.addWidget(self.btn_dt_recommend)
        rec_layout.addStretch(1)
        l1.addLayout(rec_layout)

        results_layout.addWidget(w1, 1)
        splitter.addWidget(results_widget)

        top_widget.setMinimumHeight(170)
        results_widget.setMinimumHeight(200)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.sigHandleDoubleClicked.connect(self._reset_dt_setting_splitter_layout)
        self.dt_setting_splitter = splitter
        self._restore_dt_setting_splitter_state()

    def _filter_dt_pred_list(self):
        term = self.txt_dt_pred_filter.text().strip().lower()
        for i in range(self.lst_dt_predictors.count()):
            it = self.lst_dt_predictors.item(i)
            it.setHidden(term not in it.text().lower())

    def _toggle_dt_head_details(self, checked: bool):
        if not hasattr(self, "lbl_dt_head_details"):
            return
        self.lbl_dt_head_details.setVisible(bool(checked))
        self.btn_dt_head_toggle.setText("접기" if checked else "자세히")

    def _save_dt_setting_splitter_state(self):
        if not hasattr(self, "dt_setting_splitter"):
            return
        if hasattr(self, "btn_dt_toggle_results") and self.btn_dt_toggle_results.isChecked():
            return
        try:
            state = self.dt_setting_splitter.saveState()
            self._settings.setValue("dt_setting_splitter_state", state)
            self._dt_setting_splitter_sizes = self.dt_setting_splitter.sizes()
        except Exception:
            return

    def _restore_dt_setting_splitter_state(self):
        if not hasattr(self, "dt_setting_splitter"):
            return
        restored = False
        try:
            state = self._settings.value("dt_setting_splitter_state")
            if isinstance(state, QtCore.QByteArray):
                restored = self.dt_setting_splitter.restoreState(state)
            elif isinstance(state, (bytes, bytearray)):
                restored = self.dt_setting_splitter.restoreState(QtCore.QByteArray(state))
        except Exception:
            restored = False
        if not restored:
            self.dt_setting_splitter.setSizes([380, 520])
        self._dt_setting_splitter_sizes = self.dt_setting_splitter.sizes()
        self.dt_setting_splitter.splitterMoved.connect(self._save_dt_setting_splitter_state)

    def _reset_dt_setting_splitter_layout(self):
        if not hasattr(self, "dt_setting_splitter"):
            return
        self.dt_setting_splitter.setSizes([320, 560])
        self._dt_setting_splitter_sizes = self.dt_setting_splitter.sizes()
        if hasattr(self, "btn_dt_toggle_results"):
            self.btn_dt_toggle_results.setChecked(False)
        self._save_dt_setting_splitter_state()

    def _toggle_dt_setting_results(self, checked: bool):
        if not hasattr(self, "dt_setting_splitter"):
            return
        if checked:
            self._dt_setting_splitter_sizes = self.dt_setting_splitter.sizes()
            min_top = 170
            min_bottom = 200
            total = sum(self._dt_setting_splitter_sizes or [0, 0])
            if total <= 0:
                total = 800
            self.dt_setting_splitter.setSizes([min_top, max(min_bottom, total - min_top)])
            self.btn_dt_toggle_results.setText("설정 보기(복원)")
        else:
            if self._dt_setting_splitter_sizes:
                self.dt_setting_splitter.setSizes(self._dt_setting_splitter_sizes)
            self.btn_dt_toggle_results.setText("결과 크게 보기")
            self._save_dt_setting_splitter_state()

    def _run_decision_tree_outputs(self):
        """Calculates Improve Pivot and Best Split tables with variable type awareness."""
        try:
            self._ensure_df()
            df = self.state.df

            fac_cols = self._factor_score_columns()

            deps: List[str] = []
            if self.chk_use_all_factors.isChecked() and fac_cols:
                deps.extend(fac_cols)

            extras = self._checked_or_selected_items(self.lst_dep_extra)
            for extra in extras:
                if extra not in deps:
                    deps.append(extra)

            if not deps:
                raise RuntimeError("No dependent targets selected. Run Factor Analysis first or select extra dep.")

            ind_vars = self._checked_or_selected_items(self.lst_dt_predictors)
            ind_vars = [c for c in ind_vars if c not in deps and c != "resp_id"]

            if len(ind_vars) == 0:
                # If user didn't check/select, fall back to all categorical predictors (seg/demographic)
                fallback = [
                    c for c in df.columns
                    if c not in deps and c != "resp_id" and self.state.is_categorical(c, df[c])
                ]

                if not fallback:
                    raise RuntimeError(
                        "No independent variables selected. "
                        "Check predictors or mark categorical columns in Variable Type Manager."
                    )

                ind_vars = fallback
                self._set_status(
                    "No predictors selected → auto-using categorical columns as pivots."
                )

            best_rows = []
            pivot = pd.DataFrame(index=ind_vars, columns=deps, dtype=float)

            for dep in deps:
                y = df[dep]
                # Determine task type for dependent variable
                is_factor = dep in set(self._factor_score_columns()) or str(dep).startswith("PCA")
                if is_factor:
                    task = "reg"
                elif self.state.is_categorical(dep):
                    task = "class"
                else:
                    task = "reg"

                for ind in ind_vars:
                    x = df[ind]
                    
                    # [v8.1] Use variable type from state to force categorical/numeric
                    force_cat = None
                    vtype = self.state.get_var_type(ind)
                    if vtype == VAR_TYPE_CATEGORICAL:
                        force_cat = True
                    elif vtype == VAR_TYPE_NUMERIC:
                        force_cat = False
                    # else: None = auto-detect
                    
                    best, _ = univariate_best_split(y, x, task=task, force_categorical=force_cat)

                    if best is None:
                        pivot.loc[ind, dep] = np.nan
                        continue

                    pivot.loc[ind, dep] = best["improve_rel"]

                    row = {"dep": dep, "ind": ind}
                    for k, v in best.items():
                        if k in ["left_items", "right_items"]:
                            row[k] = str(v) if v is not None else None
                        else:
                            row[k] = v

                    best_rows.append(row)

            best_df = pd.DataFrame(best_rows)
            pivot_reset = pivot.reset_index().rename(columns={"index": "ind"})

            # Predictor-level importance summary with cumulative share & top split description
            importance_summary = pd.DataFrame()
            if not best_df.empty:
                by_ind = best_df.groupby("ind")["improve_rel"].sum().sort_values(ascending=False).reset_index()
                total_imp = by_ind["improve_rel"].sum()
                by_ind["importance_pct"] = np.where(total_imp > 0, by_ind["improve_rel"] / total_imp * 100.0, np.nan)
                by_ind["cum_importance_pct"] = by_ind["importance_pct"].cumsum()

                def _split_desc(row: pd.Series) -> str:
                    stype = str(row.get("split_type", ""))
                    if stype.startswith("categorical"):
                        return f"{row['ind']} in {row.get('left_items')}"
                    return f"{row['ind']} <= {row.get('cutpoint')}"

                top_split = (
                    best_df.sort_values("improve_rel", ascending=False)
                    .groupby("ind")
                    .first()
                    .reset_index()
                )
                top_split["split_desc"] = top_split.apply(_split_desc, axis=1)
                importance_summary = by_ind.merge(top_split[["ind", "dep", "split_desc"]], on="ind", how="left")
                importance_summary = importance_summary.rename(columns={"dep": "top_dep"})

            self.tbl_dt_importance.set_df(importance_summary)
            self.tbl_dt_pivot.set_df(pivot_reset)
            if hasattr(self, "tbl_dt_bestsplit_result") and self.tbl_dt_bestsplit_result is not None:
                self.tbl_dt_bestsplit_result.set_df(best_df)

            self.state.dt_improve_pivot = pivot_reset
            self.state.dt_split_best = best_df
            self.state.dt_importance_summary = importance_summary
            self.state.dt_selected_deps = deps
            self.state.dt_selected_inds = ind_vars

            self.cmb_dt_full_dep.clear()
            self.cmb_dt_full_dep.addItems(deps)
            self.cmb_dt_full_ind.clear()
            self.cmb_dt_full_ind.addItems(ind_vars)

            self.cmb_split_dep.clear()
            self.cmb_split_dep.addItems(deps)
            self.cmb_split_ind.clear()
            self.cmb_split_ind.addItems(ind_vars)

            self._refresh_split_path_options()
            self._refresh_dt_edit_var_options()

            if self.tbl_dt_pivot.rowCount() > 0:
                self.tbl_dt_pivot.selectRow(0)

            used_msg = (
                f"Targets: {len(deps)} | Predictors (rows): {len(ind_vars)}"
            )
            self._set_status(f"Decision Tree analysis completed. {used_msg}")

        except Exception as e:
            self.state.last_error = str(e)
            show_error(self, "DT Analysis Error", e)

    def _recommend_grouping_transfer(self):
        """Auto-generates grouping mapping based on the best split for selected Ind."""
        try:
            sel_model_rows = self.tbl_dt_pivot.selectionModel().selectedRows()
            if sel_model_rows:
                row_idx = sel_model_rows[0].row()
            elif self.tbl_dt_pivot.currentRow() >= 0:
                row_idx = self.tbl_dt_pivot.currentRow()
            elif self.tbl_dt_pivot.rowCount() == 1:
                row_idx = 0
            else:
                raise RuntimeError("Please select a row (Predictor) in the Pivot table.")

            ind_item = self.tbl_dt_pivot.item(row_idx, 0)
            if ind_item is None:
                raise RuntimeError("Pivot table is empty. Run Decision Tree analysis first.")

            ind_val = ind_item.text()

            if self.state.dt_split_best is None:
                raise RuntimeError("No Best Split data. Run Analysis first.")

            relevant = self.state.dt_split_best[self.state.dt_split_best["ind"] == ind_val]
            if relevant.empty:
                raise RuntimeError(f"No valid split found for '{ind_val}'.")

            best_row = relevant.loc[relevant["improve_rel"].idxmax()]

            self._ensure_df()
            df = self.state.df

            vals = pd.Series(df[ind_val].dropna().unique()).astype(str)
            try:
                vv = vals.astype(float)
                order = np.argsort(vv.values)
                vals = vals.iloc[order]
            except Exception:
                vals = vals.sort_values()

            rec_name = self._get_recode_lookup(ind_val)
            recode_names = [rec_name.get(v, "") for v in vals.values]

            seg_labels = []
            split_type = best_row["split_type"]
            cutpoint = best_row["cutpoint"]

            if split_type == "categorical(subset)":
                left_items_raw = best_row["left_items"]
                if isinstance(left_items_raw, str):
                    try:
                        left_items = ast.literal_eval(left_items_raw)
                    except Exception:
                        left_items = []
                elif isinstance(left_items_raw, list):
                    left_items = left_items_raw
                else:
                    left_items = []

                left_set = set(map(str, left_items))

                label_L = "Group_A"
                label_R = "Group_B"

                for v in vals.values:
                    if str(v) in left_set:
                        seg_labels.append(label_L)
                    else:
                        seg_labels.append(label_R)

            elif split_type.startswith("categorical"):
                target_val = str(best_row["left_group"])
                label_L = f"Group_{target_val}"
                label_R = "Group_Rest"
                for v in vals.values:
                    if v == target_val:
                        seg_labels.append(label_L)
                    else:
                        seg_labels.append(label_R)

            else:
                try:
                    thr = float(cutpoint)
                    label_L = f"Low(v<={thr:g})"
                    label_R = f"High(v>{thr:g})"
                    for v in vals.values:
                        try:
                            vf = float(v)
                            if vf <= thr:
                                seg_labels.append(label_L)
                            else:
                                seg_labels.append(label_R)
                        except Exception:
                            seg_labels.append("Unknown")
                except Exception:
                    seg_labels = ["Manual_Fix"] * len(vals)

            map_df = pd.DataFrame({
                "source_value": vals.values,
                "recode_name": recode_names,
                "segment_label": seg_labels
            })

            self.cmb_group_source.setCurrentText(ind_val)
            self.tbl_group_map.set_df(map_df)
            self.txt_group_newcol.setText(f"{ind_val}_seg")

            self.tabs.setCurrentIndex(6)
            self._set_status(f"Recommendation for '{ind_val}' transferred to Group Tab.")

        except Exception as e:
            self.state.last_error = str(e)
            show_error(self, "Recommendation Error", e)

# =============================================================================
# app.py (Part 7/8)
# Decision Tree Results, Grouping, Segmentation Setting & Editing Tabs
# =============================================================================

    # -------------------------------------------------------------------------
    # Tab 5: Decision Tree Results (Full Tree Viewer)
    # -------------------------------------------------------------------------
    def _build_tab_dt_results(self):
        if self._dt_results_built:
            return
        self._dt_results_built = True

        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "의사결정나무 결과")
        self._register_tab_label(tab, "의사결정나무 결과", "Decision Tree Results")

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
        self.btn_dt_full_run = QtWidgets.QPushButton("분석 시작")
        style_button(self.btn_dt_full_run, level=4)
        self.btn_dt_full_run.clicked.connect(self._run_dt_full_for_selected)
        self._register_text(self.btn_dt_full_run, "분석 시작", "Run Full Tree Analysis")

        ctrl.addWidget(QtWidgets.QLabel("Target (Dep)"))
        ctrl.addWidget(self.cmb_dt_full_dep, 2)
        ctrl.addWidget(QtWidgets.QLabel("Predictor (Ind)"))
        ctrl.addWidget(self.cmb_dt_full_ind, 3)
        ctrl.addWidget(QtWidgets.QLabel("Max Depth"))
        ctrl.addWidget(self.spin_dt_full_depth)
        ctrl.addWidget(self.btn_dt_full_run)
        layout.addLayout(ctrl)

        srow = QtWidgets.QHBoxLayout()
        self.cmb_split_dep = QtWidgets.QComboBox()
        self.cmb_split_ind = QtWidgets.QComboBox()

        self.cmb_split_select = QtWidgets.QComboBox()
        self.cmb_split_select.currentIndexChanged.connect(self._split_update_detail)
        srow.addWidget(QtWidgets.QLabel("Select Split Node:"))
        srow.addWidget(self.cmb_split_select, 4)
        layout.addLayout(srow)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        tree_box = QtWidgets.QGroupBox("Visual Tree Structure")
        tb_lay = QtWidgets.QVBoxLayout(tree_box)
        self.tree_viz = VisualTreeWidget()
        tb_lay.addWidget(self.tree_viz)
        splitter.addWidget(tree_box)

        botw = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(botw)

        detail_box = QtWidgets.QGroupBox("Split Detail")
        dlay = QtWidgets.QVBoxLayout(detail_box)
        self.lbl_split_imp = QtWidgets.QLabel("No split selected.")
        self.lbl_split_imp.setWordWrap(True)
        dlay.addWidget(self.lbl_split_imp)
        self.tbl_split_detail = DataFrameTable(float_decimals=2)
        dlay.addWidget(self.tbl_split_detail, 1)
        bl.addWidget(detail_box, 1)
        note = QtWidgets.QLabel(
            "Split-level path summaries have moved to the new Decision Tree Editing tab."
        )
        note.setStyleSheet("color:#546e7a;")
        bl.addWidget(note)
        splitter.addWidget(botw)

        splitter.setSizes([260, 660])
        layout.addWidget(splitter, 1)

        self._refresh_split_path_options()

    def _compute_full_tree_internal(self, dep: str, ind: str):
        df = self.state.df
        if df is None:
            raise RuntimeError("No data.")

        if dep not in df.columns or ind not in df.columns:
            raise RuntimeError("Columns missing.")

        is_factor = dep in set(self._factor_score_columns()) or str(dep).startswith("PCA")
        if is_factor:
            task = "reg"
        elif self.state.is_categorical(dep):
            task = "class"
        else:
            task = "reg"

        # [v8.1] Get force_categorical for the predictor
        force_cat = None
        vtype = self.state.get_var_type(ind)
        if vtype == VAR_TYPE_CATEGORICAL:
            force_cat = True
        elif vtype == VAR_TYPE_NUMERIC:
            force_cat = False

        nodes_df, split_groups, branches, path_info, cond_freq = build_univariate_tree_full(
            df=df, dep=dep, ind=ind, task=task,
            max_depth=int(self.spin_dt_full_depth.value()),
            min_leaf=1,
            min_split=2,
            max_unique_cat=50,
            min_improve_rel=0.0,
            force_categorical=force_cat
        )

        split_paths = pd.DataFrame()
        if not split_groups.empty and not nodes_df.empty:
            split_paths = split_groups.merge(
                nodes_df[["node_id", "condition"]], on="node_id", how="left"
            )
            split_paths = split_paths.rename(columns={"condition": "path_to_node"})

        self.state.dt_full_nodes = nodes_df
        self.state.dt_full_split_groups = split_groups
        self.state.dt_full_split_branches = branches
        self.state.dt_full_path_info = path_info
        self.state.dt_full_split_paths = split_paths
        self.state.dt_full_condition_freq = cond_freq
        self.state.dt_full_selected = (dep, ind)
        self.state.dt_full_split_view = None
        self.state.dt_full_split_pivot = None

        return task, nodes_df, split_groups, path_info, cond_freq

    def _run_dt_full_for_selected(self):
        try:
            self._ensure_df()
            dep = self.cmb_dt_full_dep.currentText().strip()
            ind = self.cmb_dt_full_ind.currentText().strip()
            if not dep or not ind:
                raise RuntimeError("Select Dep and Ind.")

            task, nodes_df, split_groups, path_info, cond_freq = self._compute_full_tree_internal(dep, ind)

            self.tree_viz.set_tree_data(nodes_df)

            self.cmb_split_select.blockSignals(True)
            self.cmb_split_select.clear()
            if not split_groups.empty:
                for _, row in split_groups.iterrows():
                    desc = f"Split {int(row['split_num'])} @ node {int(row['node_id'])}: {row['left_group']} / {row['right_group']} (Δ={fmt_float(row['improve_rel'], 2)})"
                    self.cmb_split_select.addItem(desc, int(row["split_num"]))
            self.cmb_split_select.blockSignals(False)

            if self.cmb_split_select.count() > 0:
                self.cmb_split_select.setCurrentIndex(0)
                self._split_update_detail()

            if hasattr(self, "_select_path_filter_variable"):
                self._select_path_filter_variable(ind)
            if hasattr(self, "_update_split_path_table"):
                self._update_split_path_table()

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
            if idx < 0:
                return
            split_num = self.cmb_split_select.currentData()
            if split_num is None:
                return

            row = split_groups[split_groups["split_num"] == split_num].iloc[0]
            left_cond = row["left_group"]
            right_cond = row["right_group"]

            y = df[dep]
            is_factor = dep in set(self._factor_score_columns()) or str(dep).startswith("PCA")
            task = "class" if self.state.is_categorical(dep) and not is_factor else "reg"

            def parse_cond(cond_str):
                s = df[ind]
                cond_str = str(cond_str)
                if " in [" in cond_str and "not in" not in cond_str:
                    try:
                        items = ast.literal_eval(cond_str.split(" in ", 1)[1])
                        return s.astype(str).isin(set(map(str, items))).values
                    except Exception:
                        return np.zeros(len(s), dtype=bool)
                elif " not in [" in cond_str:
                    try:
                        items = ast.literal_eval(cond_str.split(" not in ", 1)[1])
                        return ~s.astype(str).isin(set(map(str, items))).values
                    except Exception:
                        return np.zeros(len(s), dtype=bool)
                elif "<=" in cond_str:
                    try:
                        val = float(cond_str.split("<=")[1])
                        return pd.to_numeric(s, errors='coerce').fillna(9999).values <= val
                    except Exception:
                        return np.zeros(len(s), dtype=bool)
                elif ">" in cond_str:
                    try:
                        val = float(cond_str.split(">")[1])
                        return pd.to_numeric(s, errors='coerce').fillna(-9999).values > val
                    except Exception:
                        return np.zeros(len(s), dtype=bool)
                return np.zeros(len(s), dtype=bool)

            mask_left = parse_cond(left_cond)
            mask_right = parse_cond(right_cond)

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
                if impL < impR:
                    better = "Left node cleaner"
                elif impR < impL:
                    better = "Right node cleaner"
                else:
                    better = "Similar impurity"

            base_text = f"Split {int(split_num)}: {left_cond}  /  {right_cond}\n"
            base_text += f"[task={task}] left_imp={fmt_float(impL, 2)}, right_imp={fmt_float(impR, 2)}  ->  {better}"
            self.lbl_split_imp.setText(base_text)

            detail = pd.DataFrame([{
                "split_num": int(split_num),
                "dep": dep, "ind": ind,
                "left_condition": left_cond, "right_condition": right_cond,
                "left_n": int(yL.size), "right_n": int(yR.size),
                "left_impurity": impL, "right_impurity": impR,
                "better_side": better,
            }])
            self.tbl_split_detail.set_df(detail)
        except Exception as e:
            self.state.last_error = str(e)
            show_error(self, "Split Detail Error", e)

    def _refresh_split_path_options(self):
        if not hasattr(self, "cmb_path_var") or not hasattr(self, "lst_path_vars"):
            return
        vars_available: List[str] = []
        if getattr(self.state, "dt_importance_summary", None) is not None:
            vars_available = self.state.dt_importance_summary.get("ind", pd.Series(dtype=str)).dropna().astype(str).tolist()
        elif getattr(self.state, "dt_split_best", None) is not None:
            vars_available = self.state.dt_split_best.get("ind", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        elif self.state.df is not None:
            vars_available = list(self.state.df.columns)

        vars_available = sorted(dict.fromkeys(vars_available))

        self.cmb_path_var.blockSignals(True)
        self.cmb_path_var.clear()
        self.cmb_path_var.addItem("(All)")
        self.cmb_path_var.addItems(vars_available)
        self.cmb_path_var.blockSignals(False)

        self.lst_path_vars.blockSignals(True)
        self.lst_path_vars.clear()
        for v in vars_available:
            it = QtWidgets.QListWidgetItem(v)
            self.lst_path_vars.addItem(it)
        self.lst_path_vars.blockSignals(False)

    def _toggle_path_multi_mode(self, checked: bool):
        self.cmb_path_var.setVisible(not checked)
        self.lst_path_vars.setVisible(checked)
        self._update_split_path_table()

    def _get_selected_path_vars(self) -> List[str]:
        if not self.chk_path_filter.isChecked():
            return []
        if self.chk_path_multi.isChecked():
            return [it.text() for it in self.lst_path_vars.selectedItems()]
        sel = self.cmb_path_var.currentText().strip()
        return [] if sel in ["", "(All)"] else [sel]

    def _select_path_filter_variable(self, var: str):
        if not hasattr(self, "cmb_path_var") or not hasattr(self, "lst_path_vars"):
            return
        if not var:
            return
        idx = self.cmb_path_var.findText(var)
        if idx >= 0:
            self.cmb_path_var.setCurrentIndex(idx)
        for i in range(self.lst_path_vars.count()):
            if self.lst_path_vars.item(i).text() == var:
                self.lst_path_vars.setCurrentRow(i)
                break

    def _update_split_path_table(self):
        if not hasattr(self, "tbl_split_paths"):
            return
        paths_df = getattr(self.state, "dt_full_split_paths", None)
        if paths_df is None or paths_df.empty:
            paths_df = getattr(self.state, "dt_full_path_info", None)
        if paths_df is None:
            self.tbl_split_paths.set_df(None)
            return

        df_view = paths_df.copy()
        selected_vars = self._get_selected_path_vars()
        if selected_vars:
            df_view = df_view[df_view["ind"].astype(str).isin(selected_vars)]

        self.tbl_split_paths.set_df(df_view)

    # -------------------------------------------------------------------------
    # Optional compatibility: Tab 5 (Legacy) Decision Tree Editing alias
    # -------------------------------------------------------------------------
    def _build_tab_dt_editing(self):
        if getattr(self, "_dt_editing_built", False):
            return
        self._dt_editing_built = True

        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "의사결정나무 편집")
        self._register_tab_label(tab, "의사결정나무 편집", "Decision Tree Editing")

        layout = QtWidgets.QVBoxLayout(tab)

        info = QtWidgets.QLabel(
            "의사결정나무 결과를 기준으로 구분변수별 피벗/그룹핑을 직접 편집합니다.<br>"
            "• 위의 콤보박스에서 구분변수를 선택하면 좌측 열에 하위 요인(범주)이 나열됩니다.<br>"
            "• 상단 분석에서 선택했던 모든 목적변수가 가로 축으로 표시되고, 최적 분할 시 어디에 속했는지(left/right)를 보여줍니다.<br>"
            "• 아래에서 범주를 선택해 라벨을 합치고 저장하면 *_seg 컬럼으로 바로 생성됩니다."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        filter_row = QtWidgets.QHBoxLayout()
        self.cmb_dt_edit_var = QtWidgets.QComboBox()
        self.cmb_dt_edit_var.currentIndexChanged.connect(self._load_dt_edit_grid)
        filter_row.addWidget(QtWidgets.QLabel("구분변수 선택"))
        filter_row.addWidget(self.cmb_dt_edit_var, 3)

        self.lbl_dt_edit_targets = QtWidgets.QLabel("(목적변수: 분석 실행 후 자동 채움)")
        self.lbl_dt_edit_targets.setStyleSheet("color:#546e7a;")
        filter_row.addWidget(self.lbl_dt_edit_targets, 2)
        layout.addLayout(filter_row)

        view_row = QtWidgets.QHBoxLayout()
        self.cmb_dt_edit_view_mode = QtWidgets.QComboBox()
        self.cmb_dt_edit_view_mode.addItem("안 1) 좌/우 + 개선도 요약", "split")
        self.cmb_dt_edit_view_mode.addItem("안 2) 모든 조합 + 개선도", "combo")
        self.cmb_dt_edit_view_mode.currentIndexChanged.connect(self._on_dt_edit_view_changed)
        view_row.addWidget(QtWidgets.QLabel("보기 전환"))
        view_row.addWidget(self.cmb_dt_edit_view_mode, 2)
        view_row.addStretch(1)
        layout.addLayout(view_row)

        self.tbl_dt_edit_grid = DataFrameTable(editable=True, float_decimals=2, max_col_width=260)
        self.tbl_dt_edit_combo = DataFrameTable(editable=False, float_decimals=3, max_col_width=260)
        self.dt_edit_view_stack = QtWidgets.QStackedWidget()
        self.dt_edit_view_stack.addWidget(self.tbl_dt_edit_grid)
        self.dt_edit_view_stack.addWidget(self.tbl_dt_edit_combo)
        layout.addWidget(self.dt_edit_view_stack, 1)

        summary_box = QtWidgets.QGroupBox("목적변수별 개선도 요약 (선택 구분변수)")
        summary_layout = QtWidgets.QVBoxLayout(summary_box)
        self.tbl_dt_edit_summary = DataFrameTable(editable=False, float_decimals=3, max_col_width=260)
        summary_layout.addWidget(self.tbl_dt_edit_summary)
        layout.addWidget(summary_box)

        merge_row = QtWidgets.QHBoxLayout()
        self.txt_dt_edit_merge_label = QtWidgets.QLineEdit("Merged")
        self.btn_dt_edit_merge = QtWidgets.QPushButton("선택 범주 묶기 (라벨 적용)")
        style_button(self.btn_dt_edit_merge, level=1)
        self.btn_dt_edit_merge.clicked.connect(self._dt_edit_merge_selected)
        self.btn_dt_edit_reset = QtWidgets.QPushButton("원본 라벨로 초기화")
        style_button(self.btn_dt_edit_reset, level=1)
        self.btn_dt_edit_reset.clicked.connect(self._reset_dt_edit_groups)

        merge_row.addWidget(QtWidgets.QLabel("새 그룹 라벨"))
        merge_row.addWidget(self.txt_dt_edit_merge_label, 2)
        merge_row.addWidget(self.btn_dt_edit_merge)
        merge_row.addWidget(self.btn_dt_edit_reset)
        merge_row.addStretch(1)
        layout.addLayout(merge_row)

        apply_row = QtWidgets.QHBoxLayout()
        self.txt_dt_edit_newcol = QtWidgets.QLineEdit("")
        self.txt_dt_edit_newcol.setPlaceholderText("예: gender_dt_seg")
        self.btn_dt_edit_apply = QtWidgets.QPushButton("그룹핑 저장 → 새 *_seg 생성")
        style_button(self.btn_dt_edit_apply, level=2)
        self.btn_dt_edit_apply.clicked.connect(self._apply_dt_edit_grouping)

        apply_row.addWidget(QtWidgets.QLabel("새 컬럼 이름"))
        apply_row.addWidget(self.txt_dt_edit_newcol, 3)
        apply_row.addWidget(self.btn_dt_edit_apply)
        layout.addLayout(apply_row)

        self._refresh_dt_edit_var_options()
        self._switch_dt_edit_view_mode()
        self._load_dt_edit_grid()

    def _refresh_dt_edit_var_options(self):
        if not hasattr(self, "cmb_dt_edit_var"):
            return

        current = self.cmb_dt_edit_var.currentText()
        df = self.state.df
        vars_available: List[str] = []

        if self.state.dt_split_best is not None and not self.state.dt_split_best.empty:
            vars_available = (
                self.state.dt_split_best.get("ind", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        elif self.state.dt_selected_inds:
            vars_available = list(self.state.dt_selected_inds)
        elif df is not None:
            vars_available = [c for c in df.columns if self.state.is_categorical(c)]

        if df is not None:
            vars_available = [v for v in vars_available if v in df.columns]

        vars_available = sorted(dict.fromkeys(vars_available))

        self.cmb_dt_edit_var.blockSignals(True)
        self.cmb_dt_edit_var.clear()
        self.cmb_dt_edit_var.addItems(vars_available)
        if current:
            idx = self.cmb_dt_edit_var.findText(current)
            if idx >= 0:
                self.cmb_dt_edit_var.setCurrentIndex(idx)
        self.cmb_dt_edit_var.blockSignals(False)

    def _switch_dt_edit_view_mode(self) -> str:
        """Sets the Decision Tree Editing view mode (안1 vs 안2) and toggles controls."""
        mode = getattr(self.state, "dt_edit_view_mode", "split") or "split"
        if hasattr(self, "cmb_dt_edit_view_mode"):
            data = self.cmb_dt_edit_view_mode.currentData()
            mode = data if data in ["split", "combo"] else mode
            idx = self.cmb_dt_edit_view_mode.findData(mode)
            if idx >= 0 and self.cmb_dt_edit_view_mode.currentIndex() != idx:
                self.cmb_dt_edit_view_mode.blockSignals(True)
                self.cmb_dt_edit_view_mode.setCurrentIndex(idx)
                self.cmb_dt_edit_view_mode.blockSignals(False)

        self.state.dt_edit_view_mode = mode

        if hasattr(self, "dt_edit_view_stack"):
            self.dt_edit_view_stack.setCurrentIndex(0 if mode == "split" else 1)

        can_edit = mode == "split"
        for btn_name in ["btn_dt_edit_merge", "btn_dt_edit_reset", "btn_dt_edit_apply"]:
            if hasattr(self, btn_name):
                btn = getattr(self, btn_name)
                btn.setEnabled(can_edit)

        return mode

    def _on_dt_edit_view_changed(self):
        self._switch_dt_edit_view_mode()
        self._load_dt_edit_grid()

    def _parse_items_list(self, val: Any) -> List[str]:
        if val is None:
            return []
        if isinstance(val, (list, tuple, set, np.ndarray, pd.Series)):
            return [str(v) for v in val]
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, (list, tuple, set, np.ndarray, pd.Series)):
                return [str(v) for v in parsed]
        except Exception:
            pass
        txt = str(val)
        if txt in ["", "None", "nan"]:
            return []
        return [txt]

    def _get_dt_edit_targets(self) -> List[str]:
        if self.state.dt_selected_deps:
            return list(self.state.dt_selected_deps)
        if self.state.dt_split_best is not None and not self.state.dt_split_best.empty:
            return (
                self.state.dt_split_best.get("dep", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        df = self.state.df
        if df is None:
            return []
        return self._factor_score_columns()

    def _compute_dt_candidate_splits(self, var: str, dep: str) -> List[dict]:
        df = self.state.df
        if df is None or var not in df.columns or dep not in df.columns:
            return []

        y = df[dep]
        x = df[var]

        is_factor = dep in set(self._factor_score_columns()) or str(dep).startswith("PCA")
        if is_factor:
            task = "reg"
        elif self.state.is_categorical(dep):
            task = "class"
        else:
            task = "reg"

        force_cat = None
        vtype = self.state.get_var_type(var)
        if vtype == VAR_TYPE_CATEGORICAL:
            force_cat = True
        elif vtype == VAR_TYPE_NUMERIC:
            force_cat = False

        _, rows = univariate_best_split(y, x, task=task, force_categorical=force_cat)
        for r in rows:
            r["dep"] = dep
            r["ind"] = var
        return rows

    def _format_split_combo_desc(self, row: dict) -> str:
        stype = str(row.get("split_type", ""))
        if stype.startswith("categorical"):
            left_items = self._parse_items_list(row.get("left_items"))
            right_items = self._parse_items_list(row.get("right_items"))
            if not right_items and left_items:
                right_items = ["(Else)"]
            left_txt = ",".join(left_items) if left_items else str(row.get("left_group", ""))
            right_txt = ",".join(right_items) if right_items else str(row.get("right_group", ""))
            return f"L:{left_txt} | R:{right_txt}"

        cp = row.get("cutpoint")
        cp_txt = fmt_float(cp, 3) if isinstance(cp, (int, float)) else str(cp)
        return f"<= {cp_txt} | > {cp_txt}"

    def _build_dt_combo_view_df(self, var: str, targets: List[str]) -> pd.DataFrame:
        combo_map: Dict[str, dict] = {}
        max_rows = 300

        for dep in targets:
            candidates = self._compute_dt_candidate_splits(var, dep)
            for row in candidates:
                key = self._format_split_combo_desc(row)
                rec = combo_map.setdefault(
                    key,
                    {
                        "split_desc": key,
                        "split_type": row.get("split_type", ""),
                        "left_n": row.get("n_left", np.nan),
                        "right_n": row.get("n_right", np.nan),
                    },
                )
                rec[dep] = row.get("improve_rel", np.nan)

        if not combo_map:
            return pd.DataFrame()

        df_view = pd.DataFrame(combo_map.values())
        if len(df_view) > max_rows:
            df_view["max_improve"] = df_view[targets].max(axis=1)
            df_view = df_view.sort_values("max_improve", ascending=False).head(max_rows)
            df_view = df_view.drop(columns=["max_improve"])

        cols = ["split_desc", "split_type", "left_n", "right_n"] + targets
        cols = [c for c in cols if c in df_view.columns]
        return df_view[cols]

    def _build_dt_edit_summary_df(self, var: str, targets: List[str]) -> pd.DataFrame:
        summary_rows = []
        best_df = getattr(self.state, "dt_split_best", None)
        has_best = best_df is not None and not best_df.empty

        for dep in targets:
            row = None
            if has_best:
                cand = best_df[(best_df["ind"] == var) & (best_df["dep"] == dep)]
                if not cand.empty:
                    row = cand.loc[cand["improve_rel"].idxmax()]

            if row is None:
                candidates = self._compute_dt_candidate_splits(var, dep)
                if candidates:
                    row = max(candidates, key=lambda r: r.get("improve_rel", -np.inf))

            if row is None:
                continue

            summary_rows.append(
                {
                    "dep": dep,
                    "split_type": row.get("split_type", ""),
                    "left_group": row.get("left_group", ""),
                    "right_group": row.get("right_group", ""),
                    "improve_rel": row.get("improve_rel", np.nan),
                    "n_left": row.get("n_left", np.nan),
                    "n_right": row.get("n_right", np.nan),
                }
            )

        if not summary_rows:
            return pd.DataFrame()

        return pd.DataFrame(summary_rows).sort_values("improve_rel", ascending=False)

    def _build_dt_edit_split_view_df(self, df: pd.DataFrame, var: str, targets: List[str]) -> pd.DataFrame:
        cats = (
            pd.Series(df[var].dropna().unique())
            .astype(str)
            .sort_values()
            .tolist()
        )

        split_info: Dict[str, Dict[str, set]] = {}
        if self.state.dt_split_best is not None and not self.state.dt_split_best.empty:
            sub = self.state.dt_split_best[self.state.dt_split_best["ind"] == var]
            for dep in targets:
                cand = sub[sub["dep"] == dep]
                if cand.empty:
                    continue
                row = cand.loc[cand["improve_rel"].idxmax()]
                if not str(row.get("split_type", "")).startswith("categorical"):
                    continue
                left_set = set(self._parse_items_list(row.get("left_items")))
                right_set = set(self._parse_items_list(row.get("right_items")))
                split_info[dep] = {"left": left_set, "right": right_set}

        user_map = self.state.dt_edit_group_map.get(var, {})
        rows = []
        for cat in cats:
            row = {"level": cat}
            for dep in targets:
                info = split_info.get(dep)
                if not info:
                    row[dep] = ""
                    continue
                if cat in info.get("left", set()):
                    row[dep] = "Left"
                elif info.get("right"):
                    row[dep] = "Right" if cat in info.get("right", set()) else "Right(Else)"
                elif info.get("left"):
                    row[dep] = "Right(Else)"
                else:
                    row[dep] = ""
            row["custom_group"] = user_map.get(cat, cat)
            rows.append(row)

        view = pd.DataFrame(rows)
        cols = ["level"] + targets + ["custom_group"] if targets else ["level", "custom_group"]
        view = view[cols]
        return view

    def _load_dt_edit_grid(self):
        if not hasattr(self, "tbl_dt_edit_grid"):
            return

        df = self.state.df
        var = self.cmb_dt_edit_var.currentText().strip() if hasattr(self, "cmb_dt_edit_var") else ""
        targets = self._get_dt_edit_targets()

        self.lbl_dt_edit_targets.setText(
            "목적변수: " + (", ".join(targets) if targets else "(분석 실행 필요)")
        )

        mode = self._switch_dt_edit_view_mode()

        if df is None or not var:
            self.tbl_dt_edit_grid.set_df(None)
            if hasattr(self, "tbl_dt_edit_combo"):
                self.tbl_dt_edit_combo.set_df(None)
            if hasattr(self, "tbl_dt_edit_summary"):
                self.tbl_dt_edit_summary.set_df(None)
            return

        if not var:
            self.txt_dt_edit_newcol.setPlaceholderText("예: gender_dt_seg")
        else:
            suggestion = f"{var}_dt_seg"
            if not self.txt_dt_edit_newcol.text().strip():
                self.txt_dt_edit_newcol.setText(suggestion if suggestion.endswith("_seg") else suggestion + "_seg")

        summary_df = self._build_dt_edit_summary_df(var, targets)
        if hasattr(self, "tbl_dt_edit_summary"):
            self.tbl_dt_edit_summary.set_df(summary_df if not summary_df.empty else None)

        if mode == "combo":
            combo_df = self._build_dt_combo_view_df(var, targets) if targets else pd.DataFrame()
            if hasattr(self, "tbl_dt_edit_combo"):
                self.tbl_dt_edit_combo.set_df(combo_df if not combo_df.empty else None)
            self.tbl_dt_edit_grid.set_df(None)
        else:
            split_df = self._build_dt_edit_split_view_df(df, var, targets)
            self.tbl_dt_edit_grid.set_df(split_df if not split_df.empty else None)
            if hasattr(self, "tbl_dt_edit_combo"):
                self.tbl_dt_edit_combo.set_df(None)

    def _extract_dt_edit_df(self) -> pd.DataFrame:
        table = self.tbl_dt_edit_grid
        cols = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
        data = []
        for r in range(table.rowCount()):
            row = {}
            for c, col_name in enumerate(cols):
                item = table.item(r, c)
                row[col_name] = item.text() if item is not None else ""
            data.append(row)
        return pd.DataFrame(data)

    def _dt_edit_merge_selected(self):
        try:
            if getattr(self.state, "dt_edit_view_mode", "split") != "split":
                raise RuntimeError("좌/우 보기에서만 범주 라벨을 묶을 수 있습니다. 보기 전환을 '안 1'로 변경하세요.")
            label = self.txt_dt_edit_merge_label.text().strip()
            if not label:
                raise RuntimeError("라벨을 입력하세요.")
            sel = self.tbl_dt_edit_grid.selectedItems()
            if not sel:
                raise RuntimeError("묶을 범주 행을 선택하세요.")

            rows_idx = sorted(set([it.row() for it in sel]))
            df_current = self._extract_dt_edit_df()
            for r in rows_idx:
                if "custom_group" in df_current.columns and r < len(df_current):
                    df_current.at[r, "custom_group"] = label

            var = self.cmb_dt_edit_var.currentText().strip()
            if var:
                self.state.dt_edit_group_map[var] = {
                    row["level"]: row.get("custom_group", row.get("level", ""))
                    for _, row in df_current.iterrows()
                    if str(row.get("level", "")) != ""
                }

            self.tbl_dt_edit_grid.set_df(df_current)
            self._set_status("선택한 범주가 새 라벨로 묶였습니다.")
        except Exception as e:
            show_error(self, "Grouping Error", e)

    def _reset_dt_edit_groups(self):
        try:
            if getattr(self.state, "dt_edit_view_mode", "split") != "split":
                raise RuntimeError("좌/우 보기에서만 라벨을 초기화할 수 있습니다. 보기 전환을 '안 1'로 변경하세요.")
            var = self.cmb_dt_edit_var.currentText().strip()
            if var in self.state.dt_edit_group_map:
                self.state.dt_edit_group_map.pop(var, None)
            self._load_dt_edit_grid()
            self._set_status("라벨이 원본 값으로 초기화되었습니다.")
        except Exception as e:
            show_error(self, "Reset Error", e)

    def _apply_dt_edit_grouping(self):
        try:
            if getattr(self.state, "dt_edit_view_mode", "split") != "split":
                raise RuntimeError("조합 보기에서는 저장할 수 없습니다. 보기 전환을 '안 1'로 변경하세요.")
            self._ensure_df()
            df = self.state.df
            var = self.cmb_dt_edit_var.currentText().strip()
            if not var:
                raise RuntimeError("구분변수를 선택하세요.")

            table_df = self._extract_dt_edit_df()
            if table_df.empty:
                raise RuntimeError("그룹핑할 항목이 없습니다.")

            newcol = self.txt_dt_edit_newcol.text().strip()
            if not newcol:
                newcol = f"{var}_dt_seg"
            if not newcol.endswith("_seg"):
                newcol = newcol + "_seg"

            mapping = {}
            for _, row in table_df.iterrows():
                key = str(row.get("level", ""))
                if key == "":
                    continue
                val = str(row.get("custom_group", key)).strip()
                mapping[key] = val if val else key

            s = df[var].astype(str)
            df[newcol] = s.map(mapping).fillna(s)

            self.state.df = df
            self.state.dt_edit_group_map[var] = mapping
            self.tbl_preview.set_df(df)
            self._refresh_all_column_lists()
            self._set_status(f"새 그룹 컬럼 생성: {newcol}")
        except Exception as e:
            show_error(self, "Save Grouping Error", e)

    # -------------------------------------------------------------------------
    # Tab 7: Group & Compose
    # -------------------------------------------------------------------------
    def _build_tab_grouping(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "그룹/조합")
        self._register_tab_label(tab, "그룹/조합", "Group & Compose")

        layout = QtWidgets.QVBoxLayout(tab)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_contents = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_contents)
        scroll_layout.setContentsMargins(8, 8, 8, 8)
        scroll_layout.setSpacing(10)
        scroll.setWidget(scroll_contents)
        layout.addWidget(scroll)

        # Binary Recode Section
        box_bin = QtWidgets.QGroupBox("Quick Binary Recode (2 Values) -> *_seg")
        bin_layout = QtWidgets.QHBoxLayout(box_bin)

        self.cmb_bin_col = QtWidgets.QComboBox()
        self.txt_bin_val1 = QtWidgets.QLineEdit("")
        self.txt_bin_lab1 = QtWidgets.QLineEdit("A")
        self.txt_bin_val2 = QtWidgets.QLineEdit("")
        self.txt_bin_lab2 = QtWidgets.QLineEdit("B")
        self.chk_bin_else_other = QtWidgets.QCheckBox("Else=Other")
        self.txt_bin_else_lab = QtWidgets.QLineEdit("Other")
        self.txt_bin_else_lab.setMinimumWidth(80)

        self.txt_bin_newcol = QtWidgets.QLineEdit("")
        self.txt_bin_newcol.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.txt_bin_val1.setMinimumWidth(70)
        self.txt_bin_val2.setMinimumWidth(70)
        self.txt_bin_lab1.setMinimumWidth(70)
        self.txt_bin_lab2.setMinimumWidth(70)
        self.btn_bin_apply = QtWidgets.QPushButton("이진 리코드 적용")
        style_button(self.btn_bin_apply, level=2)
        self.btn_bin_apply.clicked.connect(self._apply_binary_recode)
        self._register_text(self.btn_bin_apply, "이진 리코드 적용", "Apply Binary Recode")

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
        scroll_layout.addWidget(box_bin)

        # Mapping Table Section
        box = QtWidgets.QGroupBox("General Grouping: Source Value -> Segment Label (Mapping Table)")
        b = QtWidgets.QVBoxLayout(box)

        r1 = QtWidgets.QHBoxLayout()
        self.cmb_group_source = QtWidgets.QComboBox()
        self.cmb_group_source.currentTextChanged.connect(self._on_group_source_changed)
        self.txt_group_newcol = QtWidgets.QLineEdit("")
        self.txt_group_newcol.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_group_build = QtWidgets.QPushButton("매핑표 생성")
        style_button(self.btn_group_build, level=1)
        self.btn_group_build.clicked.connect(self._build_group_mapping)
        self.btn_group_apply = QtWidgets.QPushButton("IV 생성(매핑 적용)")
        style_button(self.btn_group_apply, level=2)
        self.btn_group_apply.clicked.connect(self._apply_group_mapping)
        self._register_text(self.btn_group_build, "매핑표 생성", "Build Mapping Table")
        self._register_text(self.btn_group_apply, "IV 생성(매핑 적용)", "Create IV (Apply Mapping)")

        r1.addWidget(QtWidgets.QLabel("Source Column"))
        r1.addWidget(self.cmb_group_source)
        r1.addSpacing(10)
        r1.addWidget(QtWidgets.QLabel("New Column Name"))
        r1.addWidget(self.txt_group_newcol)
        r1.addWidget(self.btn_group_build)
        r1.addWidget(self.btn_group_apply)
        b.addLayout(r1)

        self.tbl_group_map = DataFrameTable(editable=True, float_decimals=2)
        self.tbl_group_map.setMinimumHeight(300)
        self.tbl_group_map.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked)
        b.addWidget(self.tbl_group_map, 2)

        merge_row = QtWidgets.QHBoxLayout()
        self.txt_group_merge_label = QtWidgets.QLineEdit("MyGroup")
        self.txt_group_merge_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_group_merge_apply = QtWidgets.QPushButton("선택 행 병합(라벨 적용)")
        style_button(self.btn_group_merge_apply, level=1)
        self.btn_group_merge_apply.clicked.connect(self._merge_group_mapping_selected)
        self._register_text(self.btn_group_merge_apply, "선택 행 병합(라벨 적용)", "Merge Selected Rows (Apply Label)")

        merge_row.addWidget(QtWidgets.QLabel("Select rows above & Enter Label to merge:"))
        merge_row.addWidget(self.txt_group_merge_label, 2)
        merge_row.addWidget(self.btn_group_merge_apply)
        b.addLayout(merge_row)
        scroll_layout.addWidget(box, 3)

        # Compose Section
        box2 = QtWidgets.QGroupBox("Combine Segments: Multiple *_seg -> Combined Segment")
        c = QtWidgets.QHBoxLayout(box2)
        self.lst_compose_segs = QtWidgets.QListWidget()
        self.lst_compose_segs.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        right = QtWidgets.QVBoxLayout()
        self.txt_compose_newcol = QtWidgets.QLineEdit("combo_seg")
        self.txt_compose_newcol.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.txt_compose_sep = QtWidgets.QLineEdit("|")
        self.txt_compose_sep.setMinimumWidth(60)
        self.txt_compose_sep.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btn_compose = QtWidgets.QPushButton("세그 조합 생성")
        style_button(self.btn_compose, level=2)
        self.btn_compose.clicked.connect(self._compose_segs)
        self._register_text(self.btn_compose, "세그 조합 생성", "Create Combined Segment")

        right.addWidget(QtWidgets.QLabel("Select *_seg columns"))
        right.addWidget(QtWidgets.QLabel("New Column Name"))
        right.addWidget(self.txt_compose_newcol)
        right.addWidget(QtWidgets.QLabel("Separator"))
        right.addWidget(self.txt_compose_sep)
        right.addWidget(self.btn_compose)
        right.addStretch(1)

        c.addWidget(self.lst_compose_segs, 2)
        c.addLayout(right, 1)
        scroll_layout.addWidget(box2, 1)

        # Cleanup Section
        box3 = QtWidgets.QGroupBox("Cleanup: Delete Derived Columns (Factor / *_seg / Custom)")
        dlay = QtWidgets.QVBoxLayout(box3)

        info = QtWidgets.QLabel(
            "Select columns to delete when a derived factor/segment was created in error. "
            "Original data is untouched on disk; this removes columns from the in-memory dataset."
        )
        info.setWordWrap(True)
        dlay.addWidget(info)

        self.lst_delete_cols = QtWidgets.QListWidget()
        self.lst_delete_cols.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_delete_cols.setMaximumHeight(80)
        dlay.addWidget(self.lst_delete_cols, 1)

        del_row = QtWidgets.QHBoxLayout()
        self.btn_delete_cols = QtWidgets.QPushButton("선택 컬럼 삭제")
        style_button(self.btn_delete_cols, level=2)
        self.btn_delete_cols.clicked.connect(self._delete_selected_columns)
        self._register_text(self.btn_delete_cols, "선택 컬럼 삭제", "Delete Selected Columns")

        del_row.addStretch(1)
        del_row.addWidget(self.btn_delete_cols)
        dlay.addLayout(del_row)

        scroll_layout.addWidget(box3, 0)
        scroll_layout.addStretch(1)

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
            if not self.txt_group_newcol.text().strip():
                self.txt_group_newcol.setText(f"{src}_seg")

            vals = pd.Series(df[src].dropna().unique()).astype(str)
            try:
                vv = vals.astype(float)
                order = np.argsort(vv.values)
                vals = vals.iloc[order]
            except:
                vals = vals.sort_values()

            rec_name = self._get_recode_lookup(src)
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
            if not newcol:
                newcol = src
            if not newcol.endswith("_seg"):
                newcol += "_seg"

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

    def _on_group_source_changed(self, text: str):
        """Auto-suggest a new column name based on the source column."""
        if not text:
            return
        current = self.txt_group_newcol.text().strip()
        if current == "" or current.endswith("_seg") or current == "custom_seg":
            suggested = f"{text}_seg"
            self.txt_group_newcol.setText(suggested)

    def _compose_segs(self):
        try:
            self._ensure_df()
            df = self.state.df
            cols = [it.text() for it in self.lst_compose_segs.selectedItems()]
            if len(cols) < 2:
                raise RuntimeError("Select 2 or more *_seg columns.")
            newcol = self.txt_compose_newcol.text().strip()
            if not newcol:
                base = "_".join(cols)
                newcol = base
            if not newcol.endswith("_seg"):
                newcol += "_seg"
            sep = self.txt_compose_sep.text() or "|"

            df[newcol] = df[cols].astype(str).apply(lambda r: sep.join(r.values.tolist()), axis=1)
            self.state.df = df
            self.tbl_preview.set_df(df)
            self._refresh_all_column_lists()
            self._set_status(f"Combined segment {newcol} created.")
        except Exception as e:
            show_error(self, "Compose Error", e)

    def _delete_selected_columns(self):
        try:
            self._ensure_df()
            df = self.state.df

            selected = [it.data(QtCore.Qt.ItemDataRole.UserRole) for it in self.lst_delete_cols.selectedItems()]
            selected = [c for c in selected if c]
            if not selected:
                raise RuntimeError("삭제할 컬럼을 선택하세요.")

            confirm = QtWidgets.QMessageBox.question(
                self,
                "Delete Columns",
                "선택한 컬럼을 삭제할까요? 이 작업은 현재 세션의 데이터프레임에서만 적용됩니다.",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
                return

            df = df.drop(columns=selected, errors="ignore")

            # Remove variable type overrides for dropped columns
            self.state.var_types = {k: v for k, v in self.state.var_types.items() if k not in selected}

            # Clear factor outputs if any factor-related columns were removed
            factor_related = set(self.state.factor_cols or [])
            if self.state.factor_scores is not None:
                factor_related |= set(self.state.factor_scores.columns)
            needs_factor_reset = bool(
                set(selected) & factor_related
                or any(str(c).startswith(("Factor", "PCA")) for c in selected)
            )
            if needs_factor_reset:
                self._clear_factor_results()

            # Clear decision tree outputs if removed columns were used as dep/ind
            needs_dt_reset = False
            for df_attr in [self.state.dt_improve_pivot, self.state.dt_split_best]:
                if df_attr is None:
                    continue
                if "ind" in df_attr.columns and df_attr["ind"].astype(str).isin(selected).any():
                    needs_dt_reset = True
                if "dep" in df_attr.columns and df_attr["dep"].astype(str).isin(selected).any():
                    needs_dt_reset = True
                if needs_dt_reset:
                    break
            if self.state.dt_full_nodes is not None and "split_feature" in self.state.dt_full_nodes.columns:
                if self.state.dt_full_nodes["split_feature"].astype(str).isin(selected).any():
                    needs_dt_reset = True
            if needs_dt_reset:
                self._clear_dt_outputs()

            self.state.df = df
            self.tbl_preview.set_df(df)
            self._refresh_all_column_lists()

            cleared = " (Factor/DT 결과 초기화)" if needs_factor_reset or needs_dt_reset else ""
            self._set_status(f"Columns deleted: {', '.join(selected)}{cleared}")
        except Exception as e:
            show_error(self, "Delete Error", e)

    # -------------------------------------------------------------------------
    # Tab 8: Segmentation Setting (Demand Space)
    # -------------------------------------------------------------------------
    def _build_tab_seg_setting(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "세그먼트 설정")
        self._register_tab_label(tab, "세그먼트 설정", "Segmentation Setting")
        layout = QtWidgets.QHBoxLayout(tab)

        left = QtWidgets.QVBoxLayout()

        # [v8.1] Enhanced Mode Selection with Detailed Explanations
        mode_box = QtWidgets.QGroupBox("Analysis Mode")
        mlay = QtWidgets.QVBoxLayout(mode_box)
        
        self.cmb_demand_mode = QtWidgets.QComboBox()
        self.cmb_demand_mode.addItems([
            "Segments-as-points (Demographics/Profile based)",
            "Variables-as-points (Columns as points)"
        ])
        self.cmb_demand_mode.currentTextChanged.connect(self._on_demand_mode_changed)
        mlay.addWidget(self.cmb_demand_mode)
        
        # [v8.1] Detailed explanation label
        self.lbl_mode_explain = QtWidgets.QLabel()
        self.lbl_mode_explain.setWordWrap(True)
        self.lbl_mode_explain.setStyleSheet("color: #37474f; background-color: #eceff1; padding: 8px; border-radius: 4px;")
        mlay.addWidget(self.lbl_mode_explain)
        
        left.addWidget(mode_box)

        seg_box = QtWidgets.QGroupBox("Segments-as-points Input")
        seg_l = QtWidgets.QVBoxLayout(seg_box)
        seg_l.addWidget(QtWidgets.QLabel("Select *_seg columns to combine:"))
        self.lst_demand_segcols = QtWidgets.QListWidget()
        self.lst_demand_segcols.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        seg_l.addWidget(self.lst_demand_segcols, 2)

        tgt_header = QtWidgets.QHBoxLayout()
        tgt_header.addWidget(QtWidgets.QLabel("Target Variables (multi-select):"))
        self.chk_demand_targets_all = QtWidgets.QCheckBox("Check/Uncheck All")
        self.chk_demand_targets_all.stateChanged.connect(self._on_toggle_all_demand_targets)
        tgt_header.addStretch(1)
        tgt_header.addWidget(self.chk_demand_targets_all)
        seg_l.addLayout(tgt_header)
        self.lst_demand_targets = QtWidgets.QListWidget()
        self.lst_demand_targets.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_demand_targets.setMaximumHeight(120)
        self.lst_demand_targets.itemChanged.connect(self._on_demand_target_item_changed)
        seg_l.addWidget(self.lst_demand_targets, 1)

        r = QtWidgets.QHBoxLayout()
        self.txt_demand_seg_sep = QtWidgets.QLineEdit("|")
        self.txt_demand_seg_sep.setMaximumWidth(60)
        self.spin_demand_min_n = QtWidgets.QSpinBox()
        self.spin_demand_min_n.setRange(1, 999999)
        self.spin_demand_min_n.setValue(10)
        r.addWidget(QtWidgets.QLabel("Separator"))
        r.addWidget(self.txt_demand_seg_sep)
        r.addSpacing(12)
        r.addWidget(QtWidgets.QLabel("Min N"))
        r.addWidget(self.spin_demand_min_n)
        seg_l.addLayout(r)

        feat = QtWidgets.QHBoxLayout()
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
        self.btn_demand_check_sel = QtWidgets.QPushButton("선택 체크")
        self.btn_demand_uncheck_sel = QtWidgets.QPushButton("선택 해제")
        self.btn_demand_check_sel.clicked.connect(lambda: self._set_checked_for_selected(self.lst_demand_vars, True))
        self.btn_demand_uncheck_sel.clicked.connect(lambda: self._set_checked_for_selected(self.lst_demand_vars, False))
        self._register_text(self.btn_demand_check_sel, "선택 체크", "Check Selected")
        self._register_text(self.btn_demand_uncheck_sel, "선택 해제", "Uncheck Selected")
        vbtn.addWidget(self.btn_demand_check_sel)
        vbtn.addWidget(self.btn_demand_uncheck_sel)
        var_l.addLayout(vbtn)
        left.addWidget(var_box)

        row = QtWidgets.QHBoxLayout()
        self.cmb_demand_coord = QtWidgets.QComboBox()
        self.cmb_demand_coord.addItems(["PCA (Dim1/Dim2)", "MDS (1-corr distance)"])
        self.cmb_demand_cluster = QtWidgets.QComboBox()
        self.cmb_demand_cluster.addItems(["K-Means", "Hierarchical (Ward)"])
        self.spin_demand_k = QtWidgets.QSpinBox()
        self.spin_demand_k.setRange(2, 30)
        self.spin_demand_k.setValue(6)
        self.btn_run_demand = QtWidgets.QPushButton("분석 시작")
        style_button(self.btn_run_demand, level=4)
        self.btn_run_demand.clicked.connect(self._run_demand_space)
        self._register_text(self.btn_run_demand, "분석 시작", "Run Demand Space")

        row.addWidget(QtWidgets.QLabel("Method"))
        row.addWidget(self.cmb_demand_coord)
        row.addWidget(QtWidgets.QLabel("Clustering"))
        row.addWidget(self.cmb_demand_cluster)
        row.addWidget(QtWidgets.QLabel("Clusters (k)"))
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
        """[v8.1] Enhanced with detailed explanations for each mode."""
        txt = self.cmb_demand_mode.currentText()
        seg_mode = txt.startswith("Segments-as-points")

        self.lst_demand_segcols.setEnabled(seg_mode)
        self.txt_demand_seg_sep.setEnabled(seg_mode)
        # Targets are used in both segments-as-points and variables-as-points (multi-target pivot)
        self.lst_demand_targets.setEnabled(True)
        self.spin_demand_min_n.setEnabled(seg_mode)
        self.chk_demand_use_factors.setEnabled(seg_mode)
        self.spin_demand_factor_k.setEnabled(seg_mode)

        self.lst_demand_vars.setEnabled(not seg_mode)
        self.btn_demand_check_sel.setEnabled(not seg_mode)
        self.btn_demand_uncheck_sel.setEnabled(not seg_mode)
        
        # [v8.1] Update explanation text
        if seg_mode:
            self.lbl_mode_explain.setText(
                "<b>Segments-as-points Mode:</b><br>"
                "• Each point = A unique combination of segment values (e.g., 'Male|Young|Urban')<br>"
                "• Uses Factor scores to calculate profile similarity between segments<br>"
                "• Good for: Understanding segment positions in attitudinal/behavioral space<br>"
                "• Output: Clusters of similar consumer segments based on their Factor profiles<br>"
                "• Example: Find which demographic groups have similar attitudes toward brands"
            )
        else:
            self.lbl_mode_explain.setText(
                "<b>Variables-as-points Mode:</b><br>"
                "• Each point = A column/variable from your dataset<br>"
                "• Uses correlation between variables to calculate similarity<br>"
                "• Good for: Discovering variable groupings and data structure<br>"
                "• Output: Clusters of correlated variables (similar to Factor Analysis view)<br>"
                "• Example: Group brand attributes that tend to move together"
            )

# =============================================================================
# app.py (Part 8/8)
# Segmentation Editing, Export, AI Assistant Tabs & Main Entry Point
# =============================================================================

    # -------------------------------------------------------------------------
    # Tab 8 (cont): Demand Space Analysis Logic
    # -------------------------------------------------------------------------
    def _run_demand_space(self):
        try:
            self._ensure_df()
            seg_mode = self.cmb_demand_mode.currentText().startswith("Segments-as-points")
            mode = self.cmb_demand_coord.currentText()
            k = int(self.spin_demand_k.value())
            cluster_method = self.cmb_demand_cluster.currentText()

            def _cluster_labels(xy: np.ndarray) -> np.ndarray:
                if cluster_method.startswith("Hierarchical"):
                    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
                    labels = model.fit_predict(xy)
                else:
                    model = KMeans(n_clusters=k, n_init=10, random_state=42)
                    labels = model.fit_predict(xy)
                return labels + 1

            if seg_mode:
                seg_cols = self._checked_or_selected_items(self.lst_demand_segcols)
                if len(seg_cols) < 1:
                    raise RuntimeError("Select at least 1 *_seg column.")
                sep = self.txt_demand_seg_sep.text().strip() or "|"

                use_factors = bool(self.chk_demand_use_factors.isChecked())
                fac_k = int(self.spin_demand_factor_k.value())
                targets = self.state.demand_targets or self._selected_checked_items(self.lst_demand_targets)
                min_n = int(self.spin_demand_min_n.value())

                targets_used = list(targets)

                prof, feat_cols, labels_by_row = self._build_segment_profiles(
                    seg_cols, sep, use_factors, fac_k, targets, min_n
                )

                X = prof[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                n_segments, n_features = X.shape
                if n_segments < 2:
                    raise RuntimeError(
                        "세그먼트 조합이 1개뿐입니다. 최소 2개 이상의 세그먼트 조합이 있어야 2D 좌표와 클러스터를 계산할 수 있습니다."
                    )

                # Distance on target×segment normalized distribution (R hclust equivalent)
                dist_condensed = pdist(X.values, metric="euclidean")
                if np.allclose(dist_condensed, 0):
                    xy = np.zeros((n_segments, 2))
                    coord_name = "MDS(target×segment) (모든 세그 거리가 0)"
                else:
                    dist_square = squareform(dist_condensed)
                    coord_name = "MDS(target×segment)"
                    if mode.startswith("PCA") and n_features >= 2:
                        pca = PCA(n_components=2, random_state=42)
                        xy = pca.fit_transform(X.values)
                        coord_name = "PCA(target×segment 분포)"
                    else:
                        mds = MDS(
                            n_components=2,
                            dissimilarity="precomputed",
                            random_state=42,
                            n_init=4,
                            max_iter=500,
                        )
                        xy = mds.fit_transform(dist_square)

                ids = prof.index.astype(str).tolist()
                labels = ids[:]
                k = max(2, min(k, len(ids)))

                if cluster_method.startswith("Hierarchical"):
                    Z = linkage(dist_condensed if len(dist_condensed) else np.array([0.0]), method="complete")
                    cl = fcluster(Z, t=k, criterion="maxclust")
                else:
                    km_data = X.values
                    km = KMeans(n_clusters=k, n_init=10, random_state=42)
                    cl = km.fit_predict(km_data) + 1

                xy_df = pd.DataFrame({
                    "id": ids,
                    "label": labels,
                    "x": xy[:, 0],
                    "y": xy[:, 1],
                    "n": prof["n"].values,
                    "cluster_id": cl,
                    "targets_used": [", ".join(targets_used)] * len(ids),
                })
                cl_s = pd.Series(cl, index=ids)

                self.state.demand_mode = "Segments-as-points"
                self.state.demand_xy = xy_df
                self.state.cluster_assign = cl_s
                self.state.cluster_names = {i + 1: f"Cluster {i + 1}" for i in range(k)}
                palette = pal_hex()
                self.state.cluster_colors = {i + 1: palette[i % len(palette)] for i in range(k)}
                self.state.demand_seg_profile = prof
                self.state.demand_seg_components = seg_cols
                self.state.demand_targets = targets_used
                self.state.demand_targets_used = targets_used
                self.state.demand_features_used = feat_cols
                self.state.demand_seg_labels = labels_by_row
                self.state.demand_seg_sep = sep
                self.state.demand_seg_cluster_map = dict(zip(ids, cl))
                self.state.manual_dirty = False

                args = (ids, labels, xy, cl, self.state.cluster_names, self.state.cluster_colors)
                self.plot_preview.set_data(*args)
                self.plot_edit.set_data(*args)
                self._update_cluster_summary()
                self._update_profiler()
                self._set_new_cluster_default_name()

                tgt_txt = ", ".join(targets) if targets else "(none)"
                self.lbl_demand_status.setText(
                    f"Done: {coord_name}, segments={len(ids)}, k={k}, targets=[{tgt_txt}]."
                )
                self._set_status("Demand Space Analysis Completed.")

            else:
                cols = self._checked_or_selected_items(self.lst_demand_vars)
                targets = self._checked_or_selected_items(self.lst_demand_targets)
                if len(cols) < 2:
                    raise RuntimeError("변수를 2개 이상 선택해 주세요.")
                if not targets:
                    raise RuntimeError("타깃 변수를 1개 이상 선택해 주세요. (Variables-as-points도 타깃×변수 피벗 기반)")

                prof, feat_cols = self._build_variable_profiles(cols, targets)
                X = prof[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                n_points, n_features = X.shape
                if n_points < 2:
                    raise RuntimeError("변수 포인트가 1개뿐입니다. 최소 2개 이상이어야 합니다.")

                dist_condensed = pdist(X.values, metric="euclidean")
                if np.allclose(dist_condensed, 0):
                    xy = np.zeros((n_points, 2))
                    coord_name = "MDS(target×variable) (모든 거리가 0)"
                else:
                    dist_square = squareform(dist_condensed)
                    coord_name = "MDS(target×variable)"
                    if mode.startswith("PCA") and n_features >= 2:
                        pca = PCA(n_components=2, random_state=42)
                        xy = pca.fit_transform(X.values)
                        coord_name = "PCA(target×variable 분포)"
                    else:
                        mds = MDS(
                            n_components=2,
                            dissimilarity="precomputed",
                            random_state=42,
                            n_init=4,
                            max_iter=500,
                        )
                        xy = mds.fit_transform(dist_square)

                k = max(2, min(k, xy.shape[0]))
                if cluster_method.startswith("Hierarchical"):
                    Z = linkage(dist_condensed if len(dist_condensed) else np.array([0.0]), method="complete")
                    cl = fcluster(Z, t=k, criterion="maxclust")
                else:
                    km = KMeans(n_clusters=k, n_init=10, random_state=42)
                    cl = km.fit_predict(X.values) + 1

                ids = prof.index.astype(str).tolist()
                labels = ids[:]
                xy_df = pd.DataFrame({
                    "id": ids,
                    "label": labels,
                    "x": xy[:, 0],
                    "y": xy[:, 1],
                    "n": prof["n"].values,
                    "cluster_id": cl,
                })
                cl_s = pd.Series(cl, index=ids)

                self.state.demand_mode = "Variables-as-points"
                self.state.demand_xy = xy_df
                self.state.cluster_assign = cl_s
                self.state.cluster_names = {i + 1: f"Cluster {i + 1}" for i in range(k)}
                palette = pal_hex()
                self.state.cluster_colors = {i + 1: palette[i % len(palette)] for i in range(k)}
                self.state.demand_seg_profile = None
                self.state.demand_targets_used = None
                self.state.manual_dirty = False

                args = (ids, labels, xy, cl, self.state.cluster_names, self.state.cluster_colors)
                self.plot_preview.set_data(*args)
                self.plot_edit.set_data(*args)
                self._update_cluster_summary()
                self._update_profiler()
                self._set_new_cluster_default_name()

                tgt_txt = ", ".join(targets)
                self.lbl_demand_status.setText(f"Done: {coord_name}, vars={len(ids)}, k={k}, targets=[{tgt_txt}].")
                self._set_status("Demand Space (Vars) Analysis Completed.")

        except Exception as e:
            self.state.last_error = str(e)
            show_error(self, "Demand Space Error", e)

    def _encode_features(self, df: pd.DataFrame, feat_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        numeric_parts = []
        feature_names: List[str] = []

        for col in feat_cols:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                numeric_parts.append(pd.to_numeric(series, errors="coerce"))
                feature_names.append(col)
            else:
                dummies = pd.get_dummies(series.astype(str), prefix=col)
                numeric_parts.append(dummies)
                feature_names.extend(list(dummies.columns))

        merged = pd.concat(numeric_parts, axis=1)
        return merged, feature_names

    def _build_segment_profiles(
        self, seg_cols: List[str], sep: str, use_factors: bool, fac_k: int, targets: List[str], min_n: int
    ):
        df_all = self.state.df.copy()

        missing_seg_cols = [c for c in seg_cols if c not in df_all.columns]
        if missing_seg_cols:
            raise RuntimeError(f"다음 세그먼트 컬럼이 없습니다: {', '.join(missing_seg_cols)}")

        seg_labels_full = df_all[seg_cols].astype(str).apply(lambda r: sep.join(r.values), axis=1)
        df_all["_SEG_LABEL_"] = seg_labels_full

        min_n = max(1, int(min_n))
        seg_counts_all = df_all["_SEG_LABEL_"].value_counts()
        keep_labels = seg_counts_all[seg_counts_all >= min_n].index
        df = df_all[df_all["_SEG_LABEL_"].isin(keep_labels)].copy()

        cnt = df["_SEG_LABEL_"].value_counts()
        if cnt.empty:
            raise RuntimeError("No segments found for the selected *_seg columns.")

        # Drop underpowered segment combinations so Min N actually affects the run
        valid_levels = cnt[cnt >= max(1, int(min_n))].index.tolist()
        if not valid_levels:
            raise RuntimeError(f"모든 세그먼트가 Min N({min_n})보다 작습니다. 필터를 낮추거나 세그를 합쳐주세요.")
        df = df[df["_SEG_LABEL_"].isin(valid_levels)].copy()
        cnt = df["_SEG_LABEL_"].value_counts()

        if not targets:
            raise RuntimeError("Target 변수를 1개 이상 선택해 주세요. (Segments-as-points는 타깃×세그 분포 기반)")

        missing = [t for t in targets if t not in df.columns]
        if missing:
            raise RuntimeError(f"다음 타깃 변수가 데이터에 없습니다: {', '.join(missing)}")

        seg_levels = cnt.index.tolist()
        pivot_norms: List[pd.DataFrame] = []
        for tgt in targets:
            pivot = (
                df.assign(_cnt=1)
                .pivot_table(index="_SEG_LABEL_", columns=tgt, values="_cnt", aggfunc="sum", fill_value=0)
                .astype(float)
            )

            # Align all segment columns to the full set and normalize column-wise (distribution vector for each segment)
            pivot = pivot.reindex(columns=seg_levels, fill_value=0.0)
            col_sum = pivot.sum(axis=0).replace(0, np.nan)
            pivot_norm = pivot.divide(col_sum, axis=1).fillna(0.0)
            pivot_norm.index = [f"{tgt}::{idx}" for idx in pivot_norm.index.astype(str)]
            pivot_norms.append(pivot_norm)

        pivot_stack = pd.concat(pivot_norms, axis=0)
        seg_matrix = pivot_stack.T

        if seg_matrix.shape[0] < 2:
            raise RuntimeError("세그먼트 조합이 1개뿐입니다. 최소 2개 이상이어야 합니다.")
        seg_matrix["n"] = seg_matrix.index.map(cnt.get).fillna(0).astype(int)

        # Initialize feature list early so it always exists, even if factor logic is skipped
        feat_cols = [c for c in seg_matrix.columns if c != "n"]

        # Optional PCA/Factor profile mean features (align with R flow: target×seg pivot + PCA profile)
        if use_factors and self.state.factor_scores is not None:
            fac_cols = self._factor_score_columns()
            fac_cols = fac_cols[: max(0, fac_k)]
            if fac_cols:
                fac_df = self.state.factor_scores.reindex(df.index).copy()
                fac_df["_SEG_LABEL_"] = df["_SEG_LABEL_"].values
                fac_mean = fac_df.groupby("_SEG_LABEL_")[fac_cols].mean()
                seg_matrix = seg_matrix.join(fac_mean, how="left")
                seg_matrix[fac_cols] = seg_matrix[fac_cols].fillna(0.0)
                feat_cols = [c for c in seg_matrix.columns if c != "n"]
        labels_by_row = seg_labels_full.copy()
        return seg_matrix, feat_cols, labels_by_row

    def _build_variable_profiles(self, var_cols: List[str], targets: List[str]):
        df = self.state.df.copy()

        if not targets:
            raise RuntimeError("타깃 변수를 1개 이상 선택해 주세요.")
        missing_tgt = [t for t in targets if t not in df.columns]
        if missing_tgt:
            raise RuntimeError(f"다음 타깃 변수가 데이터에 없습니다: {', '.join(missing_tgt)}")

        missing_vars = [c for c in var_cols if c not in df.columns]
        if missing_vars:
            raise RuntimeError(f"다음 변수 컬럼이 없습니다: {', '.join(missing_vars)}")

        long_df = df[targets + var_cols].copy()
        long_df["_cnt"] = 1
        melted = long_df.melt(id_vars=targets, value_vars=var_cols, var_name="_VAR_", value_name="_VAL_")
        melted["_cnt"] = melted["_VAL_"].notna().astype(int)

        pivot_norms: List[pd.DataFrame] = []
        for tgt in targets:
            pivot = (
                melted.pivot_table(
                    index=tgt,
                    columns="_VAR_",
                    values="_cnt",
                    aggfunc="sum",
                    fill_value=0,
                )
                .astype(float)
            )
            pivot = pivot.reindex(columns=var_cols, fill_value=0.0)
            col_sum = pivot.sum(axis=0)
            pivot_norm = pivot.divide(col_sum, axis=1).fillna(0.0)
            pivot_norms.append(pivot_norm)

        if not pivot_norms:
            raise RuntimeError("피벗을 계산할 수 없습니다. 선택된 타깃/변수를 확인하세요.")

        pivot_stack = pd.concat(pivot_norms, axis=0)
        if pivot_stack.shape[1] < 2:
            raise RuntimeError("변수 포인트가 1개뿐입니다. 최소 2개 이상이어야 합니다.")

        var_matrix = pivot_stack.T
        n_counts = melted.groupby("_VAR_")["_cnt"].sum().reindex(var_cols).fillna(0).astype(int)
        var_matrix["n"] = n_counts.values

        feature_cols = [c for c in var_matrix.columns if c != "n"]
        return var_matrix, feature_cols

    def _current_segment_labels(self) -> Optional[pd.Series]:
        if self.state.df is None or not self.state.demand_seg_components:
            return None
        sep = self.state.demand_seg_sep or "|"
        lbl = self.state.df[self.state.demand_seg_components].astype(str).apply(
            lambda r: sep.join(r.values), axis=1
        )
        return lbl

    def _get_demand_seg_labels(self) -> Optional[pd.Series]:
        """Safely return cached demand segment labels or recompute without truth-value checks."""
        if self.state.demand_seg_labels is not None:
            return self.state.demand_seg_labels
        return self._current_segment_labels()

    def _variables_as_matrix(self, cols: List[str]):
        df = to_numeric_df(self.state.df, cols)
        df = df.dropna(axis=0, how="all")
        if df.shape[0] < 5:
            raise RuntimeError("Not enough data rows.")

        df = df.fillna(df.mean())
        V = df.T
        scaler = StandardScaler()
        Vz = scaler.fit_transform(V)
        return Vz, cols

    # -------------------------------------------------------------------------
    # Tab 9: Segmentation Editing
    # -------------------------------------------------------------------------
    def _build_tab_seg_editing(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "세그먼트 편집")
        self._register_tab_label(tab, "세그먼트 편집", "Segmentation Editing")
        layout = QtWidgets.QHBoxLayout(tab)

        # Initialize the editable plot first so dependent controls can reference it safely
        self.plot_edit = DemandClusterPlot(editable=True)
        self.plot_edit.sigClustersChanged.connect(self._on_manual_clusters_changed)
        self.plot_edit.sigCoordsChanged.connect(self._on_manual_coords_changed)
        self.plot_edit.sigSelectionChanged.connect(self._on_plot_selection_changed)

        left = QtWidgets.QVBoxLayout()

        toggle_group = QtWidgets.QGroupBox("Edit Mode (Toggle)")
        tgl_lay = QtWidgets.QVBoxLayout(toggle_group)
        tgl_lay.setContentsMargins(2, 2, 2, 2)
        tgl_lay.setSpacing(2)
        self.radio_edit_points = QtWidgets.QRadioButton("Edit Points (Move/Merge)")
        self.radio_edit_points.setToolTip("Drag points to move/merge. Pan is locked.")
        self.radio_edit_view = QtWidgets.QRadioButton("View/Pan Mode")
        self.radio_edit_view.setToolTip("Drag background to pan. Editing is locked.")
        # Default to Edit mode so drag-to-merge stays enabled on tab open
        self.radio_edit_points.setChecked(True)

        self.radio_edit_points.toggled.connect(self._on_edit_mode_toggled)
        self.radio_edit_view.toggled.connect(self._on_edit_mode_toggled)
        tgl_lay.addWidget(self.radio_edit_points)
        tgl_lay.addWidget(self.radio_edit_view)
        left.addWidget(toggle_group)

        opt_group = QtWidgets.QGroupBox("Point Options")
        olay = QtWidgets.QVBoxLayout(opt_group)
        self.chk_free_move_points = QtWidgets.QCheckBox("Free Move Points (No Snap)")
        self.chk_free_move_points.toggled.connect(lambda v: self.plot_edit.set_free_move_points(v))
        self.chk_show_all_point_labels = QtWidgets.QCheckBox("Show All Labels")
        self.chk_show_all_point_labels.toggled.connect(lambda v: self.plot_edit.set_show_all_point_labels(v))
        self.btn_auto_labels = QtWidgets.QPushButton("라벨 자동 정렬")
        style_button(self.btn_auto_labels, level=1)
        self.btn_auto_labels.clicked.connect(lambda: self.plot_edit.auto_arrange_labels())
        self.btn_reset_label_pos = QtWidgets.QPushButton("라벨 위치 초기화")
        style_button(self.btn_reset_label_pos, level=1)
        self.btn_reset_label_pos.clicked.connect(lambda: self.plot_edit.reset_label_positions())
        self._register_text(self.btn_auto_labels, "라벨 자동 정렬", "Auto-Arrange Labels")
        self._register_text(self.btn_reset_label_pos, "라벨 위치 초기화", "Reset Label Pos")

        olay.addWidget(self.chk_free_move_points)
        olay.addWidget(self.chk_show_all_point_labels)
        olay.addWidget(self.btn_auto_labels)
        olay.addWidget(self.btn_reset_label_pos)
        left.addWidget(opt_group)

        add_group = QtWidgets.QGroupBox("Add Cluster (Drag into empty space)")
        add_lay = QtWidgets.QVBoxLayout(add_group)
        self.chk_add_cluster_mode = QtWidgets.QCheckBox("Enable drag-to-add cluster")
        self.chk_add_cluster_mode.setToolTip(
            "Select points, enable, then drop into an empty area to spawn a new cluster with the chosen name/color."
        )
        self.chk_add_cluster_mode.toggled.connect(self._on_new_cluster_toggle)
        add_lay.addWidget(self.chk_add_cluster_mode)

        name_row = QtWidgets.QHBoxLayout()
        self.txt_new_cluster_name = QtWidgets.QLineEdit(f"Cluster {self._next_cluster_id()}")
        self.txt_new_cluster_name.textChanged.connect(self._on_cluster_name_edited)
        name_row.addWidget(QtWidgets.QLabel("Name"))
        name_row.addWidget(self.txt_new_cluster_name)
        add_lay.addLayout(name_row)

        color_row = QtWidgets.QHBoxLayout()
        self._new_cluster_color_hex = pal_hex()[0]
        self.btn_new_cluster_color = QtWidgets.QPushButton("색상 선택")
        style_button(self.btn_new_cluster_color, level=1)
        self.btn_new_cluster_color.clicked.connect(self._pick_new_cluster_color)
        self._register_text(self.btn_new_cluster_color, "색상 선택", "Pick Color")
        self.lbl_new_cluster_color = QtWidgets.QLabel(self._new_cluster_color_hex)
        color_row.addWidget(QtWidgets.QLabel("Color"))
        color_row.addWidget(self.btn_new_cluster_color)
        color_row.addWidget(self.lbl_new_cluster_color)
        add_lay.addLayout(color_row)

        self.btn_undo_new_cluster = QtWidgets.QPushButton("새 클러스터 되돌리기")
        style_button(self.btn_undo_new_cluster, level=1)
        self.btn_undo_new_cluster.clicked.connect(self._undo_last_cluster_creation)
        self._register_text(self.btn_undo_new_cluster, "새 클러스터 되돌리기", "Undo Last Cluster")
        add_lay.addWidget(self.btn_undo_new_cluster)

        self._update_new_cluster_color_button()
        self._apply_new_cluster_template()
        self._on_new_cluster_toggle(False)
        left.addWidget(add_group)

        save_group = QtWidgets.QGroupBox("세그 결과 저장")
        save_lay = QtWidgets.QVBoxLayout(save_group)
        save_row_top = QtWidgets.QHBoxLayout()
        save_row_bottom = QtWidgets.QHBoxLayout()
        self._seg_save_counter = 1
        self.txt_seg_save_name = QtWidgets.QLineEdit("seg_result_1")
        self.txt_seg_save_name.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.chk_seg_save_cluster_id = QtWidgets.QCheckBox("Also include cluster_id")
        self.chk_seg_save_cluster_id.setChecked(False)
        self.btn_seg_save = QtWidgets.QPushButton("세그 결과 저장")
        style_button(self.btn_seg_save, level=2)
        self.btn_seg_save.clicked.connect(self._save_segmentation_result)
        self._register_text(self.btn_seg_save, "세그 결과 저장", "Save Seg Result")
        save_row_top.addWidget(QtWidgets.QLabel("Column"))
        save_row_top.addWidget(self.txt_seg_save_name, 1)
        save_row_bottom.addWidget(self.chk_seg_save_cluster_id)
        save_row_bottom.addStretch(1)
        save_row_bottom.addWidget(self.btn_seg_save)
        save_lay.addLayout(save_row_top)
        save_lay.addLayout(save_row_bottom)
        left.addWidget(save_group)

        rename_box = QtWidgets.QHBoxLayout()
        self.spin_rename_cluster_id = QtWidgets.QSpinBox()
        self.spin_rename_cluster_id.setRange(1, 999)
        self.txt_rename_cluster = QtWidgets.QLineEdit("Name")
        self.btn_rename_cluster = QtWidgets.QPushButton("이름 변경")
        style_button(self.btn_rename_cluster, level=1)
        self.btn_rename_cluster.clicked.connect(self._rename_cluster)
        self._register_text(self.btn_rename_cluster, "이름 변경", "Rename")
        rename_box.addWidget(QtWidgets.QLabel("ID"))
        rename_box.addWidget(self.spin_rename_cluster_id)
        rename_box.addWidget(self.txt_rename_cluster)
        rename_box.addWidget(self.btn_rename_cluster)
        left.addLayout(rename_box)

        layout.addLayout(left, 1)

        center = QtWidgets.QVBoxLayout()
        center.addWidget(self.plot_edit, 1)
        layout.addLayout(center, 2)

        right = QtWidgets.QVBoxLayout()
        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        summary_widget = QtWidgets.QWidget()
        summary_layout = QtWidgets.QVBoxLayout(summary_widget)
        summary_layout.addWidget(QtWidgets.QLabel("<b>Cluster Summary</b> (Points & Sub-segments)"))
        self.tbl_cluster_summary = DataFrameTable(float_decimals=2)
        self.tbl_cluster_summary.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_cluster_summary.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tbl_cluster_summary.selectionModel().selectionChanged.connect(
            self._on_cluster_summary_selection_changed
        )
        summary_layout.addWidget(self.tbl_cluster_summary, 1)
        self.btn_refresh_cluster_summary = QtWidgets.QPushButton("표 업데이트 (세그 N 갱신)")
        style_button(self.btn_refresh_cluster_summary, level=1)
        self.btn_refresh_cluster_summary.clicked.connect(self._manual_refresh_cluster_table)
        self._register_text(self.btn_refresh_cluster_summary, "표 업데이트 (세그 N 갱신)", "Refresh Table (Update N)")
        summary_layout.addWidget(self.btn_refresh_cluster_summary)
        right_splitter.addWidget(summary_widget)

        profiler_widget = QtWidgets.QWidget()
        profiler_layout = QtWidgets.QVBoxLayout(profiler_widget)
        profiler_layout.addWidget(QtWidgets.QLabel("<b>Smart Segment Profiler (Z-Scores)</b>"))
        self.txt_profile_report = QtWidgets.QTextEdit()
        self.txt_profile_report.setReadOnly(True)
        profiler_layout.addWidget(self.txt_profile_report)
        right_splitter.addWidget(profiler_widget)

        right_splitter.setStretchFactor(0, 2)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([420, 220])

        right.addWidget(right_splitter)
        layout.addLayout(right, 1)

        self._on_edit_mode_toggled()

    def _on_edit_mode_toggled(self):
        is_point_edit = self.radio_edit_points.isChecked()
        self.plot_edit.set_edit_mode_active(is_point_edit)
        self.chk_add_cluster_mode.setEnabled(is_point_edit)
        if not is_point_edit and self.chk_add_cluster_mode.isChecked():
            self.chk_add_cluster_mode.setChecked(False)
        if is_point_edit:
            self._set_status("Edit Mode: Drag points/labels enabled. (Pan locked)")
        else:
            self._set_status("View Mode: Pan/Zoom enabled. (Editing locked)")

    def _sync_clusters_from_edit_plot(self, status_msg: Optional[str] = None) -> bool:
        s = self.plot_edit.get_cluster_series()
        if self.state.demand_xy is None or s is None or s.empty:
            self._set_status("세그먼트 결과가 없습니다. 먼저 Demand Space를 실행하세요.")
            return False
        self.state.cluster_assign = s

        # Bring in any newly created cluster names/colors from the edit plot
        new_clusters = self.plot_edit.consume_new_clusters()
        new_cluster_ids: List[int] = []
        for cid, name, color in new_clusters:
            if name:
                self.state.cluster_names[int(cid)] = name
            if color:
                self.state.cluster_colors[int(cid)] = color
            new_cluster_ids.append(int(cid))

        plot_names = self.plot_edit.get_cluster_names()
        plot_colors = self.plot_edit.get_cluster_colors()
        for cid in s.unique():
            cid_int = int(cid)
            if cid_int not in plot_names:
                plot_names[cid_int] = self.state.cluster_names.get(cid_int, f"Cluster {cid_int}")
        self.state.cluster_names.update(plot_names)
        self.state.cluster_colors.update(plot_colors)

        active_clusters = set(int(x) for x in s.unique())
        self.state.cluster_names = {k: v for k, v in self.state.cluster_names.items() if int(k) in active_clusters}
        self.state.cluster_colors = {k: v for k, v in self.state.cluster_colors.items() if int(k) in active_clusters}

        # 1) Update the plot's backing dataframe with the new cluster ids
        if self.state.demand_xy is not None and "id" in self.state.demand_xy.columns:
            df = self.state.demand_xy.copy()
            mapped = df["id"].astype(str).map(s)
            if "cluster_id" in df.columns:
                mapped = mapped.fillna(df["cluster_id"])
            df["cluster_id"] = mapped.astype(int)
            self.state.demand_xy = df

        # 2) When segments are the points, keep the segment→cluster map and base df in sync
        if self.state.demand_mode.startswith("Segments"):
            self.state.demand_seg_cluster_map = {k: int(v) for k, v in self.state.cluster_assign.items()}

            seg_labels = self.state.demand_seg_labels if self.state.demand_seg_labels is not None else self._current_segment_labels()
            if seg_labels is not None and self.state.df is not None:
                cl_map = {str(k): int(v) for k, v in self.state.cluster_assign.items()}
                df = self.state.df.copy()
                df["demand_seg_label"] = seg_labels.values
                df["demand_cluster_id"] = seg_labels.astype(str).map(cl_map).fillna(-1).astype(int)
                df["demand_cluster_name"] = df["demand_cluster_id"].map(self.state.cluster_names).fillna("")
                self.state.df = df

        self.state.manual_dirty = True
        self.plot_edit.set_cluster_names(self.state.cluster_names)
        self.plot_edit.set_cluster_colors(self.state.cluster_colors)
        self.plot_preview.set_cluster_names(self.state.cluster_names)
        self.plot_preview.set_cluster_colors(self.state.cluster_colors)
        self._update_cluster_summary()
        self._update_profiler()
        self._refresh_demand_preview()
        self._set_new_cluster_default_name()
        if new_cluster_ids:
            cid = new_cluster_ids[-1]
            self.plot_edit.select_cluster(cid)
            self._on_plot_selection_changed(cid)
        if status_msg:
            self._set_status(status_msg)
        return True

    def _update_new_cluster_color_button(self):
        col = getattr(self, "_new_cluster_color_hex", "#ff9800")
        self.lbl_new_cluster_color.setText(col)
        self.btn_new_cluster_color.setStyleSheet(
            f"background-color: {col}; color: #000; border: 1px solid #555;"
        )

    def _next_cluster_id(self) -> int:
        ids: List[int] = []
        if self.state.cluster_assign is not None and not self.state.cluster_assign.empty:
            try:
                ids.extend([int(v) for v in self.state.cluster_assign.unique()])
            except Exception:
                pass
        if self.state.cluster_names:
            try:
                ids.extend([int(k) for k in self.state.cluster_names.keys()])
            except Exception:
                pass
        if self.state.demand_xy is not None and "cluster_id" in self.state.demand_xy.columns:
            try:
                ids.extend([int(v) for v in self.state.demand_xy["cluster_id"].dropna().unique()])
            except Exception:
                pass
        return max(ids) + 1 if ids else 1

    def _is_default_cluster_name(self, name: str) -> bool:
        txt = name.strip()
        if not txt.startswith("Cluster "):
            return False
        tail = txt.replace("Cluster ", "", 1).strip()
        return tail.isdigit()

    def _set_new_cluster_default_name(self, force: bool = False):
        if self._active_cluster_id is not None and not force:
            return
        current = self.txt_new_cluster_name.text().strip()
        if not force and current and not self._is_default_cluster_name(current):
            return
        next_id = self._next_cluster_id()
        self._suppress_cluster_name_update = True
        self.txt_new_cluster_name.setText(f"Cluster {next_id}")
        self._suppress_cluster_name_update = False
        self._apply_new_cluster_template()

    def _apply_new_cluster_template(self):
        name = self.txt_new_cluster_name.text().strip()
        col = getattr(self, "_new_cluster_color_hex", None)
        self.plot_edit.set_new_cluster_template(name, col)

    def _on_cluster_name_edited(self):
        if self._suppress_cluster_name_update:
            return
        name = self.txt_new_cluster_name.text().strip()
        self._apply_new_cluster_template()
        if self.chk_add_cluster_mode.isChecked():
            return
        if self._active_cluster_id is None:
            return
        if not name:
            return
        self.state.cluster_names[int(self._active_cluster_id)] = name
        self.txt_rename_cluster.setText(name)
        self.plot_edit.set_cluster_names(self.state.cluster_names)
        self.plot_preview.set_cluster_names(self.state.cluster_names)
        self._update_cluster_summary()
        self._update_profiler()
        self.state.manual_dirty = True

    def _on_new_cluster_toggle(self, checked: bool):
        self.txt_new_cluster_name.setEnabled(bool(checked))
        self.btn_new_cluster_color.setEnabled(bool(checked))
        self.lbl_new_cluster_color.setEnabled(bool(checked))
        self.plot_edit.set_new_cluster_mode(bool(checked))
        if checked:
            self._active_cluster_id = None
            self.tbl_cluster_summary.clearSelection()
            self._active_cluster_id = None
            self._set_new_cluster_default_name(force=True)
            self._apply_new_cluster_template()

    def _pick_new_cluster_color(self):
        col = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._new_cluster_color_hex), self, "Select Cluster Color")
        if col.isValid():
            self._new_cluster_color_hex = col.name()
            self._update_new_cluster_color_button()
            self._apply_new_cluster_template()
            if self.chk_add_cluster_mode.isChecked():
                return
            if self._active_cluster_id is not None:
                self.state.cluster_colors[int(self._active_cluster_id)] = col.name()
                self.plot_edit.set_cluster_colors(self.state.cluster_colors)
                self.plot_preview.set_cluster_colors(self.state.cluster_colors)
                self._update_cluster_summary()
                self.state.manual_dirty = True

    def _undo_last_cluster_creation(self):
        if not hasattr(self, "plot_edit"):
            return
        cid = self.plot_edit.undo_last_new_cluster()
        if cid is None:
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "되돌릴 새 클러스터가 없습니다.")
            return
        self._sync_clusters_from_edit_plot("새 클러스터 생성을 되돌렸습니다.")
        self._active_cluster_id = None
        self._set_new_cluster_default_name(force=True)

    def _update_cluster_summary(self):
        """[v8.1] Enhanced to show sub-segment details for each cluster."""
        if self.state.demand_xy is None or self.state.cluster_assign is None:
            self.tbl_cluster_summary.set_df(None)
            self._cluster_summary_df = None
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

            for seg in items:
                rows.append({
                    "Cluster ID": int(cid),
                    "Name": names.get(int(cid), f"Cluster {int(cid)}"),
                    "Sub-segment": seg,
                    "Sub-segment N": int(n_map.get(seg, 0)) if n_map else "-",
                    "Cluster Total N": n_sum if n_map else "-",
                    "Points in Cluster": len(items),
                    "Targets Used": ", ".join(self.state.demand_targets_used or self.state.demand_targets or []),
                })
        out = pd.DataFrame(rows).sort_values(["Cluster ID", "Sub-segment"]).reset_index(drop=True)
        self._cluster_summary_df = out
        self.tbl_cluster_summary.set_df(out, max_rows=2000)
        self._apply_cluster_summary_tint(out)

    def _cluster_color_hex(self, cid: int) -> str:
        if cid in (self.state.cluster_colors or {}):
            return self.state.cluster_colors[cid]
        return pal_hex()[(int(cid) - 1) % len(pal_hex())]

    def _apply_cluster_summary_tint(self, df: pd.DataFrame):
        if df is None or df.empty or not hasattr(self, "tbl_cluster_summary"):
            return
        if "Cluster ID" not in df.columns:
            return
        for row in range(self.tbl_cluster_summary.rowCount()):
            try:
                cid = int(df.iloc[row]["Cluster ID"])
            except Exception:
                continue
            color = QtGui.QColor(self._cluster_color_hex(cid))
            color.setAlpha(36)
            brush = QtGui.QBrush(color)
            for col in range(self.tbl_cluster_summary.columnCount()):
                item = self.tbl_cluster_summary.item(row, col)
                if item is not None:
                    item.setBackground(brush)

    def _on_cluster_summary_selection_changed(self, *_args):
        if not hasattr(self, "tbl_cluster_summary"):
            return
        if self.chk_add_cluster_mode.isChecked():
            return
        if self._suppress_summary_selection:
            return

        selected = self.tbl_cluster_summary.selectedItems()
        if not selected:
            self._active_cluster_id = None
            self._set_new_cluster_default_name()
            self._suppress_summary_selection = True
            try:
                self.plot_edit.select_cluster(None)
            finally:
                self._suppress_summary_selection = False
            return

        headers = [self.tbl_cluster_summary.horizontalHeaderItem(c).text() for c in range(self.tbl_cluster_summary.columnCount())]
        try:
            n_idx = headers.index("Sub-segment N")
        except ValueError:
            n_idx = None

        total = 0
        if n_idx is not None:
            for item in selected:
                if item.column() != n_idx:
                    continue
                try:
                    total += int(str(item.text()).replace(",", "").strip())
                except Exception:
                    continue
            if total > 0:
                QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f"Σ N = {total:,}")

        try:
            row = selected[0].row()
            cid_item = self.tbl_cluster_summary.item(row, headers.index("Cluster ID"))
            if cid_item is not None:
                cid = int(cid_item.text())
                self._suppress_summary_selection = True
                try:
                    self.plot_edit.select_cluster(cid)
                finally:
                    self._suppress_summary_selection = False
                self._sync_new_cluster_fields_from_summary(cid)
        except Exception:
            return

    def _on_plot_selection_changed(self, cid: Optional[int]):
        if not hasattr(self, "tbl_cluster_summary"):
            return
        if self.chk_add_cluster_mode.isChecked():
            self._active_cluster_id = None
            self._set_new_cluster_default_name()
            return
        if self._suppress_summary_selection:
            return
        self._suppress_summary_selection = True
        try:
            self.tbl_cluster_summary.clearSelection()
            if cid is None:
                self._active_cluster_id = None
                self._set_new_cluster_default_name()
                return
            headers = [
                self.tbl_cluster_summary.horizontalHeaderItem(c).text()
                for c in range(self.tbl_cluster_summary.columnCount())
            ]
            if "Cluster ID" not in headers:
                return
            cid_col = headers.index("Cluster ID")
            first_item = None
            for row in range(self.tbl_cluster_summary.rowCount()):
                item = self.tbl_cluster_summary.item(row, cid_col)
                if item is None:
                    continue
                try:
                    if int(item.text()) == int(cid):
                        self.tbl_cluster_summary.selectRow(row)
                        if first_item is None:
                            first_item = item
                except Exception:
                    continue
            if first_item is not None:
                self.tbl_cluster_summary.scrollToItem(first_item)
            self._sync_new_cluster_fields_from_summary(int(cid))
        finally:
            self._suppress_summary_selection = False

    def _sync_new_cluster_fields_from_summary(self, cid: int):
        self._active_cluster_id = int(cid)
        name = self.state.cluster_names.get(int(cid), f"Cluster {int(cid)}")
        color = self._cluster_color_hex(int(cid))
        self._suppress_cluster_name_update = True
        self.txt_new_cluster_name.setText(name)
        self._suppress_cluster_name_update = False
        self.spin_rename_cluster_id.setValue(int(cid))
        self.txt_rename_cluster.setText(name)
        self._new_cluster_color_hex = color
        self._update_new_cluster_color_button()
        self._apply_new_cluster_template()

    def _manual_refresh_cluster_table(self):
        synced = self._sync_clusters_from_edit_plot("클러스터 표를 최신 상태로 동기화했습니다.")
        if not synced:
            self._set_status("세그먼트 결과가 없습니다. 먼저 Demand Space를 실행하세요.")
            return
        # Ensure downstream tables/previews are freshly rendered even if the sync produced no status text
        self._update_cluster_summary()
        self._update_profiler()
        self._refresh_demand_preview()

    def _save_segmentation_result(self):
        try:
            self._ensure_df()
            synced = self._sync_clusters_from_edit_plot()
            if not synced:
                raise RuntimeError("세그먼트 결과가 없습니다. 먼저 Demand Space를 실행하세요.")

            col = self.txt_seg_save_name.text().strip()
            if not col:
                raise RuntimeError("저장할 컬럼 이름을 입력하세요.")
            if not col.endswith("_seg"):
                col = f"{col}_seg"

            df = self.state.df.copy()

            out_series = pd.Series(index=df.index, dtype=object)
            id_series = pd.Series(index=df.index, dtype=object)

            name_map = {int(cid): self.state.cluster_names.get(int(cid), f"Cluster {int(cid)}") for cid in self.state.cluster_assign.unique()}

            if self.state.demand_mode.startswith("Segments"):
                seg_labels = self._get_demand_seg_labels()
                if seg_labels is None:
                    raise RuntimeError("세그먼트 라벨을 찾을 수 없습니다. 세그먼트 설정을 확인하세요.")
                seg_labels = seg_labels.astype(str)
                cl_map = {str(k): int(v) for k, v in self.state.cluster_assign.items()}
                cl_ids = seg_labels.map(cl_map)
                out_series = seg_labels
                id_series = cl_ids.fillna(-1).astype(int)
                df["Sub-segment"] = seg_labels.values
            elif "id" in df.columns:
                id_map = {str(k): int(v) for k, v in self.state.cluster_assign.items()}
                cl_ids = df["id"].astype(str).map(id_map)
                out_series = cl_ids.map(name_map).fillna("")
                id_series = cl_ids.fillna(-1).astype(int)
            else:
                raise RuntimeError("세그 결과를 원본 데이터에 매핑할 수 없습니다.")

            df[col] = out_series
            if self.chk_seg_save_cluster_id.isChecked():
                df[f"{col}_id"] = id_series
            self.state.df = df
            self.tbl_preview.set_df(df)
            self._refresh_all_column_lists()
            self._update_cluster_summary()
            self._update_profiler()
            self._refresh_demand_preview()

            self._seg_save_counter = getattr(self, "_seg_save_counter", 1) + 1
            self.txt_seg_save_name.setText(f"seg_result_{self._seg_save_counter}")
            self._set_status(f"세그 결과를 '{col}' 컬럼에 저장했습니다.")
        except Exception as e:
            show_error(self, "Save Seg Result Error", e)

    def _update_profiler(self):
        """Calculates Z-scores for each cluster to find distinctive features."""
        if self.state.df is None or self.state.cluster_assign is None:
            return

        if self.state.demand_mode.startswith("Segments") and self.state.demand_seg_profile is not None:
            prof = self.state.demand_seg_profile.copy()
            prof["Cluster"] = self.state.cluster_assign

            feats = self.state.demand_features_used
            if not feats:
                return

            g_mean = prof[feats].mean()
            g_std = prof[feats].std()

            c_means = prof.groupby("Cluster")[feats].mean()

            z_scores = (c_means - g_mean) / g_std

            report = ""
            for cid in z_scores.index:
                cname = self.state.cluster_names.get(cid, f"Cluster {cid}")
                report += f"<b>=== {cname} ===</b><br>"

                row = z_scores.loc[cid]
                high = row[row > 1.0].sort_values(ascending=False).head(5)
                if not high.empty:
                    desc = ", ".join([f"{k}(<font color='green'>+{v:.2f}</font>)" for k, v in high.items()])
                    report += f"<b>[HIGH]</b>: {desc}<br>"

                low = row[row < -1.0].sort_values().head(5)
                if not low.empty:
                    desc = ", ".join([f"{k}(<font color='red'>{v:.2f}</font>)" for k, v in low.items()])
                    report += f"<b>[LOW]</b>: {desc}<br>"

                if high.empty and low.empty:
                    report += "<i>(Average profile - no distinctive features)</i><br>"
                report += "<br>"

            self.txt_profile_report.setHtml(report)
        else:
            report = "<b>Variables-as-points Mode:</b><br><br>"
            cl = self.state.cluster_assign
            for cid in sorted(cl.unique()):
                cname = self.state.cluster_names.get(cid, f"Cluster {cid}")
                vars_in = cl[cl == cid].index.tolist()
                report += f"<b>=== {cname} ===</b><br>"
                report += ", ".join(vars_in) + "<br><br>"
            self.txt_profile_report.setHtml(report)

    def _on_manual_clusters_changed(self):
        self._sync_clusters_from_edit_plot("Manual clusters updated.")

    def _on_manual_coords_changed(self):
        if self.state.demand_xy is None:
            return
        xy_map = self.plot_edit.get_xy_map()
        try:
            df = self.state.demand_xy.copy()
            if "id" in df.columns:
                df["x"] = df["id"].astype(str).map(lambda k: xy_map.get(str(k), (np.nan, np.nan))[0])
                df["y"] = df["id"].astype(str).map(lambda k: xy_map.get(str(k), (np.nan, np.nan))[1])
                self.state.demand_xy = df
                self.state.manual_dirty = True
                self._refresh_demand_preview()
                self._set_status("Manual coords updated.")
        except Exception:
            self._set_status("Manual coords update failed.")

    def _refresh_demand_preview(self):
        if self.state.demand_xy is None or self.state.cluster_assign is None:
            return
        df = self.state.demand_xy
        if "id" not in df.columns or "x" not in df.columns or "y" not in df.columns:
            return

        ids = df["id"].astype(str).tolist()
        labels = df["label"].astype(str).tolist() if "label" in df.columns else ids
        xy = df[["x", "y"]].to_numpy()

        cl = self.state.cluster_assign.reindex(ids)
        if cl.isna().any():
            fallback = cl.dropna().mode()
            default_cl = int(fallback.iloc[0]) if not fallback.empty else 1
            cl = cl.fillna(default_cl)

        clusters = cl.astype(int).to_numpy()
        self.plot_preview.set_data(
            ids,
            labels,
            xy,
            clusters,
            self.state.cluster_names,
            self.state.cluster_colors,
        )

    def _rename_cluster(self):
        try:
            cid = int(self.spin_rename_cluster_id.value())
            name = self.txt_rename_cluster.text().strip()
            if not name:
                raise RuntimeError("Enter a name.")
            self.state.cluster_names[cid] = name

            self.plot_edit.set_cluster_names(self.state.cluster_names)
            self.plot_preview.set_cluster_names(self.state.cluster_names)
            self._update_cluster_summary()
            self._update_profiler()
            self.state.manual_dirty = True
        except Exception as e:
            show_error(self, "Rename Error", e)

    # -------------------------------------------------------------------------
    # Tab 10: Export
    # -------------------------------------------------------------------------
    def _build_tab_export(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "내보내기")
        self._register_tab_label(tab, "내보내기", "Export")
        layout = QtWidgets.QVBoxLayout(tab)

        self.lbl_export = QtWidgets.QLabel(
            "Export Results to Excel:\n"
            "Sheets: 00_Metadata, 01_Data, 02_RECODE, 03_Factor_Loadings, 04_Factor_Scores,\n"
            "05_DT_ImprovePivot, 05_DT_Importance, 06_DT_BestSplit, 07_DT_Full_Nodes, ...\n"
            "13_Demand_Clusters, 14_Variable_Types, 15_Raw_with_Clusters"
        )
        self.lbl_export.setWordWrap(True)
        layout.addWidget(self.lbl_export)

        row = QtWidgets.QHBoxLayout()
        self.txt_export_path = QtWidgets.QLineEdit()
        self.btn_export_browse = QtWidgets.QPushButton("찾기")
        style_button(self.btn_export_browse, level=1)
        self.btn_export_browse.clicked.connect(self._browse_export_path)
        self.btn_export = QtWidgets.QPushButton("엑셀 내보내기")
        style_button(self.btn_export, level=2)
        self.btn_export.clicked.connect(self._export_excel)
        self._register_text(self.btn_export_browse, "찾기", "Browse...")
        self._register_text(self.btn_export, "엑셀 내보내기", "Export to Excel")

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
            default = f"{base}_AutoSegmentTool_v81.xlsx"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Excel", default, "Excel (*.xlsx)")
        if path:
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            self.txt_export_path.setText(path)

    def _export_excel(self):
        try:
            self._ensure_df()
            out = self.txt_export_path.text().strip()
            if not out:
                raise RuntimeError("Select output path.")

            with pd.ExcelWriter(out, engine="openpyxl") as w:
                metadata_rows = []
                tgt_list = self.state.demand_targets_used or self.state.demand_targets
                if tgt_list:
                    metadata_rows.append({"key": "demand_targets_used", "value": ", ".join(tgt_list)})
                if metadata_rows:
                    pd.DataFrame(metadata_rows).to_excel(w, sheet_name="00_Metadata", index=False)

                self.state.df.to_excel(w, sheet_name="01_Data", index=False)
                if self.state.recode_df is not None:
                    self.state.recode_df.to_excel(w, sheet_name="02_RECODE", index=False)

                if self.state.factor_loadings is not None:
                    self.state.factor_loadings.reset_index().rename(columns={"index": "variable"}).to_excel(w, sheet_name="03_Factor_Loadings", index=False)
                if self.state.factor_scores is not None:
                    self.state.factor_scores.reset_index().rename(columns={"index": "row_index"}).to_excel(w, sheet_name="04_Factor_Scores", index=False)

                if self.state.dt_improve_pivot is not None:
                    self.state.dt_improve_pivot.to_excel(w, sheet_name="05_DT_ImprovePivot", index=False)
                if getattr(self.state, "dt_importance_summary", None) is not None:
                    self.state.dt_importance_summary.to_excel(w, sheet_name="05_DT_Importance", index=False)
                if self.state.dt_split_best is not None:
                    self.state.dt_split_best.to_excel(w, sheet_name="06_DT_BestSplit", index=False)

                if self.state.dt_full_nodes is not None:
                    self.state.dt_full_nodes.to_excel(w, sheet_name="07_DT_Full_Nodes", index=False)
                if self.state.dt_full_split_groups is not None:
                    self.state.dt_full_split_groups.to_excel(w, sheet_name="07_DT_Split_Groups", index=False)
                if self.state.dt_full_path_info is not None:
                    self.state.dt_full_path_info.to_excel(w, sheet_name="07_DT_Path_Info", index=False)
                if getattr(self.state, "dt_full_split_paths", None) is not None:
                    self.state.dt_full_split_paths.to_excel(w, sheet_name="07_DT_Variable_Paths", index=False)

                if self.state.demand_xy is not None:
                    self.state.demand_xy.to_excel(w, sheet_name="12_Demand_Coords", index=False)

                if self.state.cluster_assign is not None:
                    cl_df = self.state.cluster_assign.reset_index()
                    cl_df.columns = ["id", "cluster_id"]
                    cl_df["cluster_name"] = cl_df["cluster_id"].map(self.state.cluster_names).fillna("")
                    cl_df.to_excel(w, sheet_name="13_Demand_Clusters", index=False)

                if self.state.demand_mode.startswith("Segments") and self.state.cluster_assign is not None:
                    seg_labels = self._get_demand_seg_labels()
                    if seg_labels is not None:
                        cl_map = {str(k): int(v) for k, v in self.state.cluster_assign.items()}
                        raw = self.state.df.copy()
                        raw["demand_seg_label"] = seg_labels.values
                        raw["demand_cluster_id"] = seg_labels.astype(str).map(cl_map).fillna(-1).astype(int)
                        raw["demand_cluster_name"] = raw["demand_cluster_id"].map(self.state.cluster_names).fillna("")
                        raw.to_excel(w, sheet_name="15_Raw_with_Clusters", index=False)

                # [v8.1] Export variable types
                if self.state.var_types:
                    vt_df = pd.DataFrame([
                        {"variable": k, "type": v} for k, v in self.state.var_types.items()
                    ])
                    vt_df.to_excel(w, sheet_name="14_Variable_Types", index=False)

            self.lbl_export_status.setText(f"Exported successfully to {out}")
            self._set_status("Export Done.")
        except Exception as e:
            self.state.last_error = str(e)
            show_error(self, "Export Error", e)

    # -------------------------------------------------------------------------
    # Tab 11: AI Assistant (RAG Chatbot) with Retry Logic
    # -------------------------------------------------------------------------
    def _build_tab_rag(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "AI 어시스턴트 (RAG)")
        self._register_tab_label(tab, "AI 어시스턴트 (RAG)", "AI Assistant (RAG)")
        layout = QtWidgets.QVBoxLayout(tab)

        layout.addWidget(QtWidgets.QLabel("<b>AI Assistant</b>: Ask questions about your current data/analysis."))

        # [v8.1] Enhanced API Key section with status
        key_box = QtWidgets.QGroupBox("AI API Configuration (GPT / Gemini)")
        key_layout = QtWidgets.QVBoxLayout(key_box)

        key_row = QtWidgets.QHBoxLayout()
        self.cmb_ai_provider = QtWidgets.QComboBox()
        self.cmb_ai_provider.addItem("OpenAI (기본: GPT-4o-mini)", "openai")
        self.cmb_ai_provider.addItem("Gemini (기본: 3-pro-preview)", "gemini")

        self.txt_openai_key = QtWidgets.QLineEdit()
        self.txt_openai_key.setPlaceholderText("Enter OpenAI API Key (sk-...) or leave empty to generate prompt only")
        self.txt_openai_key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)

        self.txt_gemini_key = QtWidgets.QLineEdit()
        self.txt_gemini_key.setPlaceholderText("Enter Gemini API Key (AI Studio) or leave empty to generate prompt only")
        self.txt_gemini_key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)

        key_row.addWidget(QtWidgets.QLabel("Provider:"))
        key_row.addWidget(self.cmb_ai_provider)
        key_row.addWidget(QtWidgets.QLabel("OpenAI Key:"))
        key_row.addWidget(self.txt_openai_key, 1)
        key_row.addWidget(QtWidgets.QLabel("Gemini Key:"))
        key_row.addWidget(self.txt_gemini_key, 1)
        key_layout.addLayout(key_row)
        
        # [v8.1] API status and retry info
        self.lbl_api_status = QtWidgets.QLabel(
            "<i>Rate limit handling: Auto-retry up to 3 times with exponential backoff (2s → 4s → 8s)</i>"
        )
        self.lbl_api_status.setStyleSheet("color: #666;")
        key_layout.addWidget(self.lbl_api_status)
        
        layout.addWidget(key_box)

        self.txt_chat_history = QtWidgets.QTextEdit()
        self.txt_chat_history.setReadOnly(True)
        layout.addWidget(self.txt_chat_history, 1)

        input_row = QtWidgets.QHBoxLayout()
        self.txt_user_query = QtWidgets.QLineEdit()
        self.txt_user_query.setPlaceholderText("Ex: 'Interpret the Factor 1 loadings' or 'Why did the decision tree select Age?'")
        self.txt_user_query.returnPressed.connect(self._send_rag_query)

        self.btn_send_query = QtWidgets.QPushButton("질문 전송")
        style_button(self.btn_send_query, level=2)
        self.btn_send_query.clicked.connect(self._send_rag_query)
        self._register_text(self.btn_send_query, "질문 전송", "Send / Gen Prompt")

        input_row.addWidget(self.txt_user_query, 3)
        input_row.addWidget(self.btn_send_query)
        layout.addLayout(input_row)

    def _get_rag_context(self) -> str:
        ctx = ["=== SYSTEM CONTEXT ==="]
        if self.state.df is not None:
            ctx.append(f"Data Loaded: {len(self.state.df)} rows, {len(self.state.df.columns)} cols.")
            ctx.append(f"Columns: {', '.join(list(self.state.df.columns)[:50])} ...")
            
            # [v8.1] Include variable type info
            if self.state.var_types:
                ctx.append(f"Custom Variable Types: {len(self.state.var_types)} overrides")
        else:
            ctx.append("Data: None loaded.")

        if self.state.factor_loadings is not None:
            ctx.append(f"\n[Factor Analysis ({self.state.factor_mode})]")
            ctx.append("Top Loadings:")
            try:
                top = self.state.factor_loadings.head(10).to_string()
                ctx.append(top)
            except:
                pass

        if self.state.dt_split_best is not None:
            ctx.append(f"\n[Decision Tree Best Splits (Top 5 Improvement)]")
            try:
                top_splits = self.state.dt_split_best.sort_values("improve_rel", ascending=False).head(5)
                for _, row in top_splits.iterrows():
                    ctx.append(f"- Dep: {row['dep']}, Ind: {row['ind']}, Type: {row['split_type']}, Imp: {row['improve_rel']:.4f}")
            except:
                pass

        if self.state.last_error:
            ctx.append(f"\n[Recent Error Log]\n{self.state.last_error}")

        return "\n".join(ctx)

    def _send_rag_query(self):
        query = self.txt_user_query.text().strip()
        if not query:
            return

        context = self._get_rag_context()
        full_prompt = f"{context}\n\n=== USER QUESTION ===\n{query}\n\n=== INSTRUCTION ===\nAnswer based on the context above. If error log exists, suggest a fix."

        self.txt_chat_history.append(f"<b>User:</b> {query}")
        self.txt_user_query.clear()

        provider, api_key = self._get_ai_provider_and_key()
        provider_label = "Gemini" if provider == "gemini" else "ChatGPT"
        if not api_key:
            self.txt_chat_history.append(
                f"<br><i>[System] No API Key provided. Copy this prompt to {provider_label}:</i><br>"
            )
            self.txt_chat_history.append(f"<pre style='background:#f5f5f5; padding:10px;'>{full_prompt}</pre><br><hr>")
            return

        try:
            self.txt_chat_history.append("<i>... Thinking (with auto-retry on rate limit) ...</i>")
            self.lbl_api_status.setText(
                f"<b style='color:orange;'>⏳ Sending request via {provider_label}...</b>"
            )
            QtWidgets.QApplication.processEvents()

            # [v8.1] Use new retry-enabled API call
            messages = [
                {"role": "system", "content": "You are a Data Analysis Assistant for market research."},
                {"role": "user", "content": full_prompt}
            ]
            success, result = call_ai_chat(
                provider,
                api_key,
                messages,
                max_retries=3,
                initial_delay=2.0,
                timeout=30,
            )

            if success:
                self.txt_chat_history.append(f"<b>AI:</b> {result}<br><hr>")
                self.lbl_api_status.setText("<b style='color:green;'>✓ Response received</b>")
            else:
                self.txt_chat_history.append(f"<font color='red'><b>Error:</b> {result}</font><br><hr>")
                self.lbl_api_status.setText(f"<b style='color:red;'>✗ {result[:50]}...</b>")

        except Exception as e:
            self.txt_chat_history.append(f"<font color='red'>Unexpected Error: {str(e)}</font><br>")
            self.lbl_api_status.setText(f"<b style='color:red;'>✗ Error</b>")


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    font_path = resource_path("Pretendard-Regular.ttf")
    if not os.path.exists(font_path):
        font_path = str(Path(__file__).resolve().parent / "Pretendard-Regular.ttf")
    if os.path.exists(font_path):
        font_id = QtGui.QFontDatabase.addApplicationFont(font_path)
        if font_id != -1:
            families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
            if families:
                app.setFont(QtGui.QFont(families[0], 10))
    win = IntegratedApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
