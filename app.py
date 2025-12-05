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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import requests
from scipy.stats import zscore

from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

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
    """Auto-detect if series is categorical based on dtype and unique count."""
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
            if n in col_lut:
                return col_lut[n]
        return None

    q = pick("question", "문항", "문항명", "q", "item", "variable")
    c = pick("code", "코드", "값", "value", "val")
    n = pick("name", "라벨", "label", "명", "설명", "text")

    if q is None or c is None or n is None:
        if len(cols) >= 3:
            q, c, n = cols[0], cols[1], cols[2]

    out = df.copy()
    rename_map = {}
    if q:
        rename_map[q] = "QUESTION"
    if c:
        rename_map[c] = "CODE"
    if n:
        rename_map[n] = "NAME"
    out = out.rename(columns=rename_map)

    for cc in ["QUESTION", "CODE", "NAME"]:
        if cc not in out.columns:
            out[cc] = np.nan

    out["QUESTION"] = out["QUESTION"].astype(str).str.strip()
    out["CODE"] = out["CODE"].astype(str).str.strip()
    out["NAME"] = out["NAME"].astype(str).str.strip()
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

    # [v8.1 NEW] Variable Type Overrides: {column_name: VAR_TYPE_NUMERIC | VAR_TYPE_CATEGORICAL | VAR_TYPE_AUTO}
    var_types: Dict[str, str] = field(default_factory=dict)

    # Factor Analysis Data (PCA or EFA)
    factor_model: Any = None
    factor_cols: Optional[List[str]] = None
    factor_scores: Optional[pd.DataFrame] = None
    factor_loadings: Optional[pd.DataFrame] = None
    factor_mode: str = "PCA"

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
def call_openai_api(
    api_key: str,
    messages: List[dict],
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
    initial_delay: float = 2.0,
    timeout: int = 30
) -> Tuple[bool, str]:
    """
    Calls OpenAI API with exponential backoff retry logic.
    
    Returns:
        (success: bool, result: str)
        - If success: result is the AI response content
        - If failure: result is the error message
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7
    }

    delay = initial_delay
    last_error = ""

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout
            )

            # Check for rate limit (429)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    wait_time = float(retry_after)
                else:
                    wait_time = delay
                
                last_error = f"Rate limit hit. Waiting {wait_time:.1f}s... (Attempt {attempt + 1}/{max_retries})"
                time.sleep(wait_time)
                delay *= 2  # Exponential backoff
                continue

            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            return True, answer

        except requests.exceptions.Timeout:
            last_error = f"Request timeout after {timeout}s (Attempt {attempt + 1}/{max_retries})"
            time.sleep(delay)
            delay *= 2

        except requests.exceptions.HTTPError as e:
            if "429" in str(e):
                last_error = f"Rate limit exceeded. Please wait and try again."
                time.sleep(delay)
                delay *= 2
            else:
                return False, f"HTTP Error: {str(e)}"

        except requests.exceptions.RequestException as e:
            return False, f"Network Error: {str(e)}"

        except Exception as e:
            return False, f"Unexpected Error: {str(e)}"

    return False, f"Failed after {max_retries} attempts. Last error: {last_error}"
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
        self.setWindowTitle("Variable Type Manager (SPSS Style)")
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
            if shift:
                self.plot.remember_label_position(self.cluster_id, (float(pos_view.x()), float(pos_view.y())))
                return
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

        self._scatter = pg.ScatterPlotItem(size=11, pxMode=True)
        self.addItem(self._scatter)

        self._point_text_items: List[pg.TextItem] = []
        self._hull_items: Dict[int, QtWidgets.QGraphicsPathItem] = {}
        self._label_items: Dict[int, DraggableClusterLabel] = {}

        self._selected: set[int] = set()
        self._dragging = False
        self._drag_temp_positions: Optional[np.ndarray] = None
        self._drag_anchor_xy: Optional[Tuple[float, float]] = None

        self._free_move_points: bool = False
        self._show_all_point_labels: bool = False

        self._label_pos_override: Dict[int, Tuple[float, float]] = {}

    def set_edit_mode_active(self, active: bool):
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

        self._label_pos_override.clear()

        self.redraw_all()

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
            self.remember_label_position(src_cluster, drop_xy)
            self._draw_hulls_and_labels()
            return

        dst = int(best)
        self._cluster[self._cluster == int(src_cluster)] = dst

        if int(src_cluster) in self._label_pos_override:
            self._label_pos_override.pop(int(src_cluster), None)

        self.redraw_all()
        self.sigClustersChanged.emit()


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

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab Initialization
        self._build_tab_data()
        self._build_tab_recode()
        self._build_tab_factor()
        self._build_tab_dt_setting()
        self._build_tab_dt_results()
        self._build_tab_grouping()
        self._build_tab_seg_setting()
        self._build_tab_seg_editing()
        self._build_tab_export()
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

        self.cmb_dep_extra.clear()
        self.cmb_dep_extra.addItem("(None)")
        self.cmb_dep_extra.addItems(cols)

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

        self.cmb_demand_target.clear()
        self.cmb_demand_target.addItem("(None)")
        self.cmb_demand_target.addItems(cols)

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

                if self.state.df is not None:
                    self.tbl_preview.set_df(self.state.df)
                    self._refresh_all_column_lists()

                if self.state.recode_df is not None:
                    self._update_recode_tab()

                if self.state.factor_loadings is not None:
                    self.tbl_factor_loadings.set_df(self.state.factor_loadings.reset_index())

                if self.state.dt_improve_pivot is not None:
                    self.tbl_dt_pivot.set_df(self.state.dt_improve_pivot)

                self.statusBar().showMessage(f"Project loaded from {path}")
            except Exception as e:
                show_error(self, "Load Error", e)

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

        # [v8.1] Variable Type Manager Button in Data Tab
        row2 = QtWidgets.QHBoxLayout()
        self.btn_var_type_mgr = QtWidgets.QPushButton("📊 Variable Type Manager (Numeric/Categorical)")
        style_button(self.btn_var_type_mgr, level=3)
        self.btn_var_type_mgr.clicked.connect(self._open_variable_type_manager)
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

            # Load RECODE sheet if exists
            xls = pd.ExcelFile(path, engine="openpyxl")
            if "RECODE" in xls.sheet_names:
                rec = pd.read_excel(path, sheet_name="RECODE", engine="openpyxl")
                self.state.recode_df = normalize_recode_df(rec)
            else:
                self.state.recode_df = None

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

    def _reset_downstream_state(self):
        self.state.factor_model = None
        self.state.factor_cols = None
        self.state.factor_scores = None
        self.state.factor_loadings = None
        self.state.dt_improve_pivot = None
        self.state.dt_split_best = None
        self.state.dt_full_nodes = None
        
        if hasattr(self, "tbl_factor_loadings"):
            self.tbl_factor_loadings.set_df(None)
        if hasattr(self, "tbl_dt_pivot"):
            self.tbl_dt_pivot.set_df(None)

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
# =============================================================================
# app.py (Part 6/8)
# Factor Analysis (AI Naming) & Decision Tree Setting Tabs
# =============================================================================

    # -------------------------------------------------------------------------
    # Tab 3: Factor Analysis (PCA / EFA) + AI Auto-Naming
    # -------------------------------------------------------------------------
    def _build_tab_factor(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Factor Analysis")

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
        left.addWidget(self.lst_factor_cols, 1)

        # Selection Buttons
        btnrow = QtWidgets.QHBoxLayout()
        self.btn_fac_check_sel = QtWidgets.QPushButton("Check Selected")
        style_button(self.btn_fac_check_sel, level=1)
        self.btn_fac_uncheck_sel = QtWidgets.QPushButton("Uncheck Selected")
        style_button(self.btn_fac_uncheck_sel, level=1)
        self.btn_fac_check_all = QtWidgets.QPushButton("Check All Numeric")
        style_button(self.btn_fac_check_all, level=1)
        self.btn_fac_uncheck_all = QtWidgets.QPushButton("Uncheck All")
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
        self.btn_run_factor = QtWidgets.QPushButton("Run Analysis")
        style_button(self.btn_run_factor, level=2)
        self.btn_run_factor.clicked.connect(self._run_factor_analysis)

        self.btn_ai_name = QtWidgets.QPushButton("AI Auto-Name Factors")
        self.btn_ai_name.clicked.connect(self._ai_name_factors)
        style_button(self.btn_ai_name, 3)

        ctrl.addWidget(QtWidgets.QLabel("Number of Factors (k):"))
        ctrl.addWidget(self.spin_factor_k)
        ctrl.addWidget(self.btn_run_factor)
        ctrl.addWidget(self.btn_ai_name)
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

            disp = loadings.copy()
            disp["_maxabs_"] = disp.abs().max(axis=1)
            disp = disp.sort_values("_maxabs_", ascending=False).drop(columns=["_maxabs_"])
            disp = disp.reset_index().rename(columns={"index": "variable"})

            self.lbl_factor_info.setText(info_text)
            self.tbl_factor_loadings.set_df(disp)

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

    def _ai_name_factors(self):
        """Uses OpenAI API to rename factors with retry logic."""
        if self.state.factor_loadings is None:
            QtWidgets.QMessageBox.warning(self, "No Factors", "Run Factor Analysis first.")
            return
            
        key = self.txt_openai_key.text().strip()
        if not key:
            QtWidgets.QMessageBox.warning(self, "No Key", "Enter API Key in AI Tab.")
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
            success, result = call_openai_api(key, messages, max_retries=3, initial_delay=2.0)

            if success:
                QtWidgets.QMessageBox.information(self, "AI Suggestion", result)
                self._set_status("AI Naming Done.")
            else:
                QtWidgets.QMessageBox.warning(self, "AI Error", result)
                self._set_status("AI Naming Failed.")

        except Exception as e:
            show_error(self, "AI Error", e)

    # -------------------------------------------------------------------------
    # Tab 4: Decision Tree Setting
    # -------------------------------------------------------------------------
    def _build_tab_dt_setting(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Decision Tree Setting")

        layout = QtWidgets.QVBoxLayout(tab)

        # [v8.1] Enhanced Header with Variable Type Info
        head = QtWidgets.QLabel(
            "<b>Decision Tree Analysis</b><br>"
            "1. Select Dependent(Target) & Independent(Predictors) variables.<br>"
            "2. Click 'Run Analysis' to generate Improvement Pivot.<br>"
            "3. Select a cell in Pivot and click 'Recommend Grouping' to auto-create segments.<br><br>"
            "<i>Note: <span style='background-color:#fff3e0;'>Orange</span> = Categorical (Optimal Subset Split), "
            "White = Numeric (Threshold Split). Use Variable Type Manager to customize.</i>"
        )
        head.setWordWrap(True)
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

        # Controls: Predictors (Whitelist)
        pred_box = QtWidgets.QGroupBox("Select Predictors (Independent Variables)")
        pred_layout = QtWidgets.QVBoxLayout(pred_box)

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
        """Calculates Improve Pivot and Best Split tables with variable type awareness."""
        try:
            self._ensure_df()
            df = self.state.df

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

            ind_vars = self._selected_checked_items(self.lst_dt_predictors)
            ind_vars = [c for c in ind_vars if c not in deps and c != "resp_id"]

            if len(ind_vars) == 0:
                raise RuntimeError("No independent variables selected. Please check predictors.")

            best_rows = []
            pivot = pd.DataFrame(index=ind_vars, columns=deps, dtype=float)

            for dep in deps:
                y = df[dep]
                # Determine task type for dependent variable
                is_factor = str(dep).startswith("Factor")
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

            self.tbl_dt_pivot.set_df(pivot_reset)
            self.tbl_dt_bestsplit.set_df(best_df)

            self.state.dt_improve_pivot = pivot_reset
            self.state.dt_split_best = best_df

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
        """Auto-generates grouping mapping based on the best split for selected Ind."""
        try:
            sel_rows = self.tbl_dt_pivot.selectedItems()
            if not sel_rows:
                raise RuntimeError("Please select a row (Predictor) in the Pivot table.")

            row_idx = sel_rows[0].row()
            ind_val = self.tbl_dt_pivot.item(row_idx, 0).text()

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

            rec_name = {}
            if self.state.recode_df is not None:
                r = self.state.recode_df
                r = r[r["QUESTION"].astype(str).str.strip() == ind_val]
                rec_name = dict(zip(r["CODE"].astype(str).str.strip(), r["NAME"].astype(str).str.strip()))

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

            self.tabs.setCurrentIndex(5)
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
        bl.addWidget(QtWidgets.QLabel("Split Detail:"))
        self.lbl_split_imp = QtWidgets.QLabel("No split selected.")
        self.lbl_split_imp.setWordWrap(True)
        bl.addWidget(self.lbl_split_imp)
        self.tbl_split_detail = DataFrameTable(float_decimals=2)
        bl.addWidget(self.tbl_split_detail, 1)
        splitter.addWidget(botw)

        splitter.setSizes([400, 300])
        layout.addWidget(splitter, 1)

    def _compute_full_tree_internal(self, dep: str, ind: str):
        df = self.state.df
        if df is None:
            raise RuntimeError("No data.")

        if dep not in df.columns or ind not in df.columns:
            raise RuntimeError("Columns missing.")

        is_factor = str(dep).startswith("Factor") or str(dep).startswith("PCA")
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
            is_factor = str(dep).startswith("Factor") or str(dep).startswith("PCA")
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
        self.txt_bin_val1 = QtWidgets.QLineEdit("")
        self.txt_bin_lab1 = QtWidgets.QLineEdit("A")
        self.txt_bin_val2 = QtWidgets.QLineEdit("")
        self.txt_bin_lab2 = QtWidgets.QLineEdit("B")
        self.chk_bin_else_other = QtWidgets.QCheckBox("Else=Other")
        self.txt_bin_else_lab = QtWidgets.QLineEdit("Other")
        self.txt_bin_else_lab.setMaximumWidth(90)

        self.txt_bin_newcol = QtWidgets.QLineEdit("")
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

            vals = pd.Series(df[src].dropna().unique()).astype(str)
            try:
                vv = vals.astype(float)
                order = np.argsort(vv.values)
                vals = vals.iloc[order]
            except:
                vals = vals.sort_values()

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
            if not newcol:
                raise RuntimeError("Enter new column name.")
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

    def _compose_segs(self):
        try:
            self._ensure_df()
            df = self.state.df
            cols = [it.text() for it in self.lst_compose_segs.selectedItems()]
            if len(cols) < 2:
                raise RuntimeError("Select 2 or more *_seg columns.")
            newcol = self.txt_compose_newcol.text().strip()
            if not newcol:
                raise RuntimeError("Enter new column name.")
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

    # -------------------------------------------------------------------------
    # Tab 7: Segmentation Setting (Demand Space)
    # -------------------------------------------------------------------------
    def _build_tab_seg_setting(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Segmentation Setting")
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
        """[v8.1] Enhanced with detailed explanations for each mode."""
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
    # Tab 7 (cont): Demand Space Analysis Logic
    # -------------------------------------------------------------------------
    def _run_demand_space(self):
        try:
            self._ensure_df()
            seg_mode = self.cmb_demand_mode.currentText().startswith("Segments-as-points")
            mode = self.cmb_demand_coord.currentText()
            k = int(self.spin_demand_k.value())

            if seg_mode:
                seg_cols = self._selected_checked_items(self.lst_demand_segcols)
                if len(seg_cols) < 1:
                    raise RuntimeError("Select at least 1 *_seg column.")
                sep = self.txt_demand_seg_sep.text().strip() or "|"

                use_factors = bool(self.chk_demand_use_factors.isChecked())
                fac_k = int(self.spin_demand_factor_k.value())
                target = self.cmb_demand_target.currentText().strip()
                if target == "":
                    target = "(None)"
                min_n = int(self.spin_demand_min_n.value())

                prof, feat_cols = self._build_segment_profiles(seg_cols, sep, use_factors, fac_k, target, min_n)

                X = prof[feat_cols].copy()
                X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
                scaler = StandardScaler()
                Xz = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
                Xz = Xz.fillna(0)

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
                self._update_profiler()

                self.lbl_demand_status.setText(f"Done: {coord_name}, segments={len(ids)}, k={k}.")
                self._set_status("Demand Space Analysis Completed.")

            else:
                cols = self._selected_checked_items(self.lst_demand_vars)
                if len(cols) < 3:
                    raise RuntimeError("Select at least 3 variables.")
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
                self._update_profiler()

                self.lbl_demand_status.setText(f"Done: {coord_name}, vars={len(ids)}, k={k}.")
                self._set_status("Demand Space (Vars) Analysis Completed.")

        except Exception as e:
            self.state.last_error = str(e)
            show_error(self, "Demand Space Error", e)

    def _build_segment_profiles(self, seg_cols: List[str], sep: str, use_factors: bool, fac_k: int, target: str, min_n: int):
        df = self.state.df.copy()

        df["_SEG_LABEL_"] = df[seg_cols].astype(str).apply(lambda r: sep.join(r.values), axis=1)

        cnt = df["_SEG_LABEL_"].value_counts()
        valid_segs = cnt[cnt >= min_n].index
        df = df[df["_SEG_LABEL_"].isin(valid_segs)].copy()
        if df.empty:
            raise RuntimeError(f"No segments have >= {min_n} size.")

        feat_cols = []
        if use_factors:
            avail = [c for c in df.columns if str(c).startswith("Factor") and str(c)[6:].isdigit()]
            selected = [c for c in avail if int(c[6:]) <= fac_k]
            feat_cols.extend(selected)

        if target != "(None)" and target in df.columns:
            feat_cols.append(target)
            df[target] = pd.to_numeric(df[target], errors="coerce")

        if not feat_cols:
            raise RuntimeError("No features for profiling (Enable Factors or select Target).")

        prof = df.groupby("_SEG_LABEL_")[feat_cols].mean()
        prof["n"] = df.groupby("_SEG_LABEL_").size()
        return prof, feat_cols

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
    # Tab 8: Segmentation Editing
    # -------------------------------------------------------------------------
    def _build_tab_seg_editing(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Segmentation Editing")
        layout = QtWidgets.QHBoxLayout(tab)

        left = QtWidgets.QVBoxLayout()

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

        # [v8.1] Enhanced Cluster Summary with sub-segment details
        left.addWidget(QtWidgets.QLabel("<b>Cluster Summary</b> (Points & Sub-segments)"))
        self.tbl_cluster_summary = DataFrameTable(float_decimals=2)
        left.addWidget(self.tbl_cluster_summary, 1)

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

        center = QtWidgets.QVBoxLayout()
        self.plot_edit = DemandClusterPlot(editable=True)
        self.plot_edit.sigClustersChanged.connect(self._on_manual_clusters_changed)
        self.plot_edit.sigCoordsChanged.connect(self._on_manual_coords_changed)
        center.addWidget(self.plot_edit, 1)
        layout.addLayout(center, 2)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("<b>Smart Segment Profiler (Z-Scores)</b>"))
        self.txt_profile_report = QtWidgets.QTextEdit()
        self.txt_profile_report.setReadOnly(True)
        right.addWidget(self.txt_profile_report)
        layout.addLayout(right, 1)

        self._on_edit_mode_toggled()

    def _on_edit_mode_toggled(self):
        is_point_edit = self.radio_edit_points.isChecked()
        self.plot_edit.set_edit_mode_active(is_point_edit)
        if is_point_edit:
            self._set_status("Edit Mode: Drag points/labels enabled. (Pan locked)")
        else:
            self._set_status("View Mode: Pan/Zoom enabled. (Editing locked)")

    def _update_cluster_summary(self):
        """[v8.1] Enhanced to show sub-segment details for each cluster."""
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

            # [v8.1] Format items list with truncation for display
            items_display = ", ".join(items[:5])
            if len(items) > 5:
                items_display += f" ... (+{len(items)-5} more)"

            rows.append({
                "Cluster ID": int(cid),
                "Name": names.get(int(cid), f"Cluster {int(cid)}"),
                "Points": len(items),
                "Total N": n_sum if n_map else "-",
                "Sub-segments": items_display
            })
        out = pd.DataFrame(rows).sort_values(["Cluster ID"]).reset_index(drop=True)
        self.tbl_cluster_summary.set_df(out, max_rows=500)

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
        if self.state.demand_xy is None:
            return
        s = self.plot_edit.get_cluster_series()
        self.state.cluster_assign = s
        self.state.manual_dirty = True
        self._update_cluster_summary()
        self._update_profiler()
        self._set_status("Manual clusters updated.")

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
                self._set_status("Manual coords updated.")
        except Exception:
            self._set_status("Manual coords update failed.")

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
                self.state.df.to_excel(w, sheet_name="01_Data", index=False)
                if self.state.recode_df is not None:
                    self.state.recode_df.to_excel(w, sheet_name="02_RECODE", index=False)

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

                if self.state.demand_xy is not None:
                    self.state.demand_xy.to_excel(w, sheet_name="12_Demand_Coords", index=False)

                if self.state.cluster_assign is not None:
                    cl_df = self.state.cluster_assign.reset_index()
                    cl_df.columns = ["id", "cluster_id"]
                    cl_df["cluster_name"] = cl_df["cluster_id"].map(self.state.cluster_names).fillna("")
                    cl_df.to_excel(w, sheet_name="13_Demand_Clusters", index=False)

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
    # Tab 10: AI Assistant (RAG Chatbot) with Retry Logic
    # -------------------------------------------------------------------------
    def _build_tab_rag(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "AI Assistant (RAG)")
        layout = QtWidgets.QVBoxLayout(tab)

        layout.addWidget(QtWidgets.QLabel("<b>AI Assistant</b>: Ask questions about your current data/analysis."))

        # [v8.1] Enhanced API Key section with status
        key_box = QtWidgets.QGroupBox("OpenAI API Configuration")
        key_layout = QtWidgets.QVBoxLayout(key_box)
        
        key_row = QtWidgets.QHBoxLayout()
        self.txt_openai_key = QtWidgets.QLineEdit()
        self.txt_openai_key.setPlaceholderText("Enter OpenAI API Key (sk-...) or leave empty to generate prompt only")
        self.txt_openai_key.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        key_row.addWidget(QtWidgets.QLabel("API Key:"))
        key_row.addWidget(self.txt_openai_key, 1)
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

        self.btn_send_query = QtWidgets.QPushButton("Send / Gen Prompt")
        style_button(self.btn_send_query, level=2)
        self.btn_send_query.clicked.connect(self._send_rag_query)

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

        api_key = self.txt_openai_key.text().strip()
        if not api_key:
            self.txt_chat_history.append(f"<br><i>[System] No API Key provided. Copy this prompt to ChatGPT:</i><br>")
            self.txt_chat_history.append(f"<pre style='background:#f5f5f5; padding:10px;'>{full_prompt}</pre><br><hr>")
            return

        try:
            self.txt_chat_history.append("<i>... Thinking (with auto-retry on rate limit) ...</i>")
            self.lbl_api_status.setText("<b style='color:orange;'>⏳ Sending request...</b>")
            QtWidgets.QApplication.processEvents()

            # [v8.1] Use new retry-enabled API call
            messages = [
                {"role": "system", "content": "You are a Data Analysis Assistant for market research."},
                {"role": "user", "content": full_prompt}
            ]
            success, result = call_openai_api(
                api_key, 
                messages, 
                max_retries=3, 
                initial_delay=2.0,
                timeout=30
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
    win = IntegratedApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
