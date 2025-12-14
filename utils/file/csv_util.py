from pathlib import Path
from typing import Tuple
import numpy as np

from utils.file.file_util import get_root_dir
ROOT_DIR = Path(get_root_dir())

import csv
import datetime
import os

sep = os.path.sep
rootdir = get_root_dir()


def to_dataframe(abs_path=None, relative_path=None):
    import pandas as pd
    if abs_path is not None:
        path = Path(abs_path)
    elif relative_path is not None:
        rel_path = Path(relative_path)
        path = rel_path if rel_path.is_absolute() else ROOT_DIR / rel_path
    else:
        raise ValueError("abs_path 或 relative_path 至少提供一个")
    return pd.read_csv(path)

def read_numeric_csv(file_path):
    data = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            numeric_row = []
            for cell in row:
                try:
                    # Try converting to float first (handles both int and float)
                    num = float(cell)
                    # Convert to int if it's a whole number for cleaner output
                    if num.is_integer():
                        num = int(num)
                    numeric_row.append(num)
                except ValueError:
                    raise ValueError(f"Non-numeric value found: '{cell}' in row: {row}")
            data.append(numeric_row)

    return data

def read(file_path,is_numeric=False):
    if is_numeric:
        return read_numeric_csv(file_path)
    data = []
    # 读取数据
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def is_2d(array):
    if not isinstance(array, (list, tuple)):
        return False
    return all(isinstance(row, (list, tuple)) for row in array) and len(array) > 0

def write_data(file_path, data):
    assert is_2d(data), "Data must be a 2D array (list of lists or tuples)"
    # 确保文件夹存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # 写入数据
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def append_data(file_path, data):
     # 确保文件夹存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # 追加数据
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        # Add empty line if file is not empty
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            writer.writerow([])  # Write empty row

        writer.writerows(data)

def replace_nan_with_average(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    for idx, value in enumerate(arr):
        if np.isnan(value):
            prev_val = arr[idx - 1] if idx > 0 else np.nan
            next_val = arr[idx + 1] if idx < len(arr) - 1 else np.nan
            if not np.isnan(prev_val) and not np.isnan(next_val):
                arr[idx] = (prev_val + next_val) / 2
            elif not np.isnan(prev_val):
                arr[idx] = prev_val
            elif not np.isnan(next_val):
                arr[idx] = next_val
    return arr

def load_series_from_csv(csv_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    csv_path = Path(csv_path)
    df = (
        to_dataframe(abs_path=str(csv_path))
        if csv_path.is_absolute()
        else to_dataframe(relative_path=str(csv_path))
    )
    if df.shape[1] < 2:
        raise ValueError(f'文件 {csv_path} 至少需要包含 step 列和一列数据')
    steps = df.iloc[:, 0].to_numpy()
    values = np.stack(
        [replace_nan_with_average(df[col].to_numpy()) for col in df.columns[1:]],
        axis=1,
    )
    return steps, values.astype(float)
