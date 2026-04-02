import numpy as np
import pandas as pd


def robust_zscore(series: pd.Series) -> pd.Series:
    """
    用中位数 + MAD 计算稳健 z-score
    """
    median = series.median()
    mad = np.median(np.abs(series - median))

    if mad < 1e-8:
        std = series.std()
        if std < 1e-8:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - series.mean()) / std

    return 0.6745 * (series - median) / mad


def detect_anomalies(df: pd.DataFrame, sensor_cols: list, threshold: float = 3.5) -> pd.DataFrame:
    """
    对多个传感器做稳健异常检测：
    - 每个传感器算 robust z-score
    - 每个传感器单独生成异常标记：abs(rz) > threshold
    - 每个时刻取所有传感器中绝对值最大的那个作为 anomaly_score
    - 只要任一传感器超过阈值，则该时刻判为异常
    """
    result_df = df.copy()
    zscore_cols = []
    sensor_flag_cols = []

    for col in sensor_cols:
        z_col = f"{col}_rz"
        flag_col = f"{col}_is_anomaly"

        result_df[z_col] = robust_zscore(result_df[col])
        result_df[flag_col] = result_df[z_col].abs() > threshold

        zscore_cols.append(z_col)
        sensor_flag_cols.append(flag_col)

    abs_z = result_df[zscore_cols].abs()
    result_df["anomaly_score"] = abs_z.max(axis=1)
    result_df["is_anomaly"] = result_df[sensor_flag_cols].any(axis=1)
    result_df["top_sensor"] = abs_z.idxmax(axis=1).str.replace("_rz", "", regex=False)

    return result_df


def extract_anomaly_segments(result_df: pd.DataFrame, min_len: int = 3):
    """
    提取连续异常区间
    返回：
    [
        {"start": 180, "end": 219, "length": 40, "main_sensor": "sensor_3"},
        ...
    ]
    """
    anomaly_idx = result_df.index[result_df["is_anomaly"]].tolist()
    if not anomaly_idx:
        return []

    segments = []
    start = anomaly_idx[0]
    prev = anomaly_idx[0]

    for idx in anomaly_idx[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            if prev - start + 1 >= min_len:
                seg_df = result_df.loc[start:prev]
                main_sensor = seg_df["top_sensor"].mode().iloc[0]
                segments.append({
                    "start": int(start),
                    "end": int(prev),
                    "length": int(prev - start + 1),
                    "main_sensor": main_sensor
                })
            start = idx
            prev = idx

    if prev - start + 1 >= min_len:
        seg_df = result_df.loc[start:prev]
        main_sensor = seg_df["top_sensor"].mode().iloc[0]
        segments.append({
            "start": int(start),
            "end": int(prev),
            "length": int(prev - start + 1),
            "main_sensor": main_sensor
        })

    return segments


def build_anomaly_summary(result_df: pd.DataFrame, segments: list) -> dict:
    """
    生成页面展示用摘要
    """
    total_points = len(result_df)
    anomaly_points = int(result_df["is_anomaly"].sum())
    anomaly_ratio = anomaly_points / total_points if total_points > 0 else 0

    if anomaly_points > 0:
        top_sensor = result_df.loc[result_df["is_anomaly"], "top_sensor"].mode().iloc[0]
        max_score = float(result_df["anomaly_score"].max())
    else:
        top_sensor = "无"
        max_score = 0.0

    return {
        "total_points": total_points,
        "anomaly_points": anomaly_points,
        "anomaly_ratio": anomaly_ratio,
        "segment_count": len(segments),
        "top_sensor": top_sensor,
        "max_score": max_score
    }
def build_sensor_anomaly_stats(result_df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """
    统计每个传感器各自的异常点个数、占比、最大异常分数
    """
    rows = []

    total_points = len(result_df)

    for col in sensor_cols:
        flag_col = f"{col}_is_anomaly"
        rz_col = f"{col}_rz"

        anomaly_count = int(result_df[flag_col].sum())
        anomaly_ratio = anomaly_count / total_points if total_points > 0 else 0.0
        max_abs_score = float(result_df[rz_col].abs().max())

        rows.append({
            "sensor": col,
            "anomaly_count": anomaly_count,
            "anomaly_ratio": anomaly_ratio,
            "max_abs_score": max_abs_score
        })

    return pd.DataFrame(rows).sort_values(
        by=["anomaly_count", "max_abs_score"],
        ascending=False
    ).reset_index(drop=True)