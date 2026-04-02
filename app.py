import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

from src.detector import (
    detect_anomalies,
    extract_anomaly_segments,
    build_anomaly_summary,
    build_sensor_anomaly_stats
)
from src.report_text import build_rule_based_summary, build_llm_prompt
from src.llm_diagnosis import generate_llm_diagnosis

load_dotenv()
st.set_page_config(page_title="工业异常监测与诊断系统", layout="wide")

st.title("基于大模型解释的工业异常监测与诊断系统")
st.markdown("### Step 2：异常检测与异常摘要生成")

# 读取数据
file_path = "data/raw/demo_timeseries.csv"
df = pd.read_csv(file_path)

st.subheader("原始数据预览")
st.dataframe(df.head(10), use_container_width=True)

sensor_cols = [c for c in df.columns if c != "time"]

# 参数区
st.subheader("检测参数设置")
col1, col2 = st.columns(2)

with col1:
    selected_sensor = st.selectbox("选择一个传感器查看曲线", sensor_cols)

with col2:
    threshold = st.slider("异常阈值", min_value=2.0, max_value=6.0, value=3.5, step=0.1)

# 异常检测
result_df = detect_anomalies(df, sensor_cols, threshold=threshold)
segments = extract_anomaly_segments(result_df, min_len=3)
summary = build_anomaly_summary(result_df, segments)
sensor_stats_df = build_sensor_anomaly_stats(result_df, sensor_cols)
# 指标摘要
summary_text = build_rule_based_summary(summary, segments, sensor_stats_df)
llm_prompt = build_llm_prompt(summary_text)
# 指标摘要
st.subheader("异常检测摘要")
m1, m2, m3, m4 = st.columns(4)
m1.metric("总采样点数", summary["total_points"])
m2.metric("异常点数", summary["anomaly_points"])
m3.metric("异常区间数", summary["segment_count"])
m4.metric("最高异常分数", f'{summary["max_score"]:.2f}')

st.info(
    f"最可疑传感器：**{summary['top_sensor']}** ｜ "
    f"异常点占比：**{summary['anomaly_ratio']:.2%}**"
)

# 各传感器异常统计
st.subheader("各传感器异常统计")
show_sensor_stats = sensor_stats_df.copy()
show_sensor_stats["anomaly_ratio"] = show_sensor_stats["anomaly_ratio"].map(lambda x: f"{x:.2%}")
show_sensor_stats["max_abs_score"] = show_sensor_stats["max_abs_score"].map(lambda x: f"{x:.2f}")
st.dataframe(show_sensor_stats, use_container_width=True)
st.subheader("自动异常摘要文本")
st.text_area(
    "系统自动生成的异常分析摘要",
    value=summary_text,
    height=320
)

st.subheader("用于 LLM 的诊断提示词")
st.code(llm_prompt, language="text")

col_a, col_b = st.columns(2)

with col_a:
    st.download_button(
        label="下载异常摘要文本",
        data=summary_text,
        file_name="anomaly_summary.txt",
        mime="text/plain"
    )

with col_b:
    st.download_button(
        label="下载LLM提示词",
        data=llm_prompt,
        file_name="llm_prompt.txt",
        mime="text/plain"
    )

st.subheader("大模型诊断解释")

model_name = st.text_input("模型名称", value=os.getenv("ZAI_MODEL", "glm-4.7"))

if "llm_result" not in st.session_state:
    st.session_state["llm_result"] = ""

if st.button("生成大模型诊断报告"):
    with st.spinner("正在调用大模型生成诊断解释..."):
        result = generate_llm_diagnosis(llm_prompt, model=model_name)
        st.session_state["llm_result"] = result

st.text_area(
    "LLM 生成的诊断结果",
    value=st.session_state["llm_result"],
    height=320
)

if st.session_state["llm_result"]:
    st.download_button(
        label="下载LLM诊断报告",
        data=st.session_state["llm_result"],
        file_name="llm_diagnosis.txt",
        mime="text/plain"
    )
# 画图：正常曲线 + 异常点标红
# 画图：正常曲线 + 当前传感器自己的异常点标红
st.subheader("异常检测可视化")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=result_df["time"],
        y=result_df[selected_sensor],
        mode="lines",
        name=selected_sensor
    )
)

selected_flag_col = f"{selected_sensor}_is_anomaly"
anomaly_df = result_df[result_df[selected_flag_col]]

fig.add_trace(
    go.Scatter(
        x=anomaly_df["time"],
        y=anomaly_df[selected_sensor],
        mode="markers",
        name=f"{selected_sensor} 异常点",
        marker=dict(size=8, color="red", symbol="circle")
    )
)

# 给异常区间加浅红背景
for seg in segments:
    start_time = result_df.loc[seg["start"], "time"]
    end_time = result_df.loc[seg["end"], "time"]
    fig.add_vrect(
        x0=start_time,
        x1=end_time,
        opacity=0.15,
        line_width=0,
        fillcolor="red"
    )

fig.update_layout(
    title=f"{selected_sensor} 异常检测结果",
    xaxis_title="time",
    yaxis_title=selected_sensor,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# 输出异常区间表
st.subheader("异常区间列表")

if len(segments) == 0:
    st.success("当前阈值下未检测到明显异常区间。")
else:
    seg_df = pd.DataFrame(segments)
    st.dataframe(seg_df, use_container_width=True)

# 输出带标签的数据
st.subheader("带异常标签的数据预览")
show_cols = ["time"] + sensor_cols + ["anomaly_score", "is_anomaly", "top_sensor"]
st.dataframe(
    result_df.loc[result_df["is_anomaly"], show_cols],
    use_container_width=True
)

st.success("异常检测模块已完成：现在已经可以自动识别异常点和异常区间。")