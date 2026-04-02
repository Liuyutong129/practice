import pandas as pd


def build_rule_based_summary(summary: dict, segments: list, sensor_stats_df: pd.DataFrame) -> str:
    """
    基于检测结果自动生成一段中文异常摘要
    """

    lines = []

    # 1) 总体概况
    lines.append("一、总体检测结果")
    lines.append(
        f"本次共分析 {summary['total_points']} 个采样点，检测到 {summary['anomaly_points']} 个异常点，"
        f"异常点占比约为 {summary['anomaly_ratio']:.2%}，共识别出 {summary['segment_count']} 个连续异常区间。"
    )
    lines.append(
        f"全局最高异常分数为 {summary['max_score']:.2f}，当前最可疑的传感器为 {summary['top_sensor']}。"
    )

    # 2) 各传感器统计
    lines.append("")
    lines.append("二、各传感器异常统计")
    nonzero_df = sensor_stats_df[sensor_stats_df["anomaly_count"] > 0].copy()

    if len(nonzero_df) == 0:
        lines.append("当前阈值下，各传感器均未出现明显异常点。")
    else:
        for _, row in nonzero_df.iterrows():
            lines.append(
                f"- {row['sensor']}：异常点 {int(row['anomaly_count'])} 个，"
                f"占比 {row['anomaly_ratio']:.2%}，最大绝对异常分数 {row['max_abs_score']:.2f}。"
            )

    # 3) 连续异常区间
    lines.append("")
    lines.append("三、连续异常区间分析")
    if len(segments) == 0:
        lines.append("未检测到满足最小长度要求的连续异常区间。")
    else:
        for i, seg in enumerate(segments, start=1):
            lines.append(
                f"- 区间 {i}：从 time={seg['start']} 到 time={seg['end']}，"
                f"持续 {seg['length']} 个采样点，主导异常传感器为 {seg['main_sensor']}。"
            )

    # 4) 初步诊断结论
    lines.append("")
    lines.append("四、初步诊断结论")
    if len(nonzero_df) == 0:
        lines.append("当前阈值设置下系统整体运行平稳，未见显著异常。")
    elif len(nonzero_df) == 1:
        sensor_name = nonzero_df.iloc[0]["sensor"]
        lines.append(
            f"异常主要集中在 {sensor_name}，更像是单变量局部异常，"
            f"建议优先检查该传感器对应测点、采集链路或相关工艺变量。"
        )
    else:
        top2 = nonzero_df.head(2)["sensor"].tolist()
        lines.append(
            f"异常涉及多个传感器（如 {', '.join(top2)}），"
            f"可能存在变量耦合扰动、工况切换或系统级异常传播现象，"
            f"建议结合过程机理进一步排查。"
        )

    # 5) 建议
    lines.append("")
    lines.append("五、建议的后续分析方向")
    lines.append("- 结合异常区间前后的工况变化，检查是否存在设定值切换、负载突变或数据采集异常。")
    lines.append("- 对主导异常传感器进行重点排查，分析其原始信号、上下游变量及相关控制量。")
    lines.append("- 后续可接入大模型，对异常区间、异常变量和统计结果进行自动解释与诊断报告生成。")

    return "\n".join(lines)


def build_llm_prompt(summary_text: str) -> str:
    """
    生成给 LLM 使用的诊断提示词
    """

    prompt = f"""
你是一名工业过程监测与故障诊断助手。请基于下面的异常检测摘要，完成诊断分析。

要求：
1. 用正式、专业、简洁的中文输出；
2. 先概括异常现象，再分析可能原因；
3. 指出最值得优先排查的传感器或变量；
4. 给出后续诊断建议；
5. 不要编造没有提供的数据结论，如果证据不足请明确说明“需要进一步数据支持”。

异常检测摘要如下：
{summary_text}

请输出为以下结构：
1. 异常现象概述
2. 可能原因分析
3. 优先排查对象
4. 后续建议
""".strip()

    return prompt