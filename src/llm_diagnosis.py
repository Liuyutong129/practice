import os
from openai import OpenAI


def generate_llm_diagnosis(prompt: str, model: str | None = None) -> str:
    api_key = os.getenv("ZAI_API_KEY")
    base_url = os.getenv("ZAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
    default_model = os.getenv("ZAI_MODEL", "glm-4.7-flash")

    if model is None or model.strip() == "":
        model = default_model

    if not api_key:
        return "未检测到 ZAI_API_KEY，请先在 .env 文件中配置智谱 API Key。"

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一名工业过程监测与故障诊断助手。请基于给定异常摘要进行专业、谨慎、简洁的分析，不要编造未提供的数据结论。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"大模型诊断生成失败：{str(e)}"