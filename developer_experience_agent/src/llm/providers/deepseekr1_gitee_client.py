import os
from langchain_openai import ChatOpenAI

def get_deepseek_gitee_client():
    """
    针对 Gitee AI 提供的 DeepSeek-R1 接口实现。
    API 网址: https://ai.gitee.com
    """
    # 建议将这些值放入 .env 文件中
    api_key = os.getenv("GITEE_AI_API_KEY")
    base_url = os.getenv("GITEE_AI_BASE_URL")
    
    # 注意：在 Gitee AI 平台上，请确认具体的模型 ID（例如 'DeepSeek-R1'）
    model_name = os.getenv("GITEE_AI_MODEL")

    if not api_key:
        raise ValueError("请设置 GITEE_AI_API_KEY 环境变量。")

    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        # DeepSeek-R1 通常建议使用较低的 temperature 以获得更稳定的思维链
        temperature=0.6,
        streaming=False,
        # 如果模型不支持原生工具调用，我们依然会使用之前的“手动解析”逻辑
        max_retries=3
    )