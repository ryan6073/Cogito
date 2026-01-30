import os
from langchain_openai import ChatOpenAI

def get_qwen3_vllm_client():
    """
    针对 Qwen3-vLLM 端点的特定实现
    """
    # 1. 确保获取正确的环境变量，如果 .env 没加载，这里会打印出来帮助排查
    api_key = os.getenv("CUSTOM_LLM_API_KEY", "oss-compass-st1592633") # 临时硬编码默认值用于测试
    
    # 注意：LangChain/OpenAI SDK 通常需要 base_url 包含 /v1
    # 你的 curl 是 http://1.94.11.116:7000/v1/...
    base_url = os.getenv("CUSTOM_LLM_BASE_URL", "http://1.94.11.116:7000/v1") 
    
    # 2. 【关键修正】模型名称必须与 curl 中成功的名称严格一致
    # 之前是 "Qwen3-30B-A3B-GPTQ-Int4"，改为 "Qwen3-30B-A3B"
    model_name = os.getenv("CUSTOM_LLM_MODEL", "Qwen3-30B-A3B")

    print(f"--- Debug Qwen3 Client ---")
    print(f"Target URL: {base_url}")
    print(f"Target Model: {model_name}")
    print(f"--------------------------")

    if not api_key or not base_url:
        raise ValueError("请检查 .env 文件中 Qwen3 相关的配置。")

    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key, 
        model=model_name,
        # 3. 传入 extra_body 启用思考
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True}
        },
        temperature=0.7,
        streaming=False,
        model_kwargs={
            "parallel_tool_calls": False 
        }
    )