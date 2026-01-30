from langchain_community.tools import DuckDuckGoSearchRun

def run_web_search(query: str) -> str:
    """
    使用 DuckDuckGo 进行联网搜索
    """
    print(f"   [联网搜索] 正在检索关键词: {query}")
    search = DuckDuckGoSearchRun()
    try:
        result = search.run(query)
        return result
    except Exception as e:
        return f"搜索出错: {str(e)}"