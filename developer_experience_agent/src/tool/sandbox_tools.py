import requests
from langchain_core.tools import tool

# -------------------------------------------------------------------------
# AIO Sandbox HTTP Client (纯网络调用版)
# -------------------------------------------------------------------------
class SandboxClient:
    def __init__(self, base_url="http://127.0.0.1:8080"):
        self.base_url = base_url.rstrip("/")
        print(f"   [Sandbox] Client initialized. Target: {self.base_url}")

    def exec_command(self, command: str, timeout: int = 300) -> str:
        """
        调用 /v1/shell/exec 接口执行命令
        """
        url = f"{self.base_url}/v1/shell/exec"
        print(f"   [Sandbox Request] {command[:100]}..." if len(command) > 100 else f"   [Sandbox Request] {command}")
        
        try:
            response = requests.post(url, json={"command": command, "timeout": timeout}, timeout=timeout + 5)
            
            # 处理 HTTP 层面的错误
            if response.status_code != 200:
                return f"Sandbox HTTP Error ({response.status_code}): {response.text}"
            
            raw = response.json()
            
            # --- 解析返回结果 (兼容不同 API 格式) ---
            if "data" in raw and isinstance(raw["data"], dict):
                result = raw["data"]
                output = result.get("output", "")
                error = result.get("error", "")
                exit_code = result.get("exit_code")
            else:
                # 兼容直接返回的情况
                result = raw
                output = result.get("stdout", "")
                error = result.get("stderr", "")
                exit_code = result.get("exit_code")
                if exit_code is None: 
                    exit_code = result.get("returncode")

            # --- 结果处理逻辑 ---
            # 优先使用 output (stdout)，如果为空且有 error (stderr)，则使用 error
            # Git/uv 等工具经常把正常日志输出到 stderr
            display_text = output.strip()
            if not display_text and error:
                display_text = error.strip()
            
            if not display_text:
                display_text = "(Command executed with no text output)"

            if exit_code == 0:
                return display_text
            else:
                # 命令执行失败 (非0退出码)
                return f"Command Failed (Exit Code {exit_code}):\n{display_text}"

        except Exception as e:
            return f"Sandbox Connection Error: {str(e)}"

    def read_file(self, path: str) -> str:
        """
        调用 /v1/file/read 接口读取文件
        """
        url = f"{self.base_url}/v1/file/read"
        try:
            response = requests.post(url, json={"path": path}, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # 兼容 data.content 或直接 content
                content = data.get("data", {}).get("content", "")
                if not content and "content" in data:
                    content = data["content"]
                return content
            return f"Error reading file ({response.status_code}): {response.text}"
        except Exception as e:
            return f"File Read Error: {str(e)}"

# 初始化单例
sandbox_client = SandboxClient()

# -------------------------------------------------------------------------
# LangChain Tools
# -------------------------------------------------------------------------

@tool
def sandbox_exec(command: str):
    """
    Execute a shell command in the sandbox.
    Usage: sandbox_exec("ls -la") or sandbox_exec("git clone https://...")
    """
    return sandbox_client.exec_command(command)

@tool
def sandbox_read_file(path: str):
    """
    Read a file content from the sandbox.
    Usage: sandbox_read_file("/absolute/path/to/file")
    """
    return sandbox_client.read_file(path)

sandbox_tools = [sandbox_exec, sandbox_read_file]