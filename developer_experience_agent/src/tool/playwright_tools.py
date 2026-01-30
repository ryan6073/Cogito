import asyncio
import threading
import shutil
import subprocess
import os
import re
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# -------------------------------------------------------------------------
# MCP 客户端桥接器 (Docker Mode)
# -------------------------------------------------------------------------
class PlaywrightMCPClient:
    def __init__(self):
        # 1. 检查 Docker 环境
        if not shutil.which("docker"):
            if not shutil.which("npx"):
                raise RuntimeError("未找到 docker 或 npx。无法启动 Playwright MCP。")
            self.mode = "npx"
            self._ensure_browser_dependencies()
        else:
            self.mode = "docker"
            print("   [System] 检测到 Docker，将使用 Docker 运行 Playwright MCP。")

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        self._session: Optional[ClientSession] = None
        self._connected_event = threading.Event()
        self._init_error = None
        
        asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
        
        print(f"   [System] 正在启动 Playwright MCP 服务 ({self.mode})...")
        if not self._connected_event.wait(timeout=120):
            raise TimeoutError("连接 Playwright MCP 服务超时。")
        if self._init_error:
            raise RuntimeError(f"Playwright MCP 初始化失败: {self._init_error}")
        print("   [System] Playwright MCP 服务已连接。")

    def _ensure_browser_dependencies(self):
        if self.mode == "npx":
            print("   [System] 正在检查本地依赖...")
            env = os.environ.copy()
            env["PLAYWRIGHT_DOWNLOAD_HOST"] = "https://npmmirror.com/mirrors/playwright/"
            try:
                subprocess.run(["npx", "playwright", "install", "chromium"], check=True, env=env, timeout=300)
            except:
                pass

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect(self):
        if self.mode == "docker":
            command = "docker"
            args = [
                "run", "-i", "--rm", "--init",
                "mcr.microsoft.com/playwright/mcp"
            ]
        else:
            command = "npx"
            args = [
                "-y", "@playwright/mcp@latest", 
                "--browser", "chromium", "--headless", "--no-sandbox"
            ]

        server_params = StdioServerParameters(command=command, args=args, env=None)
        
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._connected_event.set()
                    await asyncio.Future() 
        except Exception as e:
            self._init_error = str(e)
            self._connected_event.set()

    def _smart_filter_snapshot(self, raw_text: str) -> str:
        """
        智能正文提取算法：
        1. 识别并移除导航栏/页脚（特征：高密度连续链接）。
        2. 保留核心内容区域（特征：包含 Heading, Text, Input）。
        3. 放宽截断限制。
        """
        lines = raw_text.split('\n')
        filtered_lines = []
        
        # 统计窗口内的链接密度
        # 简单的启发式：如果连续 5 行以上都是 link 且没有 heading/text，视为导航/菜单
        
        buffer = []
        is_nav_block = False
        
        # 正则预编译
        link_pattern = re.compile(r'\[\d+\] link')
        content_pattern = re.compile(r'heading|text|textbox|button|input')
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            is_link = bool(link_pattern.search(line))
            is_content = bool(content_pattern.search(line))
            
            if is_link and not is_content:
                buffer.append(line)
            else:
                # 遇到非纯链接内容，处理缓冲区
                if len(buffer) > 8: 
                    # 缓冲区超过8行纯链接 -> 视为导航/噪音，丢弃或仅保留少量
                    # 保留前2个和后2个作为上下文，中间折叠
                    filtered_lines.extend(buffer[:2])
                    filtered_lines.append(f"  ... [已折叠 {len(buffer)-4} 个导航链接] ...")
                    filtered_lines.extend(buffer[-2:])
                else:
                    # 不是导航块，全部保留
                    filtered_lines.extend(buffer)
                
                buffer = []
                filtered_lines.append(line)
        
        # 处理末尾缓冲区
        if len(buffer) > 5:
            filtered_lines.extend(buffer[:2])
            filtered_lines.append("  ... [已折叠页脚链接] ...")
        else:
            filtered_lines.extend(buffer)
            
        result = "\n".join(filtered_lines)
        
        # 最后的安全截断：放宽到 50,000 字符 (约 12k tokens)
        # 我们依赖后续的 LLM Summarization 来进一步压缩
        MAX_CHARS = 50000
        if len(result) > MAX_CHARS:
            return result[:MAX_CHARS] + f"\n\n[System] Snapshot truncated at {MAX_CHARS} chars."
        
        return result

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if not self._session: return "Error: MCP session not connected."

        async def _async_call():
            try:
                # 移除不支持的参数
                mcp_args = arguments.copy()
                if name == "browser_snapshot" and "focus" in mcp_args:
                    del mcp_args["focus"]
                
                result = await self._session.call_tool(name, arguments=mcp_args)
                output = []
                if hasattr(result, 'content'):
                    for item in result.content:
                        if item.type == 'text': output.append(item.text)
                
                full_text = "\n".join(output)
                
                # --- 后处理：智能提取正文 ---
                if name == "browser_snapshot":
                    full_text = self._smart_filter_snapshot(full_text)
                
                return full_text

            except Exception as e:
                return f"MCP Error: {str(e)}"

        future = asyncio.run_coroutine_threadsafe(_async_call(), self._loop)
        try:
            return future.result(timeout=60)
        except Exception as e:
            return f"Timeout: {str(e)}"

# 初始化
try:
    mcp_client = PlaywrightMCPClient()
except Exception as e:
    print(f"⚠️ Warning: Playwright Client 启动失败: {e}")
    mcp_client = None

@tool
def browser_navigate(url: str):
    """导航到指定 URL。"""
    if not mcp_client: return "Error: Client not initialized."
    print(f"   [MCP] 导航至: {url}")
    return mcp_client.call_tool("browser_navigate", {"url": url})

@tool
def browser_snapshot(focus: str = ""):
    """获取页面快照。"""
    if not mcp_client: return "Error: Client not initialized."
    print(f"   [MCP] 获取快照...")
    return mcp_client.call_tool("browser_snapshot", {"focus": focus})

@tool
def browser_click(ref: str, element: str = ""):
    """点击元素。"""
    if not mcp_client: return "Error: Client not initialized."
    print(f"   [MCP] 点击 Ref: {ref}")
    return mcp_client.call_tool("browser_click", {"ref": str(ref), "element": element})

@tool
def browser_type(ref: str, text: str, element: str = ""):
    """输入文本。"""
    if not mcp_client: return "Error: Client not initialized."
    print(f"   [MCP] 输入文本到 Ref: {ref}")
    return mcp_client.call_tool("browser_type", {"ref": str(ref), "text": text, "element": element})

mcp_tools = [browser_navigate, browser_snapshot, browser_click, browser_type]