# from langgraph.graph import StateGraph, END
# from langchain_core.messages import AIMessage
# from src.core.pipeline import (
#     AgentState, create_explorer_node, create_executor_node, 
#     create_evaluator_node, create_auditor_node
# )
# from src.llm.providers.deepseekr1_gitee_client import get_deepseek_gitee_client
# from src.tool.playwright_tools import browser_navigate, browser_snapshot, browser_click, browser_type
# from src.tool.sandbox_tools import sandbox_exec, sandbox_read_file

# def create_workflow():
#     llm = get_deepseek_gitee_client()
#     workflow = StateGraph(AgentState)

#     def manual_tools_node(state: AgentState):
#         cmd = state.get("manual_tool_call", {})
#         action = cmd.get("action")
#         print(f"   [执行工具] {action}...")
        
#         result = "未知指令"
#         try:
#             # Playwright Tools
#             if action == "browser_navigate":
#                 result = browser_navigate.invoke({"url": cmd["url"]})
#             elif action == "browser_snapshot":
#                 result = browser_snapshot.invoke({})
#             elif action == "browser_click":
#                 result = browser_click.invoke({"ref": str(cmd.get("ref")), "element": cmd.get("element", "")})
#             elif action == "browser_type":
#                 result = browser_type.invoke({"ref": str(cmd.get("ref")), "text": cmd.get("text"), "element": ""})
            
#             # Sandbox Tools
#             elif action == "sandbox_exec":
#                 result = sandbox_exec.invoke({"command": cmd["command"]})
#             elif action == "sandbox_read_file":
#                 result = sandbox_read_file.invoke({"path": cmd["path"]})
                
#         except Exception as e:
#             result = f"Error: {str(e)}"

#         return {
#             "messages": [AIMessage(content=f"[TOOL_RESULT]: {result}")],
#             "manual_tool_call": {} 
#         }

#     # 注册节点
#     workflow.add_node("explorer", create_explorer_node(llm))
#     workflow.add_node("executor", create_executor_node(llm))
#     workflow.add_node("manual_tools", manual_tools_node)
#     workflow.add_node("evaluator", create_evaluator_node(llm))
#     workflow.add_node("auditor", create_auditor_node(llm))

#     workflow.set_entry_point("explorer")

#     # -------------------------------------------------------------------------
#     # 路由逻辑 (包含失败处理)
#     # -------------------------------------------------------------------------
    
#     def route_explorer(state: AgentState):
#         if state.get("current_phase") == "execution":
#             return "executor"
#         if state.get("manual_tool_call", {}).get("action"):
#             return "manual_tools"
#         return "evaluator"

#     def route_executor(state: AgentState):
#         # 1. 检查是否强制终止 (FAILED/TIMEOUT) 或 成功 (SUCCESS)
#         last_msg = state["messages"][-1].content
#         if "MISSION_FAILED" in last_msg or "MISSION_SUCCESS" in last_msg:
#             return "auditor"

#         # 2. 工具调用
#         if state.get("manual_tool_call", {}).get("action"):
#             return "manual_tools"
            
#         return "evaluator"

#     def route_tools(state: AgentState):
#         return "evaluator"

#     def route_evaluator(state: AgentState):
#         if state.get("current_phase") == "execution":
#             return "executor"
#         return "explorer"

#     # 连线
#     workflow.add_conditional_edges("explorer", route_explorer, 
#         {"manual_tools": "manual_tools", "executor": "executor", "evaluator": "evaluator"})
    
#     workflow.add_conditional_edges("executor", route_executor, 
#         {"manual_tools": "manual_tools", "auditor": "auditor", "evaluator": "evaluator"})
        
#     workflow.add_conditional_edges("manual_tools", route_tools, 
#         {"evaluator": "evaluator"})
        
#     workflow.add_conditional_edges("evaluator", route_evaluator, 
#         {"explorer": "explorer", "executor": "executor"})

#     workflow.add_edge("auditor", END)

#     return workflow.compile()

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from src.core.pipeline import (
    AgentState, create_explorer_node, create_executor_node, 
    create_evaluator_node, create_auditor_node
)
from src.llm.providers.deepseekr1_gitee_client import get_deepseek_gitee_client
from src.tool.playwright_tools import browser_navigate, browser_snapshot, browser_click, browser_type
from src.tool.sandbox_tools import sandbox_exec, sandbox_read_file

def create_workflow():
    llm = get_deepseek_gitee_client()
    workflow = StateGraph(AgentState)

    def manual_tools_node(state: AgentState):
        cmd_data = state.get("manual_tool_call", {})
        action = cmd_data.get("action")
        print(f"   [执行工具] {action}...")
        
        # 获取当前工作目录，默认为 /root
        current_cwd = state.get("cwd", "/root")
        
        result = "未知指令"
        # 用于更新 state 的字典
        state_updates = {"manual_tool_call": {}}
        
        try:
            # --- Playwright Tools (Explorer Phase) ---
            if action == "browser_navigate":
                result = browser_navigate.invoke({"url": cmd_data["url"]})
            elif action == "browser_snapshot":
                result = browser_snapshot.invoke({})
            elif action == "browser_click":
                result = browser_click.invoke({"ref": str(cmd_data.get("ref")), "element": cmd_data.get("element", "")})
            elif action == "browser_type":
                result = browser_type.invoke({"ref": str(cmd_data.get("ref")), "text": cmd_data.get("text"), "element": ""})
            
            # --- Sandbox Tools (Executor Phase) ---
            elif action == "sandbox_exec":
                raw_command = cmd_data["command"].strip()
                
                # 1. 特殊处理 cd 命令 (状态变更)
                if raw_command.startswith("cd "):
                    target_dir = raw_command[3:].strip()
                    # 组合命令：去当前目录 -> 尝试进入新目录 -> 打印绝对路径
                    # 如果中间失败，pwd 不会执行 (&& 的特性)
                    wrapped_cmd = f"cd {current_cwd} && cd {target_dir} && pwd"
                    exec_output = sandbox_exec.invoke({"command": wrapped_cmd})
                    
                    # 检查是否执行成功
                    if "Command Failed" in exec_output or "No such file" in exec_output or "not a directory" in exec_output:
                        result = f"Error changing directory:\n{exec_output}"
                    else:
                        # 成功，最后一行应该是新的路径
                        lines = exec_output.strip().split('\n')
                        new_cwd = lines[-1].strip()
                        # 简单的路径合法性检查
                        if new_cwd.startswith("/"):
                            state_updates["cwd"] = new_cwd
                            result = f"Directory changed to: {new_cwd}"
                        else:
                            result = f"Unexpected output from cd: {exec_output}"
                        
                # 2. 处理其他命令 (自动包裹当前目录)
                else:
                    wrapped_cmd = f"cd {current_cwd} && {raw_command}"
                    result = sandbox_exec.invoke({"command": wrapped_cmd})
                    
            elif action == "sandbox_read_file":
                path = cmd_data["path"]
                # 策略：如果不确定 API 是否支持相对路径，使用 'cat' 命令更稳妥
                # 因为 cat 可以通过上面的逻辑利用 current_cwd
                if not path.startswith("/"):
                    print(f"   [Sandbox] Reading relative path via cat: {path}")
                    wrapped_cmd = f"cd {current_cwd} && cat {path}"
                    result = sandbox_exec.invoke({"command": wrapped_cmd})
                else:
                    result = sandbox_read_file.invoke({"path": path})
                
            # --- 错误处理 ---
            elif "browser_" in str(action) and state.get("current_phase") == "execution":
                result = (
                    f"❌ Error: Tool '{action}' is NOT available in EXECUTOR phase.\n"
                    "You are in a Linux Terminal (Sandbox). NO Browser access.\n"
                    "Available tools: `sandbox_exec`, `sandbox_read_file`.\n"
                    "Example: Use `sandbox_exec` to run `ls` or `git clone`."
                )
            else:
                result = f"Error: Unknown tool '{action}'. Please check your JSON format."
                
        except Exception as e:
            result = f"Execution Error: {str(e)}"

        return {
            "messages": [AIMessage(content=f"[TOOL_RESULT]: {result}")],
            **state_updates # 合并更新 (如 cwd)
        }

    # ... (其余注册节点和路由逻辑保持不变) ...
    workflow.add_node("explorer", create_explorer_node(llm))
    workflow.add_node("executor", create_executor_node(llm))
    workflow.add_node("manual_tools", manual_tools_node)
    workflow.add_node("evaluator", create_evaluator_node(llm))
    workflow.add_node("auditor", create_auditor_node(llm))

    # 入口点设置为 executor (根据您的调试需求)
    workflow.set_entry_point("executor") 

    def route_explorer(state: AgentState):
        if state.get("current_phase") == "execution": return "executor"
        if state.get("manual_tool_call", {}).get("action"): return "manual_tools"
        return "evaluator"

    def route_executor(state: AgentState):
        last_msg = state["messages"][-1].content
        if "MISSION_FAILED" in last_msg or "MISSION_SUCCESS" in last_msg: return "auditor"
        if state.get("manual_tool_call", {}).get("action"): return "manual_tools"
        return "evaluator"

    def route_tools(state: AgentState): return "evaluator"
    def route_evaluator(state: AgentState):
        if state.get("current_phase") == "execution": return "executor"
        return "explorer"

    workflow.add_conditional_edges("explorer", route_explorer, {"manual_tools": "manual_tools", "executor": "executor", "evaluator": "evaluator"})
    workflow.add_conditional_edges("executor", route_executor, {"manual_tools": "manual_tools", "auditor": "auditor", "evaluator": "evaluator"})
    workflow.add_conditional_edges("manual_tools", route_tools, {"evaluator": "evaluator"})
    workflow.add_conditional_edges("evaluator", route_evaluator, {"explorer": "explorer", "executor": "executor"})
    workflow.add_edge("auditor", END)

    return workflow.compile()