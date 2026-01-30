import re
import json
import operator
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------------------------------------------------
# 1. 状态定义 (State)
# -------------------------------------------------------------------------
class SearchStrategy(TypedDict):
    keywords: List[str]
    current_keyword_idx: int
    current_attempt: int

class EvaluationResult(TypedDict):
    step_id: int
    agent_name: str
    action_content: str
    score: int             # 1-5分
    assessment: str        # 评价理由
    suggestion: str        # 改进建议

class TraceLog(TypedDict):
    step_id: int
    agent: str             # Explorer / Executor
    action: str
    thought_reason: str
    blocker: str
    status: str            # Success / Failed / InProgress

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trace_logs: Annotated[List[TraceLog], operator.add]
    evaluations: Annotated[List[EvaluationResult], operator.add]
    strategy: SearchStrategy
    metrics: Dict[str, Any]
    manual_tool_call: Dict[str, Any]
    
    # 流程控制
    current_phase: str     # 'exploration' | 'execution' | 'audit'
    target_url: str        # 代码仓库链接
    
    # 新增状态：当前工作目录 (Current Working Directory)
    # 用于模拟有状态的 Shell 会话
    cwd: str
    
    # Executor 步数计数器
    executor_step_count: int

# -------------------------------------------------------------------------
# 2. 提示词库
# -------------------------------------------------------------------------
class AgentPrompts:
    # --- Explorer ---
    EXPLORER_TOOL_DESC = """
        浏览器工具 (JSON):
        {{"action": "browser_navigate", "url": "..."}}
        {{"action": "browser_snapshot"}}
        {{"action": "browser_click", "ref": "...", "element": "..."}}
        {{"action": "browser_type", "ref": "...", "text": "...", "element": "..."}}
        """
    EXPLORER_SYSTEM = f"""
        你是一个 CANN 社区的新手开发者。任务：搜索并找到 'ops-math' 的源码仓库地址。
        {EXPLORER_TOOL_DESC}
        【策略】
        1. **规划**：如果是初始状态，输出 [PLAN_KEYWORDS: k1, k2, ...]。
        2. **执行**：每个关键词最多尝试 5 次。必须通过快照 (browser_snapshot) 观察页面。
        3. **思考**：每次行动前输出 [THOUGHT: 原因...]。
        4. **成功**：找到类似 `gitcode.com/cann/ops-math` 或 `gitee.com/...` 的仓库链接后，回复 'MISSION_COMPLETE' 并附带链接。
        5. **切换**：当前关键词无效回复 'SWITCH_KEYWORD'。
        """

    # --- Executor ---
    # 【CRITICAL FIX】Executor 使用了 .format()，所以 JSON 的大括号必须转义为 {{...}}
    EXECUTOR_TOOL_DESC = """
        Sandbox 工具 (JSON):
        {{"action": "sandbox_exec", "command": "shell命令"}}
        {{"action": "sandbox_read_file", "path": "文件路径"}}
        """
    
    # EXECUTOR_SYSTEM = (
    #     "你是一个代码执行 Agent。你现在处于 **Linux 终端 (AIO Sandbox)** 环境中。\n"
    #     "你的**唯一目标**是：跑通 ops-math 样例并**看到运行成功的日志**。\n"
    #     "代码仓库：{target_url}\n"
    #     "当前工作目录 (CWD): {cwd}\n\n"
    #     "【⚠️ 常见错误预警】\n"
    #     "❌ **错误操作**：`git clone` 后直接 `cat README.md`。\n"
    #     "   -> 结果：`No such file`。因为项目在子目录里！\n"
    #     "✅ **正确操作**：`git clone` -> `ls -F` (看清子目录名) -> `cd <项目名>` -> `cat README.md`。\n\n"
    #     "【⚠️ 严格判定标准】\n"
    #     "1. **必须进目录**：下载后，第一件事必须是 `ls` 确认目录名，然后 `cd` 进去。\n"
    #     "2. **只读有用信息**：阅读 README 是为了找 `build.sh` 或 `run` 命令。\n"
    #     "3. **环境受限**：这是一个 Sandbox，可能没有 NPU。如果遇到 `ascend-toolkit not found`，尝试找找纯 Python 样例。\n"
    #     "4. **工具使用**：\n"
    #     f"{EXECUTOR_TOOL_DESC}\n"
    #     "- 使用 `cd` 切换目录时，系统会记住位置。\n"
    #     "- 优先用 `uv run` 或 `bash` 运行脚本。\n\n"
    #     "【执行建议路径】\n"
    #     "1. `git clone {{target_url}}`\n"
    #     "2. `ls -F` (**关键步骤**：确认克隆生成的文件夹名称)\n"
    #     "3. `cd <文件夹名>` (进入项目根目录)\n"
    #     "4. `sandbox_read_file README.md`\n"
    #     "5. **运行验证**：执行编译或测试脚本。\n\n"
    #     "每次行动前必须输出 [THOUGHT: ...] 解释你的计划。如果是最后一步且成功，回复 'MISSION_SUCCESS'。"
    # )

    EXECUTOR_SYSTEM = (
        "你是一个代码执行 Agent。你现在处于 **Linux 终端 (AIO Sandbox)** 环境中。\n"
        "目标：跑通 ops-math 样例。\n"
        "代码仓库：{target_url}\n"
        "当前工作目录 (CWD): {cwd}\n\n"
        "【⚠️ 严格操作规范】\n"
        "1. **禁止浏览器**：无显示器，只能用终端。\n"
        "2. **有状态 Shell**：系统会记住 `cd` 的位置。\n"
        "3. **一步一动**：拆解步骤，严禁多条命令混写。\n"
        f"{EXECUTOR_TOOL_DESC}\n\n"
        "【📋 标准执行流程 (SOP)】\n"
        "请严格参考以下步骤执行 (假设当前是 ops-math 项目)：\n\n"
        "**1. 获取源码**\n"
        "   - 命令：`git clone {{target_url}}`\n"
        "   - 验证：`ls -F` (确认目录名，通常是 ops-math)\n"
        "   - 进入：`cd ops-math`\n\n"
        "**2. 编译 AddExample 算子**\n"
        "   - 命令：`bash build.sh --pkg --soc=ascend910b --ops=add_example -j16`\n"
        "   - 预期：提示 `Self-extractable archive ... successfully created.`\n\n"
        "**3. 安装算子包 (关键)**\n"
        "   - 查找包：`ls build_out`\n"
        "   - 安装：`./build_out/cann-ops-math-*.run` (注意：如果无root权限且安装失败，尝试继续后续步骤或查看报错)\n\n"
        "**4. 配置环境变量 (关键)**\n"
        "   - 命令：`export LD_LIBRARY_PATH=${{ASCEND_HOME_PATH}}/opp/vendors/custom_math/op_api/lib:${{LD_LIBRARY_PATH}}`\n"
        "   - 注意：`ASCEND_HOME_PATH` 通常是 `/usr/local/Ascend`。如果环境变量未生效，运行样例可能会报错。\n\n"
        "**5. 快速验证：运行算子样例**\n"
        "   - 命令：`bash build.sh --run_example add_example eager cust --vendor_name=custom`\n"
        "   - 预期：打印 `add_example first input... result...`\n\n"
        "每次行动前输出 [THOUGHT: ...] 解释你的计划。如果最后一步成功输出计算结果，回复 'MISSION_SUCCESS'。"
    )

    # --- Evaluator (以小白体验为核心) ---
    EVALUATOR_SYSTEM = """
        你是一个极其挑剔的体验评估者（User Experience Evaluator）。
        你的视角完全代表一个**零基础、缺乏耐心的小白开发者**。
        【打分标准 (1-5分)】
        - **5分 (丝滑)**: 一次操作直接命中目标（如搜索第一条就是官网、代码一行命令跑通）。无报错，无歧义。
        - **3分 (普通)**: 需要尝试 2-3 次才找到路，或者报错但报错信息能看懂。有轻微挫败感。
        - **1分 (劝退)**: 
        - 需要反复试错（尝试 > 3次）。
        - 遇到看不懂的报错（如 'missing .so library'）。
        - 页面充斥着深奥术语（如 TBE, DaVinci）而找不到 'Run' 按钮。
        - 需要复杂的环境配置。

        请分析 Agent 的 [Action] 和 [Thought]，输出 JSON 评估：
        ```json
        {{
        "score": 1-5,
        "assessment": "从小白角度点评",
        "suggestion": "针对体验卡点的改进建议"
        }}
        """

    # --- Auditor (以折腾指数为核心) ---
    AUDITOR_SYSTEM = """
        你是审计专家。请根据 trace_logs 和 evaluations 生成《CANN 体验体检报告》。 
        【核心原则】 
        体验好坏由'尝试次数'（折腾指数）决定，而非是否最终成功。

        【要求 1：全链路执行审计】 
        请统计以下指标：

        搜索阶段折腾度: 尝试了多少个关键词？点击了多少次无效链接？(少于3次操作为优)

        运行阶段折腾度: 执行了多少次命令？遇到了多少次报错？(少于5步为优)

        【要求 2：用户旅程地图 (User Journey Map)】 
        输出 Markdown 表格，必须包含以下列： 
        | 步骤 | 阶段(Agent) | 关键动作 | 思考理由(小白心路) | 体验得分 | 体验点评 | 关键卡点 | 
        | :--- | :--- | :--- | :--- | :--- | :--- | :--- |

        注意：

        '思考理由' 直接提取自日志 [THOUGHT]。

        '体验得分' 和 '点评' 来自 Evaluator。

        如果得分低，必须在'关键卡点'列标记具体的阻碍（如：文档入口隐蔽、环境依赖缺失）。 
        """


def parse_json_safely(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    json_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1))
        except:
            pass
    bracket_match = re.search(r"\{.*?\}", text, re.DOTALL)
    if bracket_match:
        try:
            candidate = re.sub(r",\s*([}\]])", r"\1", bracket_match.group(0))
            return json.loads(candidate)
        except:
            pass
    return {}


def create_explorer_node(llm):
    def node(state: AgentState):
        strategy = state['strategy']
        kw_idx = strategy['current_keyword_idx']
        attempt = strategy['current_attempt']
        print(f"\n>>> [EXPLORER] KwIdx={kw_idx}, Attempt={attempt}/5")

        # 知识库兜底：如果策略耗尽，注入正确链接
        if (strategy['keywords'] and kw_idx >= len(strategy['keywords'])) or (kw_idx == 2 and attempt >= 5):
            correct_url = "https://gitcode.com/cann/ops-math" 
            print(f"   [知识库干预] 搜索多次失败，系统自动注入正确链接: {correct_url}")
            log_entry = {
                "step_id": len(state['messages']), "agent": "System", "action": "Inject Answer", 
                "thought_reason": "小白已经彻底迷失（搜索策略耗尽），系统不得不直接给出答案以继续流程。", 
                "blocker": "搜索路径完全阻断", "status": "Injected"
            }
            return {
                "trace_logs": [log_entry],
                "target_url": correct_url,
                "current_phase": "execution", 
                "manual_tool_call": {} 
            }

        system_msg = ""
        if not strategy['keywords']:
            system_msg = "[系统提示]: 请先分析任务，生成 3 个搜索关键词。格式: [PLAN_KEYWORDS: ...]"
        elif attempt >= 5:
            new_kw_idx = kw_idx + 1
            return {
                "strategy": {"keywords": strategy['keywords'], "current_keyword_idx": new_kw_idx, "current_attempt": 0},
                "trace_logs": [{"step_id": len(state['messages']), "agent": "System", "action": "策略切换", "thought_reason": "当前关键词尝试次数过多，小白耐心耗尽，换个词试试。", "blocker": "关键词搜索结果不佳", "status": "Switch"}]
            }
        else:
             curr_kw = strategy['keywords'][kw_idx]
             system_msg = f"[系统状态]: 当前关键词: '{curr_kw}'。第 {attempt + 1}/5 次操作。请输出 JSON 指令。"

        prompt = f"{AgentPrompts.EXPLORER_SYSTEM}\n{system_msg}"
        messages = [{"role": "system", "content": prompt}] + state["messages"][-6:] 
        response = llm.invoke(messages)
        content = response.content
        
        updates = {
            "messages": [response],
            "manual_tool_call": {},
            "strategy": strategy,
            "trace_logs": []
        }
        _parse_common_explorer_logic(content, updates, state)
        
        if "MISSION_COMPLETE" in content:
            url_match = re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
            found_url = url_match.group(0) if url_match else "https://gitcode.com/cann/ops-math" 
            updates["target_url"] = found_url
            updates["current_phase"] = "execution"

        return updates
    return node

def create_executor_node(llm):
    def node(state: AgentState):
        current_step = state.get("executor_step_count", 0)
        # 硬性终止
        if current_step >= 30:
            print(f"   [EXECUTOR] 🛑 达到最大执行步数 (30)。强制停止。")
            return {
                "trace_logs": [{
                    "step_id": len(state['messages']), "agent": "System", "action": "Force Stop",
                    "thought_reason": "Executor 超过 30 次尝试。", "blocker": "任务超时", "status": "Failed"
                }],
                "manual_tool_call": {}, 
                "messages": [AIMessage(content="[SYSTEM]: MISSION_FAILED_TIMEOUT")]
            }

        target_url = state.get('target_url', 'Unknown')
        current_cwd = state.get('cwd', '/home/jiashun/SearchAgent/tool_serve/sandbox/workspace') # 获取当前 CWD，默认为 /root
        print(f"\n>>> [EXECUTOR] Step {current_step+1}/30. Target: {target_url} | CWD: {current_cwd}")
        
        try:
            # 注入变量到 Prompt
            prompt = AgentPrompts.EXECUTOR_SYSTEM.format(
                target_url=target_url, 
                cwd=current_cwd
            )
        except Exception as e:
            # 兜底：如果格式化失败，使用带工具定义的 fallback prompt
            print(f"   [Prompt Error] {e}")
            prompt = (
                f"System Error in Prompt Formatting: {e}\n"
                f"Target: {target_url}\n"
                f"CWD: {current_cwd}\n"
                "Please output JSON to execute commands.\n"
                f"{AgentPrompts.EXECUTOR_TOOL_DESC}" # 关键：带上工具定义
            )

        # 强制 System Message 覆盖历史
        override_msg = {
            "role": "system", 
            "content": f"!!! ATTENTION: EXECUTION PHASE. You are in terminal at: {current_cwd}. OUTPUT JSON ONLY."
        }
        
        messages = [{"role": "system", "content": prompt}, override_msg] + state["messages"][-6:]
        response = llm.invoke(messages)
        content = response.content
        
        updates = {
            "messages": [response],
            "manual_tool_call": {},
            "trace_logs": [],
            "executor_step_count": current_step + 1
        }
        
        thought_match = re.search(r"\[THOUGHT: (.*?)\]", content, re.DOTALL)
        thought_text = thought_match.group(1).strip() if thought_match else "Execution step"
        
        tool_call = parse_json_safely(content)
        if tool_call and "action" in tool_call:
            print(f"   [DEBUG] Parsed tool call: {tool_call}")
            updates["manual_tool_call"] = tool_call
            updates['trace_logs'].append({
                "step_id": len(state['messages']), "agent": "Executor",
                "action": f"{tool_call.get('action')} {tool_call.get('command') or tool_call.get('path') or ''}",
                "thought_reason": thought_text, "blocker": "", "status": "InProgress"
            })
        else:
             print(f"   [DEBUG] No tool call. Content preview: {content[:50]}...")
             updates['trace_logs'].append({
                "step_id": len(state['messages']), "agent": "Executor",
                "action": "思考", "thought_reason": thought_text, "blocker": "", "status": "Thinking"
            })

        return updates
    return node

def create_evaluator_node(llm):
    def node(state: AgentState):
        if not state['trace_logs']: return {}
        last_log = state['trace_logs'][-1]
        print(f"   [EVALUATOR] 正在评估 Agent: {last_log['agent']} 的操作: {last_log['action']}")
        
        prompt = AgentPrompts.EVALUATOR_SYSTEM
        user_msg = f"Agent: {last_log['agent']}\nAction: {last_log['action']}\nThought: {last_log['thought_reason']}"
        response = llm.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": user_msg}])
        
        eval_data = parse_json_safely(response.content)
        eval_result = {}
        if eval_data:
            eval_result = {
                "step_id": last_log['step_id'], "agent_name": last_log['agent'], "action_content": last_log['action'],
                "score": eval_data.get("score", 3), "assessment": eval_data.get("assessment", "无"), "suggestion": eval_data.get("suggestion", "")
            }
            print(f"      -> 评分: {eval_result['score']}/5, 点评: {eval_result['assessment']}")
        else:
            print("      -> 评估解析失败")
        return {"evaluations": [eval_result] if eval_result else []}
    return node

def create_auditor_node(llm):
    def node(state: AgentState):
        print("\n>>> [AUDITOR] 正在生成全链路体检报告...")
        logs_text = "【全链路执行记录】:\n"
        eval_map = {e['step_id']: e for e in state['evaluations']}
        for l in state['trace_logs']:
            logs_text += f"- Step {l['step_id']} | Agent: {l['agent']} | Action: {l['action']}\n  Thought: {l['thought_reason']}\n"
            if l['step_id'] in eval_map:
                ev = eval_map[l['step_id']]
                logs_text += f"  [EVALUATION] Score: {ev['score']}/5 | Comment: {ev['assessment']}\n"
            logs_text += "---\n"
        prompt = f"{AgentPrompts.AUDITOR_SYSTEM}\n\n{logs_text}"
        response = llm.invoke([{"role": "system", "content": prompt}])
        return {"messages": [response]}
    return node

def _parse_common_explorer_logic(content, updates, state):
    kw_match = re.search(r"\[PLAN_KEYWORDS: (.*?)\]", content)
    if kw_match:
        keywords = [k.strip() for k in kw_match.group(1).split(',')]
        updates['strategy']['keywords'] = keywords
        updates['trace_logs'].append({
            "step_id": len(state['messages']), "agent": "Explorer", "action": "规划关键词", 
            "thought_reason": f"规划关键词: {keywords}", "blocker": "", "status": "Planned"
        })
        return
    thought_match = re.search(r"\[THOUGHT: (.*?)\]", content, re.DOTALL)
    thought_text = thought_match.group(1).strip() if thought_match else "无详细思考"
    tool_call = parse_json_safely(content)
    if tool_call and "action" in tool_call:
        updates["manual_tool_call"] = tool_call
        updates['strategy']['current_attempt'] += 1
        action_desc = f"{tool_call.get('action')} {tool_call.get('url', '')} {tool_call.get('element', '')}"
        updates['trace_logs'].append({
            "step_id": len(state['messages']), "agent": "Explorer", "action": action_desc,
            "thought_reason": thought_text, "blocker": "", "status": "Action"
        })
    else:
        updates['trace_logs'].append({
            "step_id": len(state['messages']), "agent": "Explorer", "action": "思考",
            "thought_reason": thought_text, "blocker": "", "status": "Thinking"
        })