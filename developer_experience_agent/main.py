# import os
# from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage
# from src.core.orchestrator import create_workflow

# load_dotenv()

# def main():
#     app = create_workflow()
#     print("🚀 启动 CANN 社区全链路仿真 (Explorer -> Evaluator -> Executor)")
    
#     initial_state = {
#         "messages": [HumanMessage(content="我现在是一个CANN社区的新手，我希望迅速跑通CANN社区的ops-math库，请你帮助我。")],
#         "trace_logs": [],
#         "evaluations": [],
#         "strategy": {
#             "keywords": [], 
#             "current_keyword_idx": 0, 
#             "current_attempt": 0
#         },
#         "metrics": {
#             "found_correct_link": False,
#             "link_verified": False,
#             "attempts_to_success": 0
#         },
#         "manual_tool_call": {},
#         "current_phase": "exploration",
#         "target_url": ""
#     }

#     try:
#         # 增加 recursion_limit 防止长链路中断
#         for output in app.stream(initial_state, {"recursion_limit": 50}):
#             for node_name, state_update in output.items():
#                 print(f"   [Flow] Node: {node_name}")
#     except Exception as e:
#         print(f"运行时发生错误: {e}")

# if __name__ == "__main__":
#     main()

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.core.orchestrator import create_workflow

load_dotenv()

def main():
    app = create_workflow()
    print("🚀 启动 CANN 社区仿真 (DEBUG模式: 仅执行阶段)")
    
    target_repo = "https://gitcode.com/cann/ops-math"
    
    initial_state = {
        "messages": [HumanMessage(content=f"我已经找到了代码仓库：{target_repo}。请帮我跑通里面的 QuickStart 样例。")],
        "trace_logs": [],
        "evaluations": [],
        "strategy": {
            "keywords": [], 
            "current_keyword_idx": 0, 
            "current_attempt": 0
        },
        "metrics": {
            "found_correct_link": True, 
            "link_verified": True,
            "attempts_to_success": 0
        },
        "manual_tool_call": {},
        "current_phase": "execution",
        "target_url": target_repo,
        "executor_step_count": 0,
        "cwd": "/home/jiashun/SearchAgent/tool_serve/sandbox/workspace" # 初始目录
    }

    try:
        for output in app.stream(initial_state, {"recursion_limit": 100}):
            for node_name, state_update in output.items():
                print(f"   [Flow] Node: {node_name}")
    except Exception as e:
        print(f"运行时发生错误: {e}")

if __name__ == "__main__":
    main()