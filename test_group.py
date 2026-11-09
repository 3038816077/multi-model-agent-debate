from app.agent.group_graph import group_app

if __name__ == "__main__":
    # 构造测试输入
    test_input = {
        "query": "如果我想学习 AI 开发，应该从 Python 基础开始，还是直接学 LangChain 搭应用？",
        "chief_model": "qwen-plus",
        "staff_models": ["glm-4-flash", "kimi-k3.1-pro"], # 选两个你配了Key的副模型
        "first_round": {},
        "discussion": {},
        "final_summary": "",
        "failed_models": []
    }

    print("🚀 开始多模型研讨测试...\n")
    
    # 运行 Graph
    result = group_app.invoke(test_input)
    
    print("\n" + "="*50)
    print("✅ 第一轮独立作答：")
    for m, ans in result["first_round"].items():
        print(f"\n--- {m} ---\n{ans}")
        
    print("\n" + "="*50)
    print("✅ 第二轮交叉研讨：")
    for m, ans in result["discussion"].items():
        print(f"\n--- {m} ---\n{ans}")

    print("\n" + "="*50)
    print("👑 首席总结（主模型）：")
    print(result["final_summary"])