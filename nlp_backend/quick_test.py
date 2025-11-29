# test_mindspore_emotion.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def comprehensive_test():
    """全面测试MindSpore情绪分析系统"""
    print("MindSpore情绪分析系统全面测试")
    print("=" * 60)

    # 测试1：基础环境
    print("\n1. 测试MindSpore基础环境")
    try:
        import mindspore as ms
        import mindspore.ops as ops
        import numpy as np

        x = ms.Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
        y = ms.Tensor(np.array([4.0, 5.0, 6.0]), ms.float32)
        z = ops.add(x, y)

        print(f"MindSpore版本: {ms.__version__}")
        print(f"张量加法测试: {x} + {y} = {z}")
        print("基础环境测试通过")

    except Exception as e:
        print(f"基础环境测试失败: {e}")
        return False

    # 测试2：模块导入
    print("\n2. 测试模块导入")
    try:
        from nlp_backend import analyze_text, test_analyze, test_mindspore_operations
        from nlp_backend.emotion_model_loader import emotion_model_loader

        print("模块导入成功")
        print(f"情绪分析版本: {emotion_model_loader.__class__.__name__}")

    except Exception as e:
        print(f"模块导入失败: {e}")
        return False

    # 测试3：MindSpore操作测试
    print("\n3. 测试MindSpore操作")
    try:
        from nlp_backend import test_mindspore_operations
        test_mindspore_operations()
    except Exception as e:
        print(f"MindSpore操作测试失败: {e}")

    # 测试4：情绪分析功能测试
    print("\n4. 情绪分析功能测试")
    test_cases = [
        ("今天很开心天气很好", "正面"),
        ("心情不好很失望", "负面"),
        ("普通的一天工作", "中性"),
        ("非常喜欢这个项目", "正面"),
        ("遇到困难很烦躁", "负面"),
        ("正常的生活节奏", "中性")
    ]

    passed = 0
    total = len(test_cases)

    for text, expected in test_cases:
        try:
            emotion, suggestion = analyze_text(text)
            status = "通过" if emotion == expected else "失败"
            if status == "通过":
                passed += 1

            print(f"测试: {text}")
            print(f"预期: {expected}, 实际: {emotion}, 状态: {status}")
            print(f"建议: {suggestion}")
            print()

        except Exception as e:
            print(f"测试失败 - 文本: {text}, 错误: {e}")

    print(f"功能测试结果: {passed}/{total} 通过")

    # 测试5：边界情况测试
    print("\n5. 边界情况测试")
    edge_cases = [
        "",
        "   ",
        "12345",
        "abcdefg",
        "！@#￥%"
    ]

    for case in edge_cases:
        try:
            emotion, suggestion = analyze_text(case)
            print(f"边界测试: '{case}' -> 情绪: {emotion}, 建议: {suggestion}")
        except Exception as e:
            print(f"边界测试失败: '{case}', 错误: {e}")

    # 测试6：性能测试
    print("\n6. 性能测试")
    import time

    test_text = "今天心情很不错"
    start_time = time.time()

    for i in range(10):
        emotion, suggestion = analyze_text(test_text)

    end_time = time.time()
    avg_time = (end_time - start_time) / 10

    print(f"10次测试平均耗时: {avg_time:.4f}秒")
    print(f"最后结果: {emotion} - {suggestion}")

    # 测试7：模型状态检查
    print("\n7. 模型状态检查")
    try:
        model = emotion_model_loader.get_model()
        if model:
            print("MindSpore模型状态: 已加载")
            print(f"模型类型: {type(model).__name__}")
        else:
            print("MindSpore模型状态: 未加载，使用规则分析")
    except Exception as e:
        print(f"模型状态检查失败: {e}")

    print("\n" + "=" * 60)
    print("测试完成")
    return passed == total


def interactive_test():
    """交互式测试"""
    print("\n交互式测试模式")
    print("输入'退出'结束测试")
    print("-" * 40)

    while True:
        user_input = input("请输入要分析的文本: ").strip()

        if user_input in ['退出', 'exit', 'quit']:
            break

        if not user_input:
            print("输入不能为空")
            continue

        try:
            emotion, suggestion = analyze_text(user_input)
            print(f"分析结果:")
            print(f"  情绪: {emotion}")
            print(f"  建议: {suggestion}")
            print("-" * 40)
        except Exception as e:
            print(f"分析失败: {e}")


if __name__ == "__main__":
    print("MindSpore情绪分析系统测试套件")
    print()

    # 运行全面测试
    success = comprehensive_test()

    if success:
        print("\n全面测试通过！开始交互式测试...")
        interactive_test()
    else:
        print("\n全面测试失败，请检查以上错误信息")

    print("\n测试结束")