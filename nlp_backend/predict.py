# nlp_backend/predict.py
import mindspore as ms
import mindspore.ops as ops
import numpy as np
import logging
import random
from .emotion_model_loader import emotion_model_loader

logger = logging.getLogger(__name__)

EMOTION_SUGGESTIONS = {
    "正面": ["继续保持积极心态", "正能量满满", "适合挑战新目标"],
    "负面": ["建议深呼吸放松", "可以运动听音乐", "给自己独处时间"],
    "中性": ["情绪稳定很好", "可以尝试新活动", "适合深度思考"]
}


def rule_based_analyzer(text):
    """基于规则的分析"""
    positive_words = ['开心', '高兴', '快乐', '幸福', '满意', '棒', '好']
    negative_words = ['伤心', '难过', '生气', '愤怒', '失望', '糟糕', '讨厌']

    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)

    if positive_count > negative_count:
        emotion = "正面"
    elif negative_count > positive_count:
        emotion = "负面"
    else:
        emotion = "中性"

    return emotion, random.choice(EMOTION_SUGGESTIONS[emotion])


def analyze_text(text):
    """使用MindSpore模型进行情绪分析"""
    try:
        if not text or len(text.strip()) == 0:
            return "中性", "请输入文字"

        # 使用MindSpore模型预测
        model_result = emotion_model_loader.predict_emotion(text)

        if model_result:
            emotion, confidence = model_result
            logger.info(f"MindSpore模型预测: {emotion} (置信度: {confidence:.3f})")

            # 结合规则分析提高准确性
            rule_emotion, _ = rule_based_analyzer(text)

            # 如果置信度低或者与规则分析冲突，使用规则结果
            if confidence < 0.6 or emotion != rule_emotion:
                final_emotion = rule_emotion
                logger.info(f"置信度低，使用规则分析: {final_emotion}")
            else:
                final_emotion = emotion
        else:
            # 模型不可用，使用规则分析
            final_emotion, _ = rule_based_analyzer(text)
            logger.info("使用规则分析")

        suggestion = random.choice(EMOTION_SUGGESTIONS[final_emotion])
        return final_emotion, suggestion

    except Exception as e:
        logger.error(f"分析失败: {e}")
        return rule_based_analyzer(text)


def test_mindspore_operations():
    """测试MindSpore操作"""
    print("测试MindSpore张量操作:")

    # 创建张量
    x = ms.Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    y = ms.Tensor(np.array([4.0, 5.0, 6.0]), ms.float32)

    # 使用MindSpore操作
    z = ops.add(x, y)
    mean_val = ops.reduce_mean(z)
    max_val = ops.reduce_max(z)

    print(f"张量加法: {x} + {y} = {z}")
    print(f"平均值: {mean_val}")
    print(f"最大值: {max_val}")
    print("✅ MindSpore操作正常")


def test_analyze():
    """测试情绪分析"""
    test_texts = [
        "今天很开心天气很好",
        "心情不好很失望",
        "普通的一天工作",
        "非常喜欢这个项目",
        "遇到困难很烦躁"
    ]

    print("MindSpore情绪分析测试")
    print("=" * 40)

    # 先测试MindSpore基础操作
    test_mindspore_operations()
    print()

    # 测试情绪分析
    for i, text in enumerate(test_texts, 1):
        emotion, suggestion = analyze_text(text)
        print(f"测试{i}: {text}")
        print(f"情绪: {emotion}")
        print(f"建议: {suggestion}")
        print("-" * 30)


if __name__ == "__main__":
    test_analyze()