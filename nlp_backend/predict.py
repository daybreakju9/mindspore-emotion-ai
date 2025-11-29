# predict.py
from mindnlp.transformers import BertTokenizer, BertModel
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


# -----------------------------------------------------
# 加载模型（只加载一次，不走 HuggingFace 权重）
# -----------------------------------------------------
print(">>> 正在加载 MindNLP 官方 bert-base-chinese 模型...")

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert = BertModel.from_pretrained("bert-base-chinese")

print(">>> 模型加载成功！已准备就绪。")

# -----------------------------------------------------
# 自定义一个轻量级情感分类头（2 分类）
# -----------------------------------------------------
class SentimentClassifier(nn.Cell):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Dense(768, 2)  # 2 分类：负面 / 正面

    def construct(self, hidden_states):
        cls_embedding = hidden_states[:, 0, :]  # CLS 向量
        cls_embedding = self.dropout(cls_embedding)
        return self.classifier(cls_embedding)


classifier = SentimentClassifier()


# -----------------------------------------------------
# 核心函数：情绪分析
# -----------------------------------------------------
def analyze_text(text):
    # 1. tokenizer
    encoded = tokenizer(text, return_tensors="ms", max_length=128,
                        truncation=True, padding="max_length")

    # 2. bert 获取向量
    outputs = bert(**encoded)
    hidden_states = outputs.last_hidden_state

    # 3. 分类
    logits = classifier(hidden_states)
    probs = ops.softmax(logits, axis=-1)
    label_id = int(probs.asnumpy().argmax())

    label_map = {
        0: "负面",
        1: "正面"
    }
    emotion = label_map[label_id]

    suggestion_map = {
        "负面": "你可能遇到了压力或负面情绪，建议适当放松和倾诉。",
        "正面": "情绪良好，继续保持积极的生活节奏吧！"
    }

    return emotion, suggestion_map[emotion]


# -----------------------------------------------------
# 自测
# -----------------------------------------------------
if __name__ == "__main__":
    tests = [
        "我真的快撑不下去了。",
        "今天心情特别好！",
        "感觉有点累，但还能坚持。",
    ]

    for t in tests:
        emo, sug = analyze_text(t)
        print(f"\n输入：{t}")
        print(f"情绪：{emo}")
        print(f"建议：{sug}")
