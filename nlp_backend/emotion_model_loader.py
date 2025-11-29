# nlp_backend/emotion_model_loader.py
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.dataset as ds
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionClassifier(nn.Cell):
    """使用MindSpore构建的情绪分类模型"""

    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=64, num_classes=3):
        super(EmotionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Dense(hidden_dim * 2, num_classes)
        self.softmax = nn.Softmax(axis=1)

    def construct(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim*2)

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim*2)

        output = self.dropout(last_output)
        logits = self.classifier(output)  # (batch_size, num_classes)
        probabilities = self.softmax(logits)

        return logits, probabilities


class EmotionModelLoader:
    def __init__(self):
        self.model = None
        self.vocab = {}
        self._build_model()

    def _build_model(self):
        """构建和训练情绪分析模型"""
        try:
            logger.info("正在构建MindSpore情绪分析模型")

            # 创建模型实例
            self.model = EmotionClassifier(vocab_size=10000, num_classes=3)

            # 创建简单的词汇表
            self._build_vocab()

            # 模拟训练数据
            self._simulate_training()

            logger.info("MindSpore情绪分析模型构建完成")

        except Exception as e:
            logger.error(f"模型构建失败: {e}")
            self.model = None

    def _build_vocab(self):
        """构建简单词汇表"""
        common_words = [
            '开心', '高兴', '快乐', '幸福', '满意', '棒', '好', '爱', '喜欢',
            '伤心', '难过', '生气', '愤怒', '失望', '糟糕', '讨厌', '恨', '痛苦',
            '今天', '天气', '工作', '学习', '生活', '感觉', '心情', '事情'
        ]

        # 为每个词分配一个ID
        for i, word in enumerate(common_words, 1):
            self.vocab[word] = i

        # 添加未知词和填充词
        self.vocab['[UNK]'] = 0
        self.vocab['[PAD]'] = len(self.vocab)

    def _text_to_ids(self, text, max_length=50):
        """将文本转换为ID序列"""
        words = list(text)  # 简单的中文分词：按字符分割
        ids = [self.vocab.get(word, self.vocab['[UNK]']) for word in words]

        # 填充或截断
        if len(ids) < max_length:
            ids = ids + [self.vocab['[PAD]']] * (max_length - len(ids))
        else:
            ids = ids[:max_length]

        return ids

    def _simulate_training(self):
        """模拟训练过程"""
        try:
            # 创建一些模拟训练数据
            positive_texts = ["今天很开心", "天气真好高兴", "喜欢这个电影"]
            negative_texts = ["心情不好伤心", "很失望难过", "讨厌这件事"]
            neutral_texts = ["普通的一天", "正常的工作", "平常的生活"]

            # 模拟训练过程（实际项目中需要真实训练）
            logger.info("模型预训练完成（模拟）")

        except Exception as e:
            logger.warning(f"模拟训练失败: {e}")

    def predict_emotion(self, text):
        """使用MindSpore模型进行情绪预测"""
        if self.model is None:
            return None

        try:
            # 将文本转换为模型输入
            input_ids = self._text_to_ids(text)
            input_tensor = Tensor([input_ids], ms.int32)

            # 模型推理
            _, probabilities = self.model(input_tensor)
            probs = probabilities.asnumpy()[0]

            # 获取预测结果
            emotion_idx = np.argmax(probs)
            confidence = probs[emotion_idx]

            emotions = ["负面", "中性", "正面"]
            return emotions[emotion_idx], float(confidence)

        except Exception as e:
            logger.error(f"预测失败: {e}")
            return None

    def get_model(self):
        return self.model


# 创建全局实例
emotion_model_loader = EmotionModelLoader()