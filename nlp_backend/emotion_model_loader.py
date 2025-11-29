# emotion_model_loader.py
from mindnlp.transformers import BertTokenizer, BertForSequenceClassification
import mindspore as ms

_MODEL = None
_TOKENIZER = None

def load_emotion_model():
    global _MODEL, _TOKENIZER

    if _MODEL is not None and _TOKENIZER is not None:
        return _TOKENIZER, _MODEL

    print(">>> 正在加载 IDEA 中文情感分析模型，请稍等...")

    # IDEA 研究院情感模型（100% 可下载）
    repo = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"

    _TOKENIZER = BertTokenizer.from_pretrained(repo)
    _MODEL = BertForSequenceClassification.from_pretrained(repo)

    print(">>> IDEA 中文情感模型加载成功！")

    return _TOKENIZER, _MODEL
