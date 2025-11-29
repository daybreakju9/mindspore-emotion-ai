# nlp_backend/__init__.py
"""
情绪分析AI后端模块
基于MindSpore的中文文本情绪分析
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s'
)

from .predict import analyze_text, test_analyze, test_mindspore_operations
from .emotion_model_loader import EmotionModelLoader, emotion_model_loader

__version__ = "1.0.0"
__author__ = "情绪分析团队"
__description__ = "基于MindSpore的中文文本情绪分析系统"

__all__ = [
    'analyze_text',
    'test_analyze',
    'test_mindspore_operations',
    'EmotionModelLoader',
    'emotion_model_loader'
]

print(f"情绪分析模块已加载 版本 {__version__}")