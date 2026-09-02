# dual_predictor.py
"""MemX 双输出预测器（论文 §3.2.2）。

把训练好的时间模型和显存模型封装成统一接口：一次 predict 调用同时返回
iteration time 和 peak memory，避免搜索阶段对每个候选配置分别调用两次模型
（对应论文 "Reducing Prediction Overhead in Search" 中的模型架构级优化）。

注意：本模块必须保持可 import（不要只在 __main__ 里定义该类），
否则 joblib 序列化后的 .pkl 在其他脚本中无法反序列化。
"""
import joblib


class DualOutputPredictor:
    """XGBoost 双输出预测器：predict(df) -> (time_pred, mem_pred)。"""

    def __init__(self, time_model, mem_model, label_encoders):
        self.time_model = time_model
        self.mem_model = mem_model
        self.label_encoders = label_encoders

    def _encode(self, df):
        df = df.copy()
        for col, le in self.label_encoders.items():
            df[col] = le.transform(df[col])
        return df

    def predict(self, df):
        """一次调用返回 (时间预测, 显存预测) 两个 numpy 数组。"""
        X = self._encode(df)
        return self.time_model.predict(X), self.mem_model.predict(X)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
