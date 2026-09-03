# dual_predictor.py

import joblib


class DualOutputPredictor:

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
        X = self._encode(df)
        return self.time_model.predict(X), self.mem_model.predict(X)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
