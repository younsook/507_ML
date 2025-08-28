from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#from ch008.tokenzier import tw_tokenzier
import joblib

# E:\work_machinelearning\ch008\tokenzier.py
import re

# pipe = Pipeline([
#     ("tfidf", TfidfVectorizer(tokenizer=tw_tokenzier)),
#     ("clf", LogisticRegression(max_iter=1000))
# ])
# train...
#joblib.dump(pipe, "model/lr_pipe.pkl")

def tw_tokenzier(text: str):
    # ※ 실제 학습 때 쓰던 전처리/토큰화 로직으로 바꾸세요
    if text is None:
        return []
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text.split()
