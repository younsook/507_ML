from flask import Flask, render_template, request, jsonify
import joblib
from konlpy.tag import Okt
from ch008.tokenzier import tw_tokenzier


app = Flask(__name__)

okt = Okt()

def tw_tokenizer(text):
    tokenzier_ko = okt.morphs(text)
    return tokenzier_ko


try:
    from ch008.tokenzier import tw_tokenzier
    model = joblib.load("model/lr_v1.pkl")
    vec = joblib.load("model/tfidf_vec_v1.pkl")
except Exception as e:
    print("모델 로드 중 오류 발생: {str(e)}")
    raise

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "텍스트가 잘못됐어요"}), 400
    text = data["text"]
    if not text.strip():
        return jsonify({"error": "이건..텍스트가 잘못됐어요"}), 400
    text_tfidf = vec.transform([text])
    predict = model.predict(text_tfidf)[0]
    return jsonify({"emotion": predict})

   
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)