import io
import os
import wave

from flask import Flask, jsonify, request, send_file, send_from_directory
from piper import PiperVoice

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "hi_IN-rohan-medium.onnx")

app = Flask(__name__)

# Model ek baar load hota hai (server start par)
voice = PiperVoice.load(MODEL_PATH)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Text khaali hai"}), 400
    if len(text) > 2000:
        return jsonify({"error": "Text 2000 characters se chhota rakho"}), 400

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        try:
            voice.synthesize_wav(text, wav_file)      # piper-tts >= 1.3
        except AttributeError:
            voice.synthesize(text, wav_file)          # purana version
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
