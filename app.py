from flask import Flask, render_template, request
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator
from langdetect import detect
from lime.lime_text import LimeTextExplainer
import shap
import os
import re

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from model import DepressionModel

torch.set_num_threads(2)

app = Flask(__name__)

############################################
# CLASS NAMES
############################################

CLASS_NAMES = [
    "Not Depressed",
    "Mild",
    "Moderate",
    "Severe"
]

############################################
# CLEAR OLD STATIC FILES
############################################

def clear_static_files():
    for f in [
        "static/lime_text.png",
        "static/lime_text.html",
        "static/lime_image.png",
        "static/shap_text.png",
        "static/temp_upload.jpg",
    ]:
        if os.path.exists(f):
            os.remove(f)


############################################
# SEVERITY SCORE ON SCALE OF 5
############################################

def compute_severity(probs):
    return round(
        float(probs[0]*0 + probs[1]*1.67 + probs[2]*3.33 + probs[3]*5.0),
        2
    )


############################################
# SEVERITY KEYWORD RULES
############################################

SEVERITY_RULES = [
    (r"\bwant to die\b",           3),
    (r"\bwanna die\b",             3),
    (r"\bkill myself\b",           3),
    (r"\bend my life\b",           3),
    (r"\bsuicid",                  3),
    (r"\bself.harm\b",             3),
    (r"\bcut myself\b",            3),
    (r"\bno reason to live\b",     3),
    (r"\bnot worth living\b",      3),
    (r"\bbetter off dead\b",       3),
    (r"\bdon.t want to be alive",  3),
    (r"\bwish i was dead\b",       3),
    (r"\bthinking about suicide",  3),
    (r"\bhopeless\b",              2),
    (r"\bcan.t go on\b",           2),
    (r"\bno point\b",              2),
    (r"\bnumb\b",                  2),
    (r"\bempty inside\b",          2),
    (r"\bbroken inside\b",         2),
    (r"\bcan.t function\b",        2),
    (r"\bgiving up\b",             2),
]

def apply_severity_rules(text, model_pred_index, model_probs):
    text_lower = text.lower()
    min_floor  = model_pred_index
    has_crisis_keyword = False

    for pattern, floor in SEVERITY_RULES:
        if re.search(pattern, text_lower):
            min_floor = max(min_floor, floor)
            if floor == 3:
                has_crisis_keyword = True

    # Guard: if model predicts Severe but NO crisis-level keyword matched,
    # cap at Moderate. Prevents mild text like "i am not fine" being
    # over-classified as Severe.
    if model_pred_index == 3 and not has_crisis_keyword:
        new_probs    = model_probs.copy()
        new_probs[2] += new_probs[3]
        new_probs[3]  = 0.0
        new_probs     = new_probs / new_probs.sum()
        return 2, new_probs

    if min_floor > model_pred_index:
        new_probs                   = model_probs.copy()
        new_probs[min_floor]       += new_probs[model_pred_index]
        new_probs[model_pred_index] = 0.0
        new_probs                   = new_probs / new_probs.sum()
        return min_floor, new_probs

    return model_pred_index, model_probs


############################################
# TRANSLATION
############################################

def translate_to_english(text):
    try:
        lang = detect(text)
        print(f"Detected language: {lang}")
        if lang != "en":
            text = GoogleTranslator(source="auto", target="en").translate(text)
            print(f"Translated to: {text}")
    except Exception as e:
        print(f"Translation error: {e}")
    return text


############################################
# TEXT MODEL — DUAL ENCODER (BERT + MuRIL)
############################################

print("Loading DepressionModel (BERT + MuRIL dual encoder)...")
text_model = DepressionModel()
text_model.load("depression_model.pt")

# Set all components to eval mode
text_model.eval()
text_model.encoder.eval()
text_model.encoder.bert_encoder.model.eval()
text_model.encoder.muril_encoder.model.eval()

print("DepressionModel loaded successfully.")


############################################
# TEXT PREDICTION
############################################

def predict_single_text(text):
    text_model.eval()
    text_model.encoder.eval()
    text_model.encoder.bert_encoder.model.eval()
    text_model.encoder.muril_encoder.model.eval()

    with torch.no_grad():
        emb           = text_model.encoder([text])
        emb           = emb.unsqueeze(0).to(text_model.device)
        logits, score = text_model.forward(emb)

        # Temperature scaling to soften overconfident predictions.
        temperature   = 1.8
        logits_scaled = logits / temperature
        probs         = torch.softmax(logits_scaled, dim=-1).cpu().numpy()[0]
        severity      = float(score.cpu().numpy()[0][0])
    return probs, severity


def predict_text_proba(texts):
    """LIME / SHAP wrapper — returns array (n, 4)"""
    results = []
    for t in texts:
        probs, _ = predict_single_text(t)
        results.append(probs)
    return np.array(results)


############################################
# WORD IMPORTANCE GRAPH (custom masking)
############################################

def generate_lime_text_graph(text, pred_class_index, save_path):
    words         = text.split()
    base_probs, _ = predict_single_text(text)
    base_score    = base_probs[pred_class_index]

    importances = []
    for i, word in enumerate(words):
        masked_words    = words[:i] + ["[MASK]"] + words[i+1:]
        masked_text     = " ".join(masked_words)
        masked_probs, _ = predict_single_text(masked_text)
        masked_score    = masked_probs[pred_class_index]
        importances.append(base_score - masked_score)

    colors = ["green" if v >= 0 else "red" for v in importances]

    fig, ax = plt.subplots(figsize=(8, max(3, len(words) * 0.6)))
    bars = ax.barh(words, importances, color=colors)
    ax.set_xlabel("Impact on prediction")
    ax.set_title(
        f"Word Importance for class: {CLASS_NAMES[pred_class_index]}",
        fontsize=12
    )
    ax.axvline(x=0, color="black", linewidth=0.8)

    for bar, val in zip(bars, importances):
        ax.text(
            val + (0.0005 if val >= 0 else -0.0005),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()


############################################
# INTERACTIVE LIME HTML
############################################

def generate_lime_html(text, pred_class_index, save_path):
    try:
        explainer = LimeTextExplainer(class_names=CLASS_NAMES)

        exp = explainer.explain_instance(
            text,
            predict_text_proba,
            num_features=10,
            num_samples=100,
            labels=[pred_class_index]
        )

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(exp.as_html())

        print("Interactive LIME HTML saved.")
        return True

    except Exception as e:
        print("LIME HTML ERROR:", e)
        return False


############################################
# SHAP TEXT EXPLANATION
############################################

def generate_shap_text_graph(text, pred_class_index, save_path):
    try:
        words = text.split()
        n     = len(words)

        if n == 0:
            return False

        def shap_predict(mask_matrix):
            results = []
            for row in mask_matrix:
                masked_words = [
                    words[i] if row[i] == 1 else "[MASK]"
                    for i in range(n)
                ]
                masked_text = " ".join(masked_words)
                probs, _    = predict_single_text(masked_text)
                results.append(probs)
            return np.array(results)

        background  = np.zeros((1, n))
        explainer   = shap.KernelExplainer(shap_predict, background)
        sample      = np.ones((1, n))
        shap_values = explainer.shap_values(sample, nsamples=100, silent=True)

        sv_for_class = shap_values[pred_class_index][0]

        colors = ["green" if v >= 0 else "red" for v in sv_for_class]

        fig, ax = plt.subplots(figsize=(8, max(3, n * 0.6)))
        bars = ax.barh(words, sv_for_class, color=colors)
        ax.set_xlabel("SHAP value (impact on prediction probability)")
        ax.set_title(
            f"SHAP Word Importance — class: {CLASS_NAMES[pred_class_index]}",
            fontsize=12
        )
        ax.axvline(x=0, color="black", linewidth=0.8)

        for bar, val in zip(bars, sv_for_class):
            ax.text(
                val + (0.0005 if val >= 0 else -0.0005),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close()

        print("SHAP text graph saved.")
        return True

    except Exception as e:
        print("SHAP TEXT ERROR:", e)
        return False


############################################
# IMAGE PREDICTION  ← FIXED
############################################

def predict_image_depression(image_path):
    try:
        from deepface import DeepFace

        result = DeepFace.analyze(
            img_path=image_path,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True
        )

        if isinstance(result, list):
            result = result[0]

        emotions = result["emotion"]
        print("Emotions detected:", emotions)

    except Exception as e:
        print("DeepFace ERROR:", str(e))
        emotions = {
            "happy":    0.0,
            "sad":      0.0,
            "fear":     0.0,
            "disgust":  0.0,
            "angry":    0.0,
            "neutral":  100.0,
            "surprise": 0.0
        }

    # Normalise to 0-1
    happy    = emotions.get("happy",    0) / 100.0
    sad      = emotions.get("sad",      0) / 100.0
    fear     = emotions.get("fear",     0) / 100.0
    disgust  = emotions.get("disgust",  0) / 100.0
    angry    = emotions.get("angry",    0) / 100.0
    neutral  = emotions.get("neutral",  0) / 100.0
    surprise = emotions.get("surprise", 0) / 100.0

    # ── FIXED scoring — reduced Not Depressed bias ──
    # Not Depressed: only strong happiness drives this
    s0 = happy * 1.5 + surprise * 0.3 - sad * 0.5 - fear * 0.5

    # Mild: neutral + low sad dominates
    s1 = neutral * 1.2 + sad * 1.5 + angry * 0.3 - happy * 0.8

    # Moderate: significant sad + fear + angry
    s2 = sad * 1.8 + fear * 1.5 + angry * 1.2 + neutral * 0.3 - happy * 1.0

    # Severe: extreme negative — fear + disgust + sad all high
    s3 = sad * 1.5 + fear * 2.0 + disgust * 2.0 + angry * 1.0 - happy * 1.5

    # Clip negatives then softmax
    raw = np.array([max(s0, 0.01), max(s1, 0.01),
                    max(s2, 0.01), max(s3, 0.01)], dtype=np.float32)

    # Higher temperature = less overconfident
    temperature = 1.2
    raw_scaled  = raw / temperature
    exp_raw     = np.exp(raw_scaled - raw_scaled.max())
    probs       = exp_raw / exp_raw.sum()

    pred = int(np.argmax(probs))
    print(f"Image probs: {probs}  →  {CLASS_NAMES[pred]}")

    return pred, probs, emotions


############################################
# EMOTION BAR CHART
############################################

def generate_emotion_graph(emotions, pred_class, save_path):
    labels = list(emotions.keys())
    values = [emotions[k] for k in labels]

    color_map = {
        "happy":    "#2ecc71",
        "neutral":  "#95a5a6",
        "surprise": "#f39c12",
        "sad":      "#3498db",
        "fear":     "#9b59b6",
        "disgust":  "#e74c3c",
        "angry":    "#c0392b",
    }
    colors = [color_map.get(l, "#aaa") for l in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Confidence (%)")
    ax.set_title(
        f"Detected Facial Emotions → Predicted: {CLASS_NAMES[pred_class]}",
        fontsize=12
    )
    ax.set_ylim(0, 110)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()


############################################
# MAIN ROUTE
############################################

@app.route("/", methods=["GET", "POST"])
def index():

    prediction      = None
    severity        = None
    translated      = None
    original_lang   = None
    has_lime_text   = False
    has_lime_html   = False
    has_lime_image  = False
    has_shap_text   = False
    image_error     = None

    if request.method == "POST":

        clear_static_files()
        os.makedirs("static", exist_ok=True)

        ###################################
        # TEXT INPUT
        ###################################

        if "text" in request.form and request.form["text"].strip():

            text = request.form["text"]

            try:
                original_lang = detect(text)
            except:
                original_lang = "en"

            translated  = translate_to_english(text)
            probs, _    = predict_single_text(translated)
            pred        = int(np.argmax(probs))
            pred, probs = apply_severity_rules(translated, pred, probs)

            prediction = CLASS_NAMES[pred]
            severity   = compute_severity(probs)

            # Word importance bar chart
            try:
                generate_lime_text_graph(
                    translated, pred, "static/lime_text.png"
                )
                has_lime_text = True
            except Exception as e:
                print("LIME GRAPH ERROR:", e)

            # Interactive LIME HTML
            has_lime_html = generate_lime_html(
                translated, pred, "static/lime_text.html"
            )

            # SHAP text explanation
            has_shap_text = generate_shap_text_graph(
                translated, pred, "static/shap_text.png"
            )

        ###################################
        # IMAGE INPUT
        ###################################

        if "image" in request.files and request.files["image"].filename != "":

            file      = request.files["image"]
            image     = Image.open(file).convert("RGB")
            temp_path = "static/temp_upload.jpg"
            image.save(temp_path)

            try:
                pred, probs, emotions = predict_image_depression(temp_path)

                prediction = CLASS_NAMES[pred]
                severity   = compute_severity(probs)

                generate_emotion_graph(emotions, pred, "static/lime_image.png")
                has_lime_image = True

            except Exception as e:
                print("IMAGE ERROR:", e)
                image_error = "Could not process image. Please use a clear facial photo."

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    return render_template(
        "index.html",
        prediction=prediction,
        severity=severity,
        translated=translated,
        original_lang=original_lang,
        has_lime_text=has_lime_text,
        has_lime_html=has_lime_html,
        has_lime_image=has_lime_image,
        has_shap_text=has_shap_text,
        image_error=image_error,
    )


if __name__ == "__main__":
    app.run(debug=True)