import os
import json
import mimetypes
from uuid import uuid4

from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import requests

from google import genai
from google.genai import types

# ========= 基本設定 =========

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# ========= API Key =========

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("請先設定環境變數 GOOGLE_API_KEY")

if not STABILITY_API_KEY:
    raise RuntimeError("請先設定環境變數 STABILITY_API_KEY")

gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"  # 可以視情況調整


# ========= 工具函式 =========

def allowed_file(filename: str) -> bool:
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def get_mime_type(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "image/png"


def get_text_bboxes_with_gemini(image_path: str):
    """
    用 Gemini 分析圖片文字區域，回傳:
      - bboxes: [(x1, y1, x2, y2), ...] 像素座標
      - raw_text: Gemini 的原始輸出（方便 debug）
    """

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    mime_type = get_mime_type(image_path)

    prompt = """
你是一個「非常嚴格」的 OCR + 版面分析工具，
目標是：把圖片上所有「看起來像文字或數字」的區域全部框出來。

請偵測這張圖片上所有你能辨識到的文字區塊，回傳 *純 JSON*，格式如下：

[
  {
    "text": "文字內容",
    "bbox": {
      "x_min": 123,
      "y_min": 45,
      "x_max": 456,
      "y_max": 78
    }
  },
  ...
]

規則：
1. bbox 座標請使用 0~1000 的整數，代表相對於圖片寬/高的比例。
2. x_min, x_max 是水平方向；y_min, y_max 是垂直方向。
3. 請把「所有路名、地名、店名、數字、地標名稱」都當成文字，一律框起來。
4. 寧可多框，不要漏框；只要看起來像字或數字，就當成文字。
5. 回傳必須是合法 JSON 陣列，不要在 JSON 外多加任何說明文字。
"""


    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            ),
            prompt
        ]
    )

    raw_text = (response.text or "").strip()

    # 如果模型用 ```json ... ``` 包起來，清掉
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # 解析失敗就不做 inpaint，回傳原始結果讓前端顯示
        return [], raw_text

    img = Image.open(image_path)
    width, height = img.size

    bboxes = []
    for item in data:
        bbox = item.get("bbox", {})
        try:
            x_min = int(bbox.get("x_min", 0))
            y_min = int(bbox.get("y_min", 0))
            x_max = int(bbox.get("x_max", 0))
            y_max = int(bbox.get("y_max", 0))
        except (TypeError, ValueError):
            continue

        # 把 0~1000 的比例座標換成像素
        x1 = int(x_min / 1000 * width)
        y1 = int(y_min / 1000 * height)
        x2 = int(x_max / 1000 * width)
        y2 = int(y_max / 1000 * height)
        bboxes.append((x1, y1, x2, y2))

    return bboxes, raw_text


def build_mask_from_bboxes(image_path: str, bboxes, mask_path: str, padding_ratio: float = 0.2):
    """
    根據 bboxes 建立 mask 圖：
      - 黑色 (0): 不修補
      - 白色 (255): 要修補（文字區域）
    padding_ratio: 每個框左右上下再多出來的比例（0.2 = 多出 20%）
    """
    img = Image.open(image_path)
    width, height = img.size

    mask = Image.new("L", (width, height), 0)  # 全黑
    draw = ImageDraw.Draw(mask)

    for x1, y1, x2, y2 in bboxes:
        w = x2 - x1
        h = y2 - y1
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)

        xx1 = max(0, x1 - pad_x)
        yy1 = max(0, y1 - pad_y)
        xx2 = min(width - 1, x2 + pad_x)
        yy2 = min(height - 1, y2 + pad_y)

        draw.rectangle([xx1, yy1, xx2, yy2], fill=255)

    mask.save(mask_path)
    return mask_path



def call_stability_inpaint(image_path: str, mask_path: str, output_path: str, prompt: str = None):
    """
    呼叫 Stability Inpainting API：
      - image: 原圖
      - mask: 白色區塊為要修補的區域
    """
    if prompt is None:
        prompt = (
            "Same style map background, roads and buildings, "
            "but absolutely no text, numbers, labels, watermarks or logos."
        )

    url = "https://api.stability.ai/v2beta/stable-image/edit/inpaint"

    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "image/*",
    }

    files = {
        "image": (
            os.path.basename(image_path),
            open(image_path, "rb"),
            get_mime_type(image_path),
        ),
        "mask": (
            os.path.basename(mask_path),
            open(mask_path, "rb"),
            "image/png",
        ),
    }

    data = {
        "prompt": prompt,
        "negative_prompt": (
            "text, letter, number, label, word, watermark, logo, 字, 漢字, 中文, 英文"
        ),
        "mode": "inpainting",
        "output_format": "png",
    }

    try:
        resp = requests.post(
            url, headers=headers, files=files, data=data, timeout=120
        )

        if resp.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return True, None
        else:
            return False, f"Stability API 錯誤 {resp.status_code}: {resp.text}"
    finally:
        try:
            files["image"][1].close()
            files["mask"][1].close()
        except Exception:
            pass



# ========= 路由 =========

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': '沒有檔案部分'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': '沒有選取的檔案'}), 400

    if not (file and allowed_file(file.filename)):
        return jsonify({'error': '檔案格式不符，只接受 jpg / jpeg / png'}), 400

    # 產生安全且不會重複的檔名
    ext = file.filename.rsplit('.', 1)[1].lower()
    safe_name = secure_filename(file.filename.rsplit('.', 1)[0])
    uid = uuid4().hex
    filename = f"{safe_name}_{uid}.{ext}"

    original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], f"mask_{uid}.png")
    processed_filename = f"processed_{safe_name}_{uid}.png"
    processed_path = os.path.join(app.config['STATIC_FOLDER'], processed_filename)

    # 儲存原始圖片
    file.save(original_path)

    # === 1) 用 Gemini 找文字區塊 ===
    try:
        bboxes, raw_gemini_output = get_text_bboxes_with_gemini(original_path)
    except Exception as e:
        return jsonify({'error': f'呼叫 Gemini 失敗: {e}'}), 500

    if not bboxes:
        # 沒偵測到文字，回前端提示 + Gemini 原始輸出
        return jsonify({
            'success': False,
            'error': 'Gemini 沒有偵測到任何文字區塊，未進行修補。',
            'gemini_raw_output': raw_gemini_output
        }), 200

    # === 2) 根據 bboxes 建 mask ===
    try:
        build_mask_from_bboxes(original_path, bboxes, mask_path)
    except Exception as e:
        return jsonify({'error': f'建立遮罩 (mask) 失敗: {e}'}), 500

    # === 3) 呼叫 Stability 做 Inpainting ===
    ok, err = call_stability_inpaint(original_path, mask_path, processed_path)
    if not ok:
        return jsonify({'error': err}), 500

    processed_url = f"/static/{processed_filename}"

    return jsonify({
        'success': True,
        'processed_image_url': processed_url,
        'bbox_count': len(bboxes),
        'gemini_raw_output': raw_gemini_output
    })


@app.route('/static/<path:filename>')
def send_static_file(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)


if __name__ == '__main__':
    # 0.0.0.0 讓外部（或 SSH tunnel）可以連
    app.run(host='0.0.0.0', port=80, debug=True)
