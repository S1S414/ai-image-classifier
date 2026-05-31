"""
目标检测系统 - Flask 后端服务 (基于 YOLOv8 + Grounding DINO 双模型)
"""
import os
import io
import gc
import base64
import time
import logging
from datetime import datetime
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import cv2
import numpy as np
import torch

# HuggingFace 镜像（本地开发时启用，云端部署走官方源）
# os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

# 加载环境变量
load_dotenv()

# 配置
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
BOX_THRESHOLD = float(os.getenv('BOX_THRESHOLD', 0.35))
TEXT_THRESHOLD = float(os.getenv('TEXT_THRESHOLD', 0.25))
# COCO 80 类全景列表作为默认提示词，覆盖绝大多数常见物体
COCO_80_PROMPT = 'person . bicycle . car . motorcycle . airplane . bus . train . truck . boat . traffic light . fire hydrant . stop sign . parking meter . bench . bird . cat . dog . horse . sheep . cow . elephant . bear . zebra . giraffe . backpack . umbrella . handbag . tie . suitcase . frisbee . skis . snowboard . sports ball . kite . baseball bat . baseball glove . skateboard . surfboard . tennis racket . bottle . wine glass . cup . fork . knife . spoon . bowl . banana . apple . sandwich . orange . broccoli . carrot . pizza . donut . cake . chair . couch . potted plant . bed . dining table . toilet . tv . laptop . mouse . remote . keyboard . cell phone . microwave . oven . toaster . sink . refrigerator . book . clock . vase . scissors . teddy bear . hair drier . toothbrush'
TEXT_PROMPT = os.getenv('TEXT_PROMPT', COCO_80_PROMPT)

# Grounding DINO 配置
GDINO_CONFIG = os.path.join(os.path.dirname(__file__), 'GroundingDINO', 'groundingdino', 'config', 'GroundingDINO_SwinT_OGC.py')
GDINO_WEIGHTS = os.path.join(os.path.dirname(__file__), 'weights', 'groundingdino_swint_ogc.pth')

# YOLOv8 模型配置
YOLO_MODEL_NAME = 'yolov8s.pt'  # 平衡版：21.5 MB

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 初始化 Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 全局模型
model = None  # Grounding DINO
yolo_model = None  # YOLOv8


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    """加载 Grounding DINO 和 YOLOv8 模型"""
    global model, yolo_model

    print("=" * 50)
    print("正在加载模型...")
    
    # ===== 加载 YOLOv8（GPU）=====
    print(f"正在加载 YOLOv8 模型 ({YOLO_MODEL_NAME})...")
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(YOLO_MODEL_NAME)
        if torch.cuda.is_available():
            yolo_model.to('cuda')
            print(f"✅ YOLOv8 模型加载成功 (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print("⚠️  CUDA 不可用，YOLOv8 将使用 CPU")
    except Exception as e:
        print(f"❌ YOLOv8 模型加载失败: {e}")
        yolo_model = None
    
    # ===== 加载 Grounding DINO（CPU）=====
    print("\n正在加载 Grounding DINO 模型...")
    print(f"配置文件: {GDINO_CONFIG}")
    print(f"权重文件: {GDINO_WEIGHTS}")
    print(f"检测提示: {TEXT_PROMPT}")
    print("=" * 50)

    # 检查并自动下载权重文件
    if not os.path.exists(GDINO_WEIGHTS):
        print(f"权重文件不存在，正在从 HuggingFace 下载...")
        os.makedirs(os.path.dirname(GDINO_WEIGHTS), exist_ok=True)
        import urllib.request
        url = "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth"
        try:
            urllib.request.urlretrieve(url, GDINO_WEIGHTS)
            print(f"✅ 权重文件下载完成")
        except Exception as e:
            raise FileNotFoundError(
                f"权重文件下载失败: {e}\n"
                f"请手动从以下地址下载并放到 weights 文件夹:\n"
                f"https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            )

    try:
        from groundingdino.util.inference import load_model as gdino_load_model
        model = gdino_load_model(GDINO_CONFIG, GDINO_WEIGHTS, device='cpu')
        print("✅ Grounding DINO 模型加载成功!")
    except Exception as e:
        print(f"❌ Grounding DINO 模型加载失败: {e}")
        raise

    print("=" * 50)
    print("所有模型准备就绪!")
    print("=" * 50)


def detect_objects_yolov8(image_bytes):
    """使用 YOLOv8 (GPU) 检测图像中的物体（COCO 80 类）"""
    global yolo_model
    
    if yolo_model is None:
        raise RuntimeError("YOLOv8 模型未加载，请访问 /load_yolo 重新加载")
    
    # 读取图像
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # YOLOv8 推理（GPU）
    results = yolo_model(pil_image)
    
    # 绘制检测框
    draw = ImageDraw.Draw(pil_image)
    
    # 颜色列表
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    
    detections = []
    
    # 解析 YOLOv8 结果
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 获取坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = r.names[cls]
            
            color_idx = hash(label) % len(colors)
            color = colors[color_idx]
            
            # 绘制矩形框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 绘制标签
            text_label = f"{label} {conf:.2f}"
            text_bbox = draw.textbbox((x1, y1), text_label)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1), text_label, fill=(255, 255, 255))
            
            detections.append({
                'class': label,
                'confidence': round(conf * 100, 2),
                'bbox': [round(x) for x in [x1, y1, x2, y2]]
            })
    
    # 转换为 base64
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG', quality=95)
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    
    return img_base64, detections


def detect_objects_gdino(image_bytes, custom_prompt=None):
    """使用 Grounding DINO (CPU) 检测图像中的物体（开放词汇）"""
    global model

    from groundingdino.util.inference import predict as gdino_predict

    # 使用自定义提示词或默认提示词
    prompt = custom_prompt if custom_prompt else TEXT_PROMPT

    # 读取图像
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 预处理图像为 Tensor
    import groundingdino.datasets.transforms as T

    # 预处理图像为 Tensor
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(pil_image, None)

    boxes, logits, phrases = gdino_predict(
        model=model,
        image=image_transformed,
        caption=prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device='cpu'
    )

    # Grounding DINO 返回的 boxes 是归一化坐标 (0~1)，需要乘以图片宽高
    img_w, img_h = pil_image.size

    # 绘制检测框
    draw = ImageDraw.Draw(pil_image)

    # 颜色列表
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]

    detections = []

    for i in range(len(boxes)):
        # 归一化坐标 → 像素坐标
        cx, cy, w, h = boxes[i].tolist()
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h

        # 确保坐标合法
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        conf = float(logits[i])
        label = phrases[i].strip()

        color_idx = hash(label) % len(colors)
        color = colors[color_idx]

        # 绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 绘制标签
        text_label = f"{label} {conf:.2f}"
        text_bbox = draw.textbbox((x1, y1), text_label)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1), text_label, fill=(255, 255, 255))

        detections.append({
            'class': label,
            'confidence': round(conf * 100, 2),
            'bbox': [round(x) for x in [x1, y1, x2, y2]]
        })

    # 转换为 base64
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG', quality=95)
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    return img_base64, detections


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def handle_detect():
    """处理图像检测请求（双模型策略）"""
    # 请求日志
    request_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(request)}"
    print(f"[{request_id}] 新检测请求 | 来源: {request.remote_addr}")
    start_time = time.time()

    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if file and allowed_file(file.filename):
        try:
            # 读取图像
            image_bytes = file.read()

            # 读取自定义提示词（可选）
            custom_prompt = request.form.get('prompt', '').strip() or None
            
            # ===== prompt 参数验证 =====
            MAX_PROMPT_LENGTH = 500
            if custom_prompt and len(custom_prompt) > MAX_PROMPT_LENGTH:
                return jsonify({'error': f'提示词过长，最多 {MAX_PROMPT_LENGTH} 字符'}), 400
            
            # ===== 双模型策略 =====
            if not _models_ready:
                return jsonify({'error': '模型正在加载中，请稍后再试', 'status': 'loading'}), 503
                
            if custom_prompt is None or custom_prompt == '':
                # 空提示词 → 使用 YOLOv8 (GPU) 检测 COCO 80 类
                print(f"[{request_id}] 🔵 使用 YOLOv8 (GPU) 检测 COCO 80 类")
                result_image, detections = detect_objects_yolov8(image_bytes)
            else:
                # 有自定义提示词 → 使用 Grounding DINO (CPU) 开放词汇检测
                print(f"[{request_id}] 🟢 使用 Grounding DINO (CPU) 检测: {custom_prompt[:50]}...")
                result_image, detections = detect_objects_gdino(image_bytes, custom_prompt)

            elapsed = time.time() - start_time
            print(f"[{request_id}] ✅ 完成 - 检测到 {len(detections)} 个目标，耗时 {elapsed:.2f}s")

            return jsonify({
                'success': True,
                'image': result_image,
                'detections': detections,
                'count': len(detections)
            })

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[{request_id}] ❌ 检测失败 ({elapsed:.2f}s): {str(e)}")
            import traceback
            tb = traceback.format_exc()
            print(f"=== 检测错误 ===")
            print(tb)
            print(f"================")
            # 生产环境不返回traceback，防止信息泄露
            return jsonify({'error': '检测失败，请稍后重试'}), 500

    return jsonify({'error': '不支持的文件类型'}), 400


@app.route('/gpu_status')
def gpu_status():
    """查看 GPU 状态（YOLOv8 是否加载）"""
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'yolo_loaded': yolo_model is not None,
        'gdino_loaded': model is not None
    }
    
    if torch.cuda.is_available():
        gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
        gpu_info['gpu_memory_allocated_mb'] = round(torch.cuda.memory_allocated(0) / 1024 / 1024, 2)
        gpu_info['gpu_memory_reserved_mb'] = round(torch.cuda.memory_reserved(0) / 1024 / 1024, 2)
    
    return jsonify(gpu_info)


@app.route('/unload_yolo', methods=['POST'])
def unload_yolo():
    """手动卸载 YOLOv8（释放 GPU 显存）"""
    global yolo_model
    
    if yolo_model is None:
        return jsonify({'status': 'YOLOv8 未加载，无需卸载'})
    
    try:
        del yolo_model
        yolo_model = None
        gc.collect()  # 强制垃圾回收
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return jsonify({'status': '✅ YOLOv8 已从 GPU 卸载，显存已释放'})
    except Exception as e:
        return jsonify({'error': f'卸载失败: {str(e)}'}), 500


@app.route('/load_yolo', methods=['POST'])
def load_yolo():
    """重新加载 YOLOv8 到 GPU"""
    global yolo_model
    
    if yolo_model is not None:
        return jsonify({'status': 'YOLOv8 已加载，无需重复加载'})
    
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(YOLO_MODEL_NAME)
        if torch.cuda.is_available():
            yolo_model.to('cuda')
            return jsonify({'status': f'✅ YOLOv8 已重新加载到 GPU ({torch.cuda.get_device_name(0)})'})
        else:
            return jsonify({'status': '⚠️  CUDA 不可用，YOLOv8 加载到 CPU'})
    except Exception as e:
        return jsonify({'error': f'加载失败: {str(e)}'}), 500


@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok' if _models_ready else 'loading',
        'models': {
            'grounding_dino': model is not None,
            'yolov8': yolo_model is not None
        },
        'ready': _models_ready
    })


# 启动时后台加载模型（不阻塞 gunicorn 启动，HF 健康检查要求 30min 内就绪）
import threading
_models_ready = False

def _background_load_models():
    global _models_ready
    try:
        load_models()
        _models_ready = True
        print("✅ 所有模型后台加载完成")
    except Exception as e:
        print(f"❌ 模型后台加载失败: {e}")

threading.Thread(target=_background_load_models, daemon=True).start()

if __name__ == '__main__':
    # 启动服务（兼容 HuggingFace Spaces / Render / 本地）
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('PORT', os.getenv('FLASK_PORT', 5000)))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

    print(f"\n🚀 目标检测服务已启动 (双模型模式)")
    print(f"访问地址: http://localhost:{port}")
    print(f"上传目录: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"模型:")
    print(f"  - YOLOv8: GPU (RTX 3050) - 检测 COCO 80 类")
    print(f"  - Grounding DINO: CPU - 开放词汇检测")
    print(f"检测策略:")
    print(f"  - 空提示词 → YOLOv8 (GPU) 快速检测")
    print(f"  - 自定义提示词 → Grounding DINO (CPU) 精准检测")
    print(f"框阈值: {BOX_THRESHOLD} | 文本阈值: {TEXT_THRESHOLD}")
    print("\n按 Ctrl+C 停止服务\n")

    app.run(host=host, port=port, debug=debug)
