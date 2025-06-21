import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

# 类别映射
CLASS_MAP = {
    0: "Yellow",  # 黄灯
    1: "Red",    # 红灯
    2: "Green"   # 绿灯
}

# 中文名称映射
CHINESE_NAMES = {
    "Yellow": "黄灯",
    "Red": "红灯",
    "Green": "绿灯"
}

# 状态颜色映射
COLOR_MAP = {
    "Yellow": (0, 255, 255),  # 黄色
    "Red": (0, 0, 255),       # 红色
    "Green": (0, 255, 0)      # 绿色
}

def load_model(model_path):
    """加载训练好的模型"""
    model = YOLO(model_path)
    return model

def analyze_image(model, image_path, output_dir):
    """分析单张图像并保存结果"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    # 使用模型进行预测
    results = model(image, conf=0.3)  # 置信度阈值设为0.3
    
    # 获取检测结果
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            class_name = CLASS_MAP[cls_id]
            conf = float(box.conf)
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            
            detections.append({
                "class": class_name,
                "bbox": bbox,
                "conf": conf
            })
    
    # 获取主要状态（置信度最高的检测）
    main_state = None
    if detections:
        # 按置信度排序
        detections.sort(key=lambda x: x["conf"], reverse=True)
        main_state = detections[0]["class"]
    
    # 输出结果
    print(f"图像: {os.path.basename(image_path)}")
    if main_state:
        print(f"红绿灯状态: {CHINESE_NAMES[main_state]}")
    else:
        print("未检测到红绿灯")
    print("-" * 50)
    
    # 绘制检测结果
    annotated_image = draw_detections(image, detections, main_state)
    
    # 保存结果图像
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, annotated_image)
    
    return main_state

def draw_detections(image, detections, main_state):
    """在图像上绘制检测结果"""
    # 将OpenCV图像转换为PIL图像（更好的中文支持）
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        # 尝试加载中文字体
        font = ImageFont.truetype("simhei.ttf", 30)
        state_font = ImageFont.truetype("simhei.ttf", 60)
    except:
        # 回退到默认字体
        font = ImageFont.load_default()
        state_font = ImageFont.load_default()
    
    # 绘制所有检测框
    for det in detections:
        class_name = det["class"]
        bbox = det["bbox"]
        conf = det["conf"]
        
        # 绘制边界框
        color = COLOR_MAP.get(class_name, (255, 255, 255))
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], 
                      outline=color, width=4)
        
        # 绘制标签
        label = f"{CHINESE_NAMES[class_name]} {conf:.2f}"
        text_size = font.getbbox(label)
        draw.rectangle([(bbox[0], bbox[1] - text_size[3] + text_size[1]), 
                       (bbox[0] + text_size[2] - text_size[0], bbox[1])], 
                      fill=color)
        draw.text((bbox[0], bbox[1] - text_size[3] + text_size[1]), 
                 label, fill=(0, 0, 0), font=font)
    
    # 在图像顶部添加状态信息
    if main_state:
        state_text = f"当前状态: {CHINESE_NAMES[main_state]}"
        state_color = COLOR_MAP[main_state]
        
        # 获取文本尺寸
        text_size = state_font.getbbox(state_text)
        img_width, img_height = img_pil.size
        
        # 绘制背景
        draw.rectangle([(0, 0), (img_width, text_size[3] - text_size[1] + 50)], 
                      fill=state_color)
        
        # 绘制状态文本
        draw.text(((img_width - text_size[2] + text_size[0]) // 2, 10), 
                 state_text, fill=(0, 0, 0), font=state_font)
    
    # 将PIL图像转换回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def test_model(model_path, test_dir, output_dir):
    """测试模型在指定目录的所有图像上"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(model_path)
    print(f"已加载模型: {model_path}")
    
    # 获取所有测试图像
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                  if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_paths:
        print(f"在目录 {test_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_paths)} 张测试图像")
    print("=" * 80)
    
    # 分析每张图像
    state_counts = defaultdict(int)
    for image_path in image_paths:
        state = analyze_image(model, image_path, output_dir)
        if state:
            state_counts[state] += 1
    
    # 打印总体统计
    print("\n测试结果汇总:")
    print("=" * 80)
    print(f"测试图像总数: {len(image_paths)}")
    
    total_detected = sum(state_counts.values())
    if total_detected > 0:
        print(f"检测到红绿灯的图像: {total_detected} ({total_detected/len(image_paths)*100:.1f}%)")
        
        print("\n红绿灯状态分布:")
        for state, count in state_counts.items():
            print(f"  {CHINESE_NAMES[state]}: {count}个 ({count/total_detected*100:.1f}%)")
    else:
        print("未在任何图像中检测到红绿灯")
    
    print("=" * 80)
    print(f"所有标记后的图像已保存到: {output_dir}")

if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = "D:/All_save/Code_save/yolov8_trainmodel/runs/lamp/train/weights/last.pt"
    TEST_DIR = "D:\All_save\Code_save\yolov8_trainmodel\\allpic\crossroadlamp\images"  # 测试图像目录
    OUTPUT_DIR = "D:\All_save\Code_save\yolov8_trainmodel\\val_save\lamp"  # 结果保存目录
    
    # 运行测试
    test_model(MODEL_PATH, TEST_DIR, OUTPUT_DIR)