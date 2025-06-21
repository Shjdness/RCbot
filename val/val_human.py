import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

# 类别映射
CLASS_MAP = {
    0: "human",
    1: "career"
}

# 中文名称映射
CHINESE_NAMES = {
    "human": "普通人",
    "career": "职业人"
}

# 颜色映射
COLOR_MAP = {
    "human": (0, 165, 255),    # 橙色
    "career": (255, 0, 0)    # 深蓝色
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
    
    # 统计检测结果
    count_human = 0
    count_career = 0
    detections = []
    
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            class_name = CLASS_MAP[cls_id]
            conf = float(box.conf)
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            
            if class_name == "human":
                count_human += 1
            elif class_name == "career":
                count_career += 1
            
            detections.append({
                "class": class_name,
                "bbox": bbox,
                "conf": conf
            })
    
    # 输出结果
    print(f"图像: {os.path.basename(image_path)}")
    print(f"检测到 {count_human + count_career} 个人")
    print(f"- 普通人(human): {count_human}个")
    print(f"- 职业人(career): {count_career}个")
    print("-" * 50)
    
    # 绘制检测结果
    annotated_image = draw_detections(image, detections, count_human, count_career)
    
    # 保存结果图像
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, annotated_image)
    
    return {
        "total": count_human + count_career,
        "human": count_human,
        "career": count_career
    }

def draw_detections(image, detections, count_human, count_career):
    """在图像上绘制检测结果"""
    # 将OpenCV图像转换为PIL图像（更好的中文支持）
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        # 尝试加载中文字体
        font = ImageFont.truetype("simhei.ttf", 24)
    except:
        # 回退到默认字体
        font = ImageFont.load_default()
    
    # 绘制所有检测框
    for det in detections:
        class_name = det["class"]
        bbox = det["bbox"]
        conf = det["conf"]
        
        # 绘制边界框
        color = COLOR_MAP.get(class_name, (255, 255, 255))
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], 
                      outline=color, width=2)
        
        # 绘制标签
        label = f"{CHINESE_NAMES[class_name]} {conf:.2f}"
        text_size = font.getbbox(label)
        draw.rectangle([(bbox[0], bbox[1] - text_size[3] + text_size[1]), 
                       (bbox[0] + text_size[2] - text_size[0], bbox[1])], 
                      fill=color)
        draw.text((bbox[0], bbox[1] - text_size[3] + text_size[1]), 
                 label, fill=(255, 255, 255), font=font)
    
    # 在图像顶部添加汇总信息
    summary = f"总人数: {count_human + count_career} | 普通人: {count_human} | 职业人: {count_career}"
    
    # 绘制汇总信息
    text_size = font.getbbox(summary)
    draw.rectangle([(10, 10), (10 + text_size[2] - text_size[0] + 20, 10 + text_size[3] - text_size[1] + 20)], 
                  fill=(0, 0, 0, 180))
    draw.text((20, 20), summary, fill=(255, 255, 255), font=font)
    
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
    all_results = []
    for image_path in image_paths:
        result = analyze_image(model, image_path, output_dir)
        if result:
            all_results.append(result)
    
    # 打印总体统计
    print("\n测试结果汇总:")
    print("=" * 80)
    
    total_people = 0
    total_human = 0
    total_career = 0
    
    for result in all_results:
        total_people += result["total"]
        total_human += result["human"]
        total_career += result["career"]
    
    print(f"测试图像总数: {len(all_results)}")
    print(f"检测到总人数: {total_people}")
    print(f"- 普通人(human): {total_human} ({total_human/total_people*100:.1f}%)")
    print(f"- 职业人(career): {total_career} ({total_career/total_people*100:.1f}%)")
    
    print("=" * 80)
    print(f"所有标记后的图像已保存到: {output_dir}")

if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = "D:/All_save/Code_save/yolov8_trainmodel/runs/human/train/weights/last.pt"
    TEST_DIR = "D:\All_save\Code_save\yolov8_trainmodel\\allpic\\human\images"  # 测试图像目录
    OUTPUT_DIR = "D:\All_save\Code_save\yolov8_trainmodel\\val_save\human"  # 结果保存目录
    
    # 运行测试
    test_model(MODEL_PATH, TEST_DIR, OUTPUT_DIR)