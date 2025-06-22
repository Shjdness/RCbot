import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import math

# 类别映射
CLASS_MAP = {
    0: "Blue",    # 蓝色可回收物
    1: "Red",     # 红色有害垃圾
    2: "Grey",    # 灰色其他垃圾
    3: "Green",   # 绿色厨余垃圾
    4: "Open",    # 未投放
    5: "Close"    # 已投放
}

# 中文名称映射
CHINESE_NAMES = {
    "Blue": "蓝色可回收物",
    "Red": "红色有害垃圾",
    "Grey": "灰色其他垃圾",
    "Green": "绿色厨余垃圾",
    "Open": "未投放",
    "Close": "已投放"
}

# 垃圾桶类型颜色映射
TRASH_COLOR_MAP = {
    "Blue": (255, 0, 0),      # 蓝色
    "Red": (0, 0, 255),       # 红色
    "Grey": (128, 128, 128),  # 灰色
    "Green": (0, 128, 0)      # 绿色
}

# 状态颜色映射
STATUS_COLOR_MAP = {
    "Open": (0, 255, 0),    # 绿色 - 未投放
    "Close": (0, 0, 255)    # 红色 - 已投放
}

def load_model(model_path):
    """加载训练好的模型"""
    model = YOLO(model_path)
    return model

def calculate_center(bbox):
    """计算边界框的中心点"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def match_trash_bins(trash_bins, statuses):
    """将垃圾桶和状态进行匹配"""
    matched_bins = []
    
    # 对于每个垃圾桶，寻找最近的状态
    for bin in trash_bins:
        bin_center = bin["center"]
        closest_status = None
        min_distance = float('inf')
        
        for status in statuses:
            status_center = status["center"]
            distance = math.sqrt((bin_center[0] - status_center[0])**2 + 
                                (bin_center[1] - status_center[1])**2)
            
            # 考虑状态框应该在垃圾桶框的上方或附近
            vertical_distance = status_center[1] - bin_center[1]
            if vertical_distance < 0:  # 状态在垃圾桶上方
                distance *= 0.8  # 降低距离权重
            
            if distance < min_distance:
                min_distance = distance
                closest_status = status
        
        # 如果找到合适的状态且距离在阈值内
        if closest_status and min_distance < min(bin["width"] * 1.5, bin["height"] * 1.5):
            matched_bins.append({
                "type": bin["type"],
                "bbox": bin["bbox"],
                "status": closest_status["type"],
                "status_bbox": closest_status["bbox"]
            })
    
    return matched_bins

def analyze_image(model, image_path, output_dir):
    """分析单张图像并保存结果"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    # 使用模型进行预测
    results = model(image, conf=0.25)  # 置信度阈值设为0.25
    
    # 分离垃圾桶类型和状态检测结果
    trash_bins = []   # 垃圾桶类型 (Blue, Red, Grey, Green)
    statuses = []     # 垃圾桶状态 (Open, Close)
    
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            class_name = CLASS_MAP[cls_id]
            conf = float(box.conf)
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            
            # 计算中心点和尺寸
            center = calculate_center(bbox)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            # 分类存储
            if class_name in ["Blue", "Red", "Grey", "Green"]:
                trash_bins.append({
                    "type": class_name,
                    "bbox": bbox,
                    "conf": conf,
                    "center": center,
                    "width": width,
                    "height": height
                })
            elif class_name in ["Open", "Close"]:
                statuses.append({
                    "type": class_name,
                    "bbox": bbox,
                    "conf": conf,
                    "center": center
                })
    
    # 匹配垃圾桶和状态
    matched_bins = match_trash_bins(trash_bins, statuses)
    
    # 输出结果
    print(f"图像: {os.path.basename(image_path)}")
    if matched_bins:
        print(f"检测到 {len(matched_bins)} 个垃圾桶:")
        for bin in matched_bins:
            print(f"- {CHINESE_NAMES[bin['type']]}: {CHINESE_NAMES[bin['status']]}")
    else:
        print("未检测到匹配的垃圾桶")
    
    # 统计每种垃圾桶的数量
    bin_counts = defaultdict(int)
    for bin in matched_bins:
        bin_counts[bin["type"]] += 1
    
    for bin_type, count in bin_counts.items():
        print(f"  {CHINESE_NAMES[bin_type]}: {count}个")
    
    print("-" * 50)
    
    # 绘制检测结果
    annotated_image = draw_detections(image, matched_bins)
    
    # 保存结果图像
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, annotated_image)
    
    return {
        "total": len(matched_bins),
        "bins": matched_bins,
        "counts": bin_counts
    }

def draw_detections(image, matched_bins):
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
    
    # 绘制所有匹配的垃圾桶
    for bin in matched_bins:
        bin_type = bin["type"]
        status = bin["status"]
        bin_bbox = bin["bbox"]
        status_bbox = bin["status_bbox"]
        
        # 绘制垃圾桶类型框
        bin_color = TRASH_COLOR_MAP.get(bin_type, (255, 255, 255))
        draw.rectangle([(bin_bbox[0], bin_bbox[1]), (bin_bbox[2], bin_bbox[3])], 
                      outline=bin_color, width=3)
        
        # 绘制垃圾桶状态框
        status_color = STATUS_COLOR_MAP.get(status, (255, 255, 255))
        draw.rectangle([(status_bbox[0], status_bbox[1]), (status_bbox[2], status_bbox[3])], 
                      outline=status_color, width=2)
        
        # 绘制连接线
        bin_center = calculate_center(bin_bbox)
        status_center = calculate_center(status_bbox)
        draw.line([bin_center, status_center], fill=(255, 255, 0), width=2)
        
        # 绘制垃圾桶类型标签
        bin_label = f"{CHINESE_NAMES[bin_type]}"
        bin_text_size = font.getbbox(bin_label)
        draw.rectangle([(bin_bbox[0], bin_bbox[1] - bin_text_size[3] + bin_text_size[1] - 5), 
                       (bin_bbox[0] + bin_text_size[2] - bin_text_size[0] + 10, bin_bbox[1] + 5)], 
                      fill=bin_color)
        draw.text((bin_bbox[0] + 5, bin_bbox[1] - bin_text_size[3] + bin_text_size[1] - 5), 
                 bin_label, fill=(255, 255, 255), font=font)
        
        # 绘制状态标签
        status_label = f"状态: {CHINESE_NAMES[status]}"
        status_text_size = font.getbbox(status_label)
        draw.rectangle([(status_bbox[0], status_bbox[1] - status_text_size[3] + status_text_size[1] - 5), 
                       (status_bbox[0] + status_text_size[2] - status_text_size[0] + 10, status_bbox[1] + 5)], 
                      fill=status_color)
        draw.text((status_bbox[0] + 5, status_bbox[1] - status_text_size[3] + status_text_size[1] - 5), 
                 status_label, fill=(0, 0, 0), font=font)
    
    # 在图像顶部添加汇总信息
    if matched_bins:
        summary = f"检测到 {len(matched_bins)} 个垃圾桶"
        bin_types = defaultdict(int)
        for bin in matched_bins:
            bin_types[bin["type"]] += 1
        
        for bin_type, count in bin_types.items():
            summary += f" | {CHINESE_NAMES[bin_type]}: {count}"
        
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
    
    total_bins = 0
    bin_type_counts = defaultdict(int)
    status_counts = defaultdict(int)
    
    for result in all_results:
        total_bins += result["total"]
        for bin in result["bins"]:
            bin_type_counts[bin["type"]] += 1
            status_counts[bin["status"]] += 1
    
    print(f"测试图像总数: {len(all_results)}")
    print(f"检测到垃圾桶总数: {total_bins}")
    
    if total_bins > 0:
        print("\n垃圾桶类型分布:")
        for bin_type, count in bin_type_counts.items():
            print(f"  {CHINESE_NAMES[bin_type]}: {count}个 ({count/total_bins*100:.1f}%)")
        
        print("\n垃圾桶状态分布:")
        for status, count in status_counts.items():
            print(f"  {CHINESE_NAMES[status]}: {count}个 ({count/total_bins*100:.1f}%)")
    else:
        print("\n未检测到任何垃圾桶")
    
    print("=" * 80)
    print(f"所有标记后的图像已保存到: {output_dir}")

if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = "D:/All_save/Code_save/yolov8_trainmodel/runs/trashbar/train/weights/last.pt"
    TEST_DIR = "D:\All_save\Code_save\yolov8_trainmodel\\allpic\\trashbar\images"  # 测试图像目录
    OUTPUT_DIR = "D:\All_save\Code_save\yolov8_trainmodel\\val_save\\trashbar"  # 结果保存目录
    
    # 运行测试
    test_model(MODEL_PATH, TEST_DIR, OUTPUT_DIR)