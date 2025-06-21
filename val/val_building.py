import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

# 类别映射
CLASS_MAP = {
    0: "5F",
    1: "4F",
    2: "3F",
    3: "2F",
    4: "1F",
    5: "fire",
    6: "shop",
    7: "market"
}

# 中文名称映射
CHINESE_NAMES = {
    "5F": "五楼",
    "4F": "四楼",
    "3F": "三楼",
    "2F": "二楼",
    "1F": "一楼",
    "fire": "火源",
    "shop": "美丽商场",
    "market": "电子超市"
}

# 颜色映射
COLOR_MAP = {
    "5F": (0, 165, 255),    # 橙色
    "4F": (0, 255, 255),    # 黄色
    "3F": (0, 255, 0),      # 绿色
    "2F": (255, 255, 0),    # 青色
    "1F": (255, 0, 255),    # 紫色
    "fire": (0, 0, 255),    # 红色
    "shop": (0, 191, 255),  # 深蓝色
    "market": (128, 0, 128) # 紫色
}

def load_model(model_path):
    """加载训练好的模型"""
    model = YOLO(model_path)
    return model

def get_building_type(results):
    """确定建筑物类型（shop或market）"""
    shop_conf = 0
    market_conf = 0
    
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            
            if CLASS_MAP[cls_id] == "shop" and conf > shop_conf:
                shop_conf = conf
            elif CLASS_MAP[cls_id] == "market" and conf > market_conf:
                market_conf = conf
    
    # 返回置信度更高的类型
    if shop_conf > market_conf and shop_conf > 0.3:
        return "shop", shop_conf
    elif market_conf > 0.3:
        return "market", market_conf
    return "unknown", 0.0

def get_floor_boxes(results):
    """获取所有楼层检测框"""
    floor_boxes = []
    
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            class_name = CLASS_MAP[cls_id]
            
            if class_name.endswith("F"):  # 楼层标签
                conf = float(box.conf)
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                floor_boxes.append({
                    "class": class_name,
                    "bbox": bbox,
                    "conf": conf,
                    "center_y": (bbox[1] + bbox[3]) // 2  # 计算中心点y坐标
                })
    
    # 按y坐标排序（从低到高：1F在最下面，5F在最上面）
    floor_boxes.sort(key=lambda x: x["center_y"], reverse=True)
    return floor_boxes

def get_fire_boxes(results):
    """获取所有火源检测框"""
    fire_boxes = []
    
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            if CLASS_MAP[cls_id] == "fire":
                conf = float(box.conf)
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                fire_boxes.append({
                    "bbox": bbox,
                    "conf": conf,
                    "bottom_y": bbox[3]  # 火源框的底部y坐标
                })
    
    return fire_boxes

def assign_fires_to_floors(fire_boxes, floor_boxes):
    """将火源分配到最近的楼层"""
    fire_assignments = defaultdict(list)
    
    for fire in fire_boxes:
        fire_y = fire["bottom_y"]
        closest_floor = None
        min_distance = float('inf')
        
        # 找到最近的楼层
        for floor in floor_boxes:
            distance = abs(fire_y - floor["center_y"])
            if distance < min_distance:
                min_distance = distance
                closest_floor = floor["class"]
        
        if closest_floor:
            fire_assignments[closest_floor].append(fire)
    
    return fire_assignments

def draw_detections(image, results, floor_boxes, fire_assignments):
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
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            class_name = CLASS_MAP[cls_id]
            conf = float(box.conf)
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            
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
    building_type = None
    fire_count = 0
    fire_locations = []
    
    # 获取建筑物类型
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            class_name = CLASS_MAP[cls_id]
            if class_name in ["shop", "market"]:
                building_type = CHINESE_NAMES[class_name]
                break
        if building_type:
            break
    
    if not building_type:
        building_type = "未知建筑"
    
    # 统计火源信息
    for floor, fires in fire_assignments.items():
        fire_count += len(fires)
        fire_locations.append(f"{CHINESE_NAMES[floor]}有{len(fires)}个火源")
    
    summary = f"建筑类型: {building_type}"
    if fire_count > 0:
        summary += f" | 火源总数: {fire_count} | {' | '.join(fire_locations)}"
    else:
        summary += " | 无火源"
    
    # 绘制汇总信息
    text_size = font.getbbox(summary)
    draw.rectangle([(10, 10), (10 + text_size[2] - text_size[0] + 20, 10 + text_size[3] - text_size[1] + 20)], 
                  fill=(0, 0, 0, 180))
    draw.text((20, 20), summary, fill=(255, 255, 255), font=font)
    
    # 将PIL图像转换回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def analyze_image(model, image_path, output_dir):
    """分析单张图像并保存结果"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    # 使用模型进行预测
    results = model(image, conf=0.25)  # 置信度阈值设为0.25
    
    # 获取建筑物类型
    building_type, conf = get_building_type(results)
    
    # 获取楼层框
    floor_boxes = get_floor_boxes(results)
    
    # 获取火源框
    fire_boxes = get_fire_boxes(results)
    
    # 将火源分配到楼层
    fire_assignments = assign_fires_to_floors(fire_boxes, floor_boxes)
    
    # 输出结果
    print(f"图像: {os.path.basename(image_path)}")
    print(f"建筑类型: {CHINESE_NAMES.get(building_type, '未知建筑')}")
    print(f"检测到楼层: {', '.join([CHINESE_NAMES[box['class']] for box in floor_boxes])}")
    
    if fire_boxes:
        print(f"火源总数: {len(fire_boxes)}")
        for floor, fires in fire_assignments.items():
            print(f"{CHINESE_NAMES[floor]}: {len(fires)}个火源")
    else:
        print("无火源")
    print("-" * 50)
    
    # 绘制检测结果
    annotated_image = draw_detections(image, results, floor_boxes, fire_assignments)
    
    # 保存结果图像
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, annotated_image)
    
    return {
        "building": building_type,
        "floors": [box["class"] for box in floor_boxes],
        "fires": fire_assignments
    }

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
    
    building_counts = defaultdict(int)
    total_fires = 0
    floor_fires = defaultdict(int)
    
    for result in all_results:
        building_counts[result["building"]] += 1
        
        for floor, fires in result["fires"].items():
            floor_fires[floor] += len(fires)
            total_fires += len(fires)
    
    print(f"测试图像总数: {len(all_results)}")
    print(f"建筑类型分布:")
    for btype, count in building_counts.items():
        print(f"  {CHINESE_NAMES.get(btype, btype)}: {count}张图像")
    
    print(f"\n火源总数: {total_fires}")
    if total_fires > 0:
        print("火源楼层分布:")
        for floor, count in floor_fires.items():
            print(f"  {CHINESE_NAMES[floor]}: {count}个火源")
    
    print("=" * 80)
    print(f"所有标记后的图像已保存到: {output_dir}")

if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = "D:/All_save/Code_save/yolov8_trainmodel/runs/building/train/weights/last.pt"
    TEST_DIR = "D:\All_save\Code_save\yolov8_trainmodel\\allpic\\buildings\images"  # 测试图像目录
    OUTPUT_DIR = "D:\All_save\Code_save\yolov8_trainmodel\\val_save\\building"  # 结果保存目录
    
    # 运行测试
    test_model(MODEL_PATH, TEST_DIR, OUTPUT_DIR)