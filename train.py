import os
import random
import shutil
import yaml
import argparse
from datetime import datetime
from ultralytics import YOLO

def prepare_dataset(dataset_root='datasets', output_dir=None, val_ratio=0.2):
    """
    准备YOLOv8数据集
    :param dataset_root: 数据集根目录
    :param output_dir: 输出目录
    :param val_ratio: 验证集比例
    :return: data.yaml路径
    """
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置输出目录
    if not output_dir:
        output_dir = f"dataset_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建YOLO格式目录结构
    yolo_dirs = {
        'images/train': os.path.join(output_dir, 'images', 'train'),
        'images/val': os.path.join(output_dir, 'images', 'val'),
        'labels/train': os.path.join(output_dir, 'labels', 'train'),
        'labels/val': os.path.join(output_dir, 'labels', 'val')
    }
    
    for path in yolo_dirs.values():
        os.makedirs(path, exist_ok=True)
    
    # 获取所有图片文件
    image_dir = os.path.join(dataset_root, 'images')
    all_images = [f for f in os.listdir(image_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 获取对应的标签文件
    label_dir = os.path.join(dataset_root, 'labels')
    valid_pairs = []
    for img_name in all_images:
        base_name = os.path.splitext(img_name)[0]
        label_file = os.path.join(label_dir, f"{base_name}.txt")
        
        if os.path.exists(label_file):
            valid_pairs.append((img_name, base_name))
    
    # 读取类别文件
    classes_path = os.path.join(label_dir, 'classes.txt')
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        print(f"从classes.txt加载了{len(class_names)}个类别")
    else:
        # 如果没有classes.txt，尝试从标签中推断
        class_names = set()
        for _, base_name in valid_pairs:
            label_path = os.path.join(label_dir, f"{base_name}.txt")
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if parts:
                            class_names.add(parts[0])
            except:
                continue
        
        class_names = sorted(list(class_names))
        print(f"从标签文件推断出{len(class_names)}个类别")
    
    # 随机打乱并划分数据集
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * (1 - val_ratio))
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    
    # 复制文件到训练集
    for img_name, base_name in train_pairs:
        # 图片
        src_img = os.path.join(image_dir, img_name)
        dst_img = os.path.join(yolo_dirs['images/train'], img_name)
        shutil.copy2(src_img, dst_img)
        
        # 标签
        src_label = os.path.join(label_dir, f"{base_name}.txt")
        dst_label = os.path.join(yolo_dirs['labels/train'], f"{base_name}.txt")
        shutil.copy2(src_label, dst_label)
    
    # 复制文件到验证集
    for img_name, base_name in val_pairs:
        # 图片
        src_img = os.path.join(image_dir, img_name)
        dst_img = os.path.join(yolo_dirs['images/val'], img_name)
        shutil.copy2(src_img, dst_img)
        
        # 标签
        src_label = os.path.join(label_dir, f"{base_name}.txt")
        dst_label = os.path.join(yolo_dirs['labels/val'], f"{base_name}.txt")
        shutil.copy2(src_label, dst_label)
    
    # 创建类别映射
    class_mapping = {i: name for i, name in enumerate(class_names)}
    
    # 创建data.yaml
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': class_mapping,
        'nc': len(class_names)
    }
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    print(f"数据集准备完成! 共 {len(train_pairs)} 训练样本, {len(val_pairs)} 验证样本")
    print(f"检测到 {len(class_names)} 个类别: {', '.join(class_names)}")
    
    return yaml_path

def train_model(data_yaml, config):
    """训练YOLOv8模型"""
    # 创建带时间戳的结果目录 - 确保在runs目录下
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('runs', f'train_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # 初始化模型
    if config['pretrained']:
        print(f"使用预训练模型: {config['model']}")
        model = YOLO(config['model'] + '.pt')  # 迁移学习
    else:
        print(f"从零开始训练模型: {config['model']}")
        model = YOLO(config['model'] + '.yaml')  # 从零开始训练
    
    # 训练参数 - 完全使用CPU训练
    train_args = {
        'data': data_yaml,
        'epochs': config['epochs'],
        'batch': config['batch'],
        'imgsz': config['imgsz'],
        'project': results_dir,  # 所有结果保存在此目录
        'name': '',  # 空名称，所有结果直接保存在project目录
        'exist_ok': False,
        'save_period': config['save_period'],
        'device': 'cpu',  # 强制使用CPU
        'workers': config['workers'],
        'lr0': config['lr0'],
        'lrf': config['lrf'],
        'momentum': config['momentum'],
        'weight_decay': config['weight_decay'],
        'warmup_epochs': config['warmup_epochs'],
        'optimizer': config['optimizer'],
        'seed': config['seed'],
        'patience': config['patience']
    }
    
    # 开始训练
    print("开始训练...")
    results = model.train(**train_args)
    
    # 重命名最佳模型
    weights_dir = os.path.join(results_dir, 'weights')
    best_model_path = os.path.join(weights_dir, 'best.pt')
    final_model_path = os.path.join(results_dir, f'model_final_{timestamp}.pt')  # 直接放在结果目录
    
    if os.path.exists(best_model_path):
        os.rename(best_model_path, final_model_path)
        print(f"最终模型保存为: {final_model_path}")
    else:
        print("警告: 未找到最佳模型文件")
    
    return results_dir

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv8训练脚本')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='yolov8n', 
                       choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                       help='模型架构')
    parser.add_argument('--pretrained', action='store_true',
                       help='使用预训练模型')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,  # 减少默认epochs
                       help='训练轮数')
    parser.add_argument('--batch', type=int, default=4,  # 减小默认batch_size
                       help='批次大小')
    parser.add_argument('--imgsz', type=int, default=320,  # 减小默认图像尺寸
                       help='输入图像尺寸')
    parser.add_argument('--save-period', type=int, default=10, 
                       help='每多少轮保存一次模型')
    
    # 优化参数
    parser.add_argument('--lr0', type=float, default=0.01, 
                       help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.01, 
                       help='最终学习率 = lr0 * lrf')
    parser.add_argument('--momentum', type=float, default=0.937, 
                       help='动量')
    parser.add_argument('--weight-decay', type=float, default=0.0005, 
                       help='权重衰减')
    parser.add_argument('--warmup-epochs', type=float, default=3.0, 
                       help='预热轮数')
    parser.add_argument('--optimizer', type=str, default='auto', 
                       choices=['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'auto'],
                       help='优化器')
    
    # 数据参数
    parser.add_argument('--val-ratio', type=float, default=0.2, 
                       help='验证集比例')
    
    # 系统参数
    parser.add_argument('--workers', type=int, default=2,  # 减少默认工作线程数
                       help='数据加载工作线程数')
    parser.add_argument('--seed', type=int, default=42, 
                       help='随机种子')
    parser.add_argument('--patience', type=int, default=30, 
                       help='早停耐心值')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建配置字典
    config = {
        # 模型参数
        'model': args.model,
        'pretrained': args.pretrained,
        
        # 训练参数
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'save_period': args.save_period,
        
        # 优化参数
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'optimizer': args.optimizer,
        
        # 系统参数
        'workers': args.workers,
        'seed': args.seed,
        'patience': args.patience
    }
    
    # 打印配置
    print("\n训练配置:")
    for key, value in config.items():
        print(f"- {key}: {value}")
    
    # 准备数据集
    data_yaml = prepare_dataset(
        dataset_root='datasets',
        val_ratio=args.val_ratio
    )
    
    # 训练模型
    results_dir = train_model(data_yaml, config)
    
    print(f"\n训练完成! 结果保存在: {results_dir}")
    print("训练结果包含:")
    print(f"- 训练日志: {os.path.join(results_dir, 'results.csv')}")
    print(f"- 可视化结果: {os.path.join(results_dir, 'results.png')}")
    print(f"- 模型权重: {os.path.join(results_dir, 'weights')}")
    print(f"- 配置文件: {os.path.join(results_dir, 'args.yaml')}")

if __name__ == '__main__':
    main()