import os
from datasets import load_dataset

# =================配置区域=================
DATASET_NAME = "videophysics/videophy2_test"
LOCAL_SAVE_PATH = "./local_dataset"  # 数据保存的本地文件夹名称
# =========================================

print(f"📥 正在连接 HuggingFace 下载 {DATASET_NAME} ...")

try:
    # 1. 下载数据集
    # 注意：如果不确定 split 是 'train' 还是 'test'，可以先不指定 split，下载所有
    ds = load_dataset(DATASET_NAME)
    
    print("✅ 下载完成，正在写入本地磁盘...")
    
    # 2. 保存到本地 (Save to Disk)
    ds.save_to_disk(LOCAL_SAVE_PATH)
    
    print(f"🎉 数据集已成功保存在: {os.path.abspath(LOCAL_SAVE_PATH)}")
    print(f"包含的 Split: {list(ds.keys())}")

except Exception as e:
    print(f"❌ 下载失败: {e}")