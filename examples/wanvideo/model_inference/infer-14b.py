import os
import torch
from datasets import load_from_disk
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video

# ================= 配置区域 =================
LOCAL_DATA_PATH = "/dev_vepfs/fly/resoning/DiffSynth-Studio/local_dataset"
OUTPUT_DIR = "test_output_14b"
MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B"
TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
# ===========================================

# 1. 准备环境
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. 加载数据
print(f"📂 正在从本地加载数据: {LOCAL_DATA_PATH} ...")
if not os.path.exists(LOCAL_DATA_PATH):
    raise FileNotFoundError(f"找不到数据文件夹: {LOCAL_DATA_PATH}")

dataset_dict = load_from_disk(LOCAL_DATA_PATH)
ds = dataset_dict.get('test') or dataset_dict.get('train') or dataset_dict[list(dataset_dict.keys())[0]]
subset = ds.select(range(50))
print(f"✅ 数据准备就绪，共 {len(subset)} 条任务")

# 3. 初始化模型
print(f"🚀 正在初始化 Wan 模型，model_id: {MODEL_ID}")

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id=MODEL_ID, origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
            offload_dtype=torch.bfloat16, offload_device="cpu",
            onload_dtype=torch.bfloat16, onload_device="cpu",
            preparing_dtype=torch.bfloat16, preparing_device="cuda",
            computation_dtype=torch.bfloat16, computation_device="cuda",
        ),
        ModelConfig(
            model_id=MODEL_ID, origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
            offload_dtype=torch.bfloat16, offload_device="cpu",
            onload_dtype=torch.bfloat16, onload_device="cpu",
            preparing_dtype=torch.bfloat16, preparing_device="cuda",
            computation_dtype=torch.bfloat16, computation_device="cuda",
        ),
        ModelConfig(
            model_id=MODEL_ID, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            offload_dtype=torch.bfloat16, offload_device="cpu",
            onload_dtype=torch.bfloat16, onload_device="cpu",
            preparing_dtype=torch.bfloat16, preparing_device="cuda",
            computation_dtype=torch.bfloat16, computation_device="cuda",
        ),
        ModelConfig(
            model_id=MODEL_ID, origin_file_pattern="Wan2.1_VAE.pth",
            offload_dtype=torch.bfloat16, offload_device="cpu",
            onload_dtype=torch.bfloat16, onload_device="cpu",
            preparing_dtype=torch.bfloat16, preparing_device="cuda",
            computation_dtype=torch.bfloat16, computation_device="cuda",
        ),
    ],
    tokenizer_config=ModelConfig(model_id=TOKENIZER_MODEL_ID, origin_file_pattern="google/umt5-xxl/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
)

# 4. 循环生成
print(f"🎬 开始处理前 {len(subset)} 条数据...")

for i, item in enumerate(subset):
    caption = item.get('caption') or item.get('text') or item.get('prompt')
    if not caption:
        continue

    file_prefix = f"{i:03d}"
    video_filename = os.path.join(OUTPUT_DIR, f"{file_prefix}.mp4")
    txt_filename = os.path.join(OUTPUT_DIR, f"{file_prefix}.txt")

    print(f"\n[{i+1}/{len(subset)}] 生成中: {caption[:40]}...")

    try:
        video = pipe(
            prompt=caption,
            seed=42,
            tiled=True
        )
        save_video(video, video_filename, fps=15, quality=5)

        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(caption)
        print(f"   ✅ 保存完毕 -> {video_filename}")

    except Exception as e:
        print(f"   ❌ 生成出错: {e}")
        with open(os.path.join(OUTPUT_DIR, "errors.log"), "a", encoding="utf-8") as f:
            f.write(f"Index {i}: {e}\n")

print("\n🏁 全部任务完成！")
