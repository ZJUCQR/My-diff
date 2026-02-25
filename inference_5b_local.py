import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

base_path = "/dev_vepfs/fly/resoning/DiffSynth-Studio/5B_Wan"

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            path=f"{base_path}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors",
            offload_dtype=torch.bfloat16, offload_device="cpu",
            onload_dtype=torch.bfloat16, onload_device="cpu",
            preparing_dtype=torch.bfloat16, preparing_device="cuda",
            computation_dtype=torch.bfloat16, computation_device="cuda",
        ),
        ModelConfig(
            path=[
                f"{base_path}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
                f"{base_path}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
                f"{base_path}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors",
            ],
            offload_dtype=torch.bfloat16, offload_device="cpu",
            onload_dtype=torch.bfloat16, onload_device="cpu",
            preparing_dtype=torch.bfloat16, preparing_device="cuda",
            computation_dtype=torch.bfloat16, computation_device="cuda",
        ),
        ModelConfig(
            path=f"{base_path}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors",
            offload_dtype=torch.bfloat16, offload_device="cpu",
            onload_dtype=torch.bfloat16, onload_device="cpu",
            preparing_dtype=torch.bfloat16, preparing_device="cuda",
            computation_dtype=torch.bfloat16, computation_device="cuda",
        ),
    ],
    tokenizer_config=ModelConfig(path=f"{base_path}/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
)

# Text-to-video
video = pipe(
    prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
    seed=0, tiled=True,
    height=704, width=1248,
    num_frames=121,
)
save_video(video, "video_1_Wan2.2-TI2V-5B.mp4", fps=15, quality=5)
