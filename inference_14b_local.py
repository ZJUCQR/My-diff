import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

base_path = "/dev_vepfs/fly/resoning/DiffSynth-Studio/14B_Wan"

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            path=[
                f"{base_path}/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-0000{i}-of-00006.safetensors"
                for i in range(1, 7)
            ],
            offload_dtype=torch.bfloat16, offload_device="cpu",
            onload_dtype=torch.bfloat16, onload_device="cpu",
            preparing_dtype=torch.bfloat16, preparing_device="cuda",
            computation_dtype=torch.bfloat16, computation_device="cuda",
        ),
        ModelConfig(
            path=[
                f"{base_path}/Wan-AI/Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model-0000{i}-of-00006.safetensors"
                for i in range(1, 7)
            ],
            offload_dtype=torch.bfloat16, offload_device="cpu",
            onload_dtype=torch.bfloat16, onload_device="cpu",
            preparing_dtype=torch.bfloat16, preparing_device="cuda",
            computation_dtype=torch.bfloat16, computation_device="cuda",
        ),
        ModelConfig(
            path=f"{base_path}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors",
            offload_dtype=torch.bfloat16, offload_device="cpu",
            onload_dtype=torch.bfloat16, onload_device="cpu",
            preparing_dtype=torch.bfloat16, preparing_device="cuda",
            computation_dtype=torch.bfloat16, computation_device="cuda",
        ),
        ModelConfig(
            path=f"{base_path}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors",
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
    prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    seed=0, tiled=True,
)
save_video(video, "video_2_Wan2.2-T2V-A14B.mp4", fps=15, quality=5)
