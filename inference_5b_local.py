import os
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

output_dir = "/dev_vepfs/fly/resoning/DiffSynth-Studio/results/5b"
os.makedirs(output_dir, exist_ok=True)

prompts = [
    "A player rides their horse, preparing to strike the ball, their mallet poised.",
    "A soccer player runs, plants their foot, and drop kicks a soccer ball high into the air, the ball arcing visibly.",
    "A gymnast drops from the parallel bars and lands safely on the mat below.",
    "A gymnast performs a transition from a front support to a back support on the pommel horse.",
    "A player dunks the basketball, the basketball soaring upward before slamming through the net.",
    "A person wearing a helmet performs a handspring over a platform.",
    "A person plays squash on an indoor court.",
    "A gymnast stands on a balance beam, then performs a forward roll, landing smoothly.",
    "A wooden pencil is carefully dipped into a glass of crystal-clear water, showing the intriguing visual shifts and reflections caused by the interaction between the pencil and the liquid.",
    "A small burning ball of paper was thrown into a pile of dry paper.",
    "A golfer addresses the ball, takes a backswing, and hits the ball, sending it arcing high into the air.",
    "A metal pipe smashes a pumpkin, causing its insides to spill out.",
    "A baseball bat smashes a glass bottle, sending shards flying in all directions.",
    "A volleyball player takes a hard swing, the ball contacting the palm and fingers, followed by a rapid downward motion sending the ball over the net.",
    "A weightlifter completes a snatch with a 25kg barbell, holding it momentarily overhead.",
]

# Text-to-video
for i, prompt in enumerate(prompts):
    print(f"[{i+1}/{len(prompts)}] Generating: {prompt[:60]}...")
    video = pipe(
        prompt=prompt,
        seed=0, tiled=True,
        height=704, width=1248,
        num_frames=121,
    )
    save_path = os.path.join(output_dir, f"video_5b_{i+1}.mp4")
    save_video(video, save_path, fps=15, quality=5)
    print(f"Saved {save_path}")
