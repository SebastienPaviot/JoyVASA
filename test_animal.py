import os
import tyro
import subprocess
import os.path as osp
import platform
from src.utils.helper import load_description
from src.gradio_pipeline import GradioPipeline, GradioPipelineAnimal
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig

if platform.system() == "Windows":
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath


def partial_fields(target_class, kwargs):
    """Initialize class with specific arguments."""
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    """Check if FFmpeg is installed."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


# Ensure FFmpeg is available
ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
if osp.exists(ffmpeg_dir):
    os.environ["PATH"] += (os.pathsep + ffmpeg_dir)
if not fast_check_ffmpeg():
    raise ImportError(
        "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
    )

# Set tyro theme
tyro.extras.set_accent_color("bright_cyan")

# Parse arguments
args = tyro.cli(ArgumentConfig)

# Specify configurations for inference
inference_cfg = partial_fields(InferenceConfig, args.__dict__)
crop_cfg = partial_fields(CropConfig, args.__dict__)

# Initialize pipelines
gradio_pipeline_human = GradioPipeline(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg,
    args=args
)
gradio_pipeline_animal = GradioPipelineAnimal(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg,
    args=args
)


def execute_pipeline(input_image_path, input_audio_path, animation_mode="human",
                     flag_do_crop=True, cfg_scale=4.0, normalize_lip=True,
                     relative_motion=True, driving_multiplier=1.0, driving_option="expression-friendly",
                     scale=2.3, vx_ratio=0.0, vy_ratio=-0.125, stitching=False, remap_input=False):
    """
    Execute animation-to-video pipeline based on input parameters.

    Args:
        input_image_path (str): Path to the reference image.
        input_audio_path (str): Path to the input audio.
        animation_mode (str): Animation mode, either 'human' or 'animal'.
        flag_do_crop (bool): Whether to crop the input image.
        cfg_scale (float): Configuration scale.
        normalize_lip (bool): Whether to normalize lips in animation.
        relative_motion (bool): Use relative motion for animation.
        driving_multiplier (float): Multiplier for driving parameters.
        driving_option (str): Option for driving ('expression-friendly' or 'pose-friendly').
        scale (float): Crop scale for reference image.
        vx_ratio (float): Horizontal cropping adjustment.
        vy_ratio (float): Vertical cropping adjustment.
        stitching (bool): Enable or disable stitching.
        remap_input (bool): Enable or disable remapping input.

    Returns:
        str: Path to the generated output video.
    """
    # Select pipeline based on animation mode
    pipeline = gradio_pipeline_animal if animation_mode == "animal" else gradio_pipeline_human

    # Execute the chosen pipeline
    output_video_path = pipeline.execute_a2v(
        input_image_path=input_image_path,
        input_audio_path=input_audio_path,
        do_crop=flag_do_crop,
        cfg_scale=cfg_scale,
        normalize_lip=normalize_lip,
        relative_motion=relative_motion,
        driving_multiplier=driving_multiplier,
        driving_option=driving_option,
        scale=scale,
        vx_ratio=vx_ratio,
        vy_ratio=vy_ratio,
        stitching=stitching,
        remap_input=remap_input
    )
    return output_video_path


# Example usage
if __name__ == "__main__":
    # Define input paths and parameters for human animation
    # input_image_human = "assets/examples/imgs/joyvasa_001.png"
    # input_audio_human = "assets/examples/audios/joyvasa_001.wav"
    # animation_mode_human = "human"

    # # Execute human pipeline
    # output_video_human = execute_pipeline(
    #     input_image_path=input_image_human,
    #     input_audio_path=input_audio_human,
    #     animation_mode=animation_mode_human,
    #     flag_do_crop=True,
    #     cfg_scale=4.0,
    #     normalize_lip=True,
    #     relative_motion=True,
    #     driving_multiplier=1.0,
    #     driving_option="expression-friendly",
    #     scale=2.3,
    #     vx_ratio=0.0,
    #     vy_ratio=-0.125,
    #     stitching=False,
    #     remap_input=False
    # )
    # print(f"Human animation video saved at: {output_video_human}")

    # Define input paths and parameters for animal animation
    input_image_animal = "assets/examples/imgs/joyvasa_001.png"
    input_audio_animal = "assets/examples/audios/joyvasa_001.wav"
    animation_mode_animal = "animal"

    # Execute animal pipeline
    output_video_animal = execute_pipeline(
        input_image_path=input_image_animal,
        input_audio_path=input_audio_animal,
        animation_mode=animation_mode_animal,
        flag_do_crop=True,
        cfg_scale=4.0,
        normalize_lip=True,
        relative_motion=True,
        driving_multiplier=1.0,
        driving_option="pose-friendly",
        scale=2.3,
        vx_ratio=0.0,
        vy_ratio=-0.125,
        stitching=False,
        remap_input=False
    )
    print(f"Animal animation video saved at: {output_video_animal}")
