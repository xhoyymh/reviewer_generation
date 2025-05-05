import os
import json
import whisper
import tempfile
import subprocess  # 导入 subprocess 模块
from pathlib import Path
from tqdm import tqdm  # 导入 tqdm 模块以实现进度条

# --- 配置 (保持不变) ---
VIDEO_FOLDER_PATH = "C:\\Users\\Orange\\Desktop\\review_generation\\video"
OUTPUT_JSON_PATH = "C:\\Users\\Orange\\Desktop\\review_generation\\data\\sound_to_text.json"
SUPPORTED_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv')
WHISPER_MODEL_NAME = "base"
# --- 配置结束 ---

def extract_audio_with_ffmpeg(video_path, output_audio_path):
    """使用 ffmpeg 提取音频"""
    command = [
        'ffmpeg',
        '-i', str(video_path),
        '-vn',
        '-acodec', 'mp3',  # 或者 'aac', 或者 'copy' (如果格式兼容)
        '-y',
        str(output_audio_path)
    ]
    try:
        print(f"  执行 ffmpeg 命令: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        print("  ffmpeg 命令执行成功。")
        return True
    except FileNotFoundError:
        print("错误：找不到 ffmpeg 命令。请确保 ffmpeg 已安装并已添加到系统 PATH。")
        return False
    except subprocess.CalledProcessError as e:
        print(f"错误：ffmpeg 执行失败。返回码: {e.returncode}")
        print(f"  ffmpeg 错误输出:\n{e.stderr}")
        return False
    except Exception as e:
        print(f"执行 ffmpeg 时发生未知错误: {e}")
        return False

def extract_and_transcribe(video_path, whisper_model):
    """从视频文件中提取音频并进行转录"""
    print(f"--- 开始处理: {video_path.name} ---")
    audio_path = None
    temp_audio_file = None
    try:
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        audio_path = Path(temp_audio_file.name)
        temp_audio_file.close()

        print(f"  [1/3] 正在使用 ffmpeg 提取音频到: {audio_path}...")
        if not extract_audio_with_ffmpeg(video_path, audio_path):
            raise RuntimeError("ffmpeg 音频提取失败")

        if not audio_path.exists() or audio_path.stat().st_size == 0:
            print(f"  警告: 提取的音频文件不存在或为空: {audio_path}")
            raise RuntimeError("提取的音频文件为空或不存在")

        print(f"  [2/3] 正在转录音频 (使用模型: {WHISPER_MODEL_NAME})...")
        result = whisper_model.transcribe(str(audio_path), fp16=False)
        transcribed_text = result["text"]
        print(f"  转录完成。")

        return transcribed_text

    except Exception as e:
        print(f"  处理视频 {video_path.name} 时出错: {e}")
        return f"错误: {e}"

    finally:
        if audio_path and audio_path.exists():
            try:
                os.remove(audio_path)
                print(f"  [3/3] 临时音频文件已删除: {audio_path}")
            except OSError as e:
                print(f"  无法删除临时文件 {audio_path}: {e}")
        elif temp_audio_file:
            try:
                os.remove(temp_audio_file.name)
                print(f"  [3/3] 临时音频文件已删除: {temp_audio_file.name}")
            except OSError as e:
                print(f"  无法删除临时文件 {temp_audio_file.name}: {e}")

        print(f"--- 完成处理: {video_path.name} ---\n")

def process_videos_in_folder(folder_path_str, output_json_str):
    """处理指定文件夹中的所有视频文件，并将结果保存到 JSON 文件"""
    video_folder = Path(folder_path_str)
    output_json_file = Path(output_json_str)

    if not video_folder.is_dir():
        print(f"错误：指定的视频文件夹路径不存在或不是一个目录: {folder_path_str}")
        return

    output_json_file.parent.mkdir(parents=True, exist_ok=True)

    results = []
    total_videos = 0
    all_files = list(video_folder.iterdir())

    for item in all_files:
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            total_videos += 1

    print(f"在文件夹 {video_folder} 中找到 {total_videos} 个支持的视频文件。")
    print(f"将使用 Whisper 模型: {WHISPER_MODEL_NAME}")
    print(f"结果将保存到: {output_json_file}")
    print("-" * 30)

    try:
        print("正在加载 Whisper 模型...")
        model = whisper.load_model(WHISPER_MODEL_NAME)
        print("Whisper 模型加载成功。")
    except Exception as e:
        print(f"错误：无法加载 Whisper 模型 '{WHISPER_MODEL_NAME}'. 请确保模型名称正确且已安装。错误: {e}")
        return

    with tqdm(total=total_videos, desc="处理视频进度", unit="个") as pbar:
        for item in all_files:
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                print(f"\n>>> 正在处理视频: {item.name}")
                video_full_path = str(item.resolve())
                transcription = extract_and_transcribe(item, model)
                results.append({
                    "video_path": video_full_path,
                    "sound_to_text": transcription if transcription is not None else "转录失败或无音频"
                })
                pbar.update(1)

    print(f"\n>>> 所有视频处理完毕，正在将结果写入 JSON 文件: {output_json_file}")
    try:
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print("JSON 文件写入成功！")
    except IOError as e:
        print(f"错误：无法写入 JSON 文件 {output_json_file}: {e}")

if __name__ == "__main__":
    if "你的视频文件夹路径" in VIDEO_FOLDER_PATH or "输出的json文件路径/文件名.json" in OUTPUT_JSON_PATH:
        print("警告：请修改 'VIDEO_FOLDER_PATH' 和 'OUTPUT_JSON_PATH' 为实际路径！")
    else:
        process_videos_in_folder(VIDEO_FOLDER_PATH, OUTPUT_JSON_PATH)
