import json
import re
import requests
from tqdm import tqdm

# 输入文件路径（存储图片描述）
input_file = "C:/Users/Orange/Desktop/review_generation/conversation/llava_first50video.json"
additional_file = "C:/Users/Orange/Desktop/review_generation/conversation/sound_to_text.json"  # 新增文件路径
output_file = "C:/Users/Orange/Desktop/review_generation/data/llava_first50video_reviews.json"

# 设置 Ollama 模型的参数
OLLAMA_MODEL = "llama3"  # 选择模型（可以换成 "mistral"、"llama2" 等）
TEMPERATURE = 0.7  # 控制输出的随机性
TOP_P = 0.9  # 核采样，影响生成内容的多样性
MAX_TOKENS = 256  # 限制生成内容的最大长度
REPETITION_PENALTY = 1.2  # 设定重复惩罚系数，减少重复
PRESENCE_PENALTY = 0.6  # 增加新词汇的出现概率

# 读取 JSON 文件
with open(input_file, "r", encoding="utf-8") as f:
    qa_records = json.load(f)

with open(additional_file, "r", encoding="utf-8") as f:
    additional_records = json.load(f)

# 组织数据，按视频编号分组
video_descriptions = {}

for record in qa_records:
    image_name = record["image_name"]
    description = record["response"]

    # 提取视频编号（如 10_1.jpg 提取出 10）
    match = re.match(r"(\d+)_\d+\.jpg", image_name)
    if match:
        video_id = match.group(1)
        if video_id not in video_descriptions:
            video_descriptions[video_id] = []
        video_descriptions[video_id].append(description)

# 合并声音转文本数据
for record in additional_records:
    video_path = record["video_path"]
    sound_to_text = record["sound_to_text"]

    # 提取视频编号（如 18.mp4 提取出 18）
    match = re.match(r"(\d+)\.mp4", video_path)
    if match:
        video_id = match.group(1)
        if video_id not in video_descriptions:
            video_descriptions[video_id] = []
        video_descriptions[video_id].append(f"Sound transcription: {sound_to_text}")

# 调用 Ollama 生成评论（使用主进度条）
video_reviews = []
total_videos = len(video_descriptions)

with tqdm(total=total_videos, desc="正在生成视频评论") as pbar:
    for video_id, descriptions in video_descriptions.items():
        combined_description = " ".join(descriptions)  # 合并同一视频所有图片的描述和声音转文本内容

        # 构建 Ollama 请求数据
        data = {
            "model": OLLAMA_MODEL,
            "prompt": f"Based on the following descriptions and sound transcription, generate a concise and insightful review for the video:\n{combined_description}, constraint: The response should be within 15 words. Emojis can be added as appropriate. The comments generated should be humorous enough.",
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "repetition_penalty": REPETITION_PENALTY,
            "presence_penalty": PRESENCE_PENALTY
        }

        # 发送请求到 Ollama 本地服务器
        response = requests.post("http://localhost:11434/api/generate", json=data, stream=True)

        response_text = ""

        # 逐行解析 Ollama 返回的数据（防止 JSONDecodeError）
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    response_text += chunk.get("response", "")
                except json.JSONDecodeError as e:
                    print(f"JSON 解析错误：{e}")

        # 保存视频评论
        video_reviews.append({
            "video_id": video_id,
            "review": response_text
        })

        pbar.update(1)  # 更新主进度条

# 写入 JSON 文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(video_reviews, f, ensure_ascii=False, indent=4)

print(f"\n视频评论已保存到 {output_file}")
