import yt_dlp
import json

# 读取 JSON 文件
json_path = 'C:/Users/Orange/Desktop/review_generation/data/50hot_review.json'  # 指定 JSON 文件路径
with open(json_path, 'r', encoding='utf-8') as f:
    video_data = json.load(f)

# 提取 URL 列表和对应 ID
video_urls = [entry["video_url"] for entry in video_data]
video_ids = [str(entry["id"]) for entry in video_data]  # 转换 ID 为字符串

# 配置下载参数，包括指定存储路径
ydl_opts = {
    'format': 'best',
    'outtmpl': 'C:/Users/Orange/Desktop/review_generation/video/%(autonumber)d.mp4',  # 以 JSON 中的 ID 命名文件
}

# 执行下载
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for idx, url in enumerate(video_urls):
        ydl.download([url])
        print(f"视频 {video_ids[idx]} 下载完成")
