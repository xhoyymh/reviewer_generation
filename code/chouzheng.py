import cv2
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np
from tqdm import tqdm  # 进度条库

def process_videos(input_folder, output_root, similarity_threshold=0.7, frame_interval=1):
    # 确保输出主文件夹存在
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 获取所有视频文件
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mkv'))]

    # 遍历输入文件夹中的视频
    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        video_name = os.path.splitext(video_file)[0]  # 提取视频名称

        # 为当前视频创建专属文件夹
        output_folder = os.path.join(output_root, video_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
        frame_skip = int(fps * frame_interval)  # 计算需要跳过的帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数

        success, prev_frame = cap.read()
        saved_index = 1  # 图片编号从 1 开始

        # 进度条
        with tqdm(total=total_frames, desc=f"Processing {video_file}", unit="frame") as pbar:
            while success:
                # 跳过指定数量的帧
                for _ in range(frame_skip):
                    success, curr_frame = cap.read()
                    if not success:
                        break

                if not success:
                    break

                # 调整帧为灰度图以计算SSIM
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

                # 计算帧之间的相似性
                score, _ = ssim(prev_gray, curr_gray, full=True)

                # 保存前一帧
                saved_frame_path = os.path.join(output_folder, f"{video_name}_{saved_index}.jpg")
                cv2.imwrite(saved_frame_path, prev_frame)
                saved_index += 1

                if score < similarity_threshold:  # 如果相似性小于阈值，则保留当前帧
                    saved_frame_path = os.path.join(output_folder, f"{video_name}_{saved_index}.jpg")
                    cv2.imwrite(saved_frame_path, curr_frame)
                    saved_index += 1

                prev_frame = curr_frame
                pbar.update(frame_skip)  # 更新进度条

            # 保存最后一帧
            saved_frame_path = os.path.join(output_folder, f"{video_name}_{saved_index}.jpg")
            cv2.imwrite(saved_frame_path, prev_frame)

        cap.release()

    print(f"处理完成，结果保存在 {output_root} 中。")

# 示例用法
input_folder = "C:/Users/Orange/Desktop/review_generation/video"
output_root = "C:/Users/Orange/Desktop/review_generation/image"
frame_interval = 1  # 选择抽帧的时间间隔（秒）
similarity_threshold = 0.5  # 设定帧之间的相似性阈值

process_videos(input_folder, output_root, similarity_threshold, frame_interval)
