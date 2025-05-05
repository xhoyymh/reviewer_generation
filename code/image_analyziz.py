import os
import base64
import requests
import json
from tqdm import tqdm
from collections import defaultdict
import logging

# --- 配置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 常量和配置 ---
# 主文件夹路径 (请确保路径正确)
ROOT_FOLDER = "C:/Users/Orange/Desktop/review_generation/image"
# 保存问答记录的文件路径 (修改了文件名以区分)
OUTPUT_FILE = "C:/Users/Orange/Desktop/review_generation/conversation/llava_batched_results_cn.json"
# Ollama API 地址
API_URL = "http://localhost:11434/api/generate"
# 模型名称
MODEL_NAME = "llava"
# 每批处理的图片数量
BATCH_SIZE = 5

# --- 模型参数 ---
# (注意：Ollama API 可能使用不同的参数名，例如 num_predict 而不是 max_tokens)
# (具体请参考你的 Ollama 服务配置或文档)
MODEL_OPTIONS = {
    "temperature": 0.7,  # 控制输出的随机性
    "top_p": 0.9,        # 核采样，影响生成内容的多样性
    "num_predict": 300, # 限制生成内容的最大 token 数 (LLaMA/Ollama 通常用 num_predict)
    "repeat_penalty": 1.2 # 设定惩罚系数，减少重复
    # "presence_penalty": 0.6 # Ollama 可能不支持此参数，需确认
}

# --- 函数：分组和排序图片 ---
def group_and_sort_images(root_dir):
    """扫描目录，按子文件夹分组图片，并尝试按数字排序"""
    images_by_subfolder = defaultdict(list)
    total_image_count = 0
    logging.info(f"开始扫描图片文件夹: {root_dir}")
    if not os.path.isdir(root_dir):
        logging.error(f"错误：指定的根目录不存在或不是一个文件夹: {root_dir}")
        return images_by_subfolder, total_image_count

    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            logging.warning(f"跳过非文件夹项目: {subfolder_path}")
            continue

        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        # 尝试按文件名中的数字排序 (例如 1.jpg, 2.jpg, 10.jpg)
        def sort_key(filename):
            name_part = os.path.splitext(filename)[0]
            try:
                return int(name_part)
            except ValueError:
                # 如果无法转为整数，返回一个很大的数和原始名称，以将非数字名称排在后面
                return (float('inf'), name_part)

        sorted_files = sorted(image_files, key=sort_key)

        for filename in sorted_files:
            full_path = os.path.join(subfolder_path, filename)
            # 存储元组: (子文件夹名, 原始文件名, 完整路径)
            images_by_subfolder[subfolder].append((subfolder, filename, full_path))
            total_image_count += 1

    logging.info(f"扫描完成。在 {len(images_by_subfolder)} 个子文件夹中找到 {total_image_count} 张图片。")
    return images_by_subfolder, total_image_count

# --- 函数：处理单个批次 ---
def process_batch(batch_data, model_name, api_url, model_options):
    """处理一个批次的图片，发送 API 请求并返回结果"""
    encoded_images = []
    batch_image_names_for_json = [] # 用于 JSON 输出的 '子文件夹_序号.jpg' 格式
    batch_original_filenames = []   # 用于 Prompt 的原始文件名

    subfolder_name = batch_data[0][0] # 获取批次所属的子文件夹名

    for index, (subfolder, filename, file_path) in enumerate(batch_data):
        try:
            with open(file_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            encoded_images.append(encoded_image)
            # 使用原始文件名创建 JSON 中的 image_name
            # 注意：这里不再使用enumerate的index，而是用原始文件名
            batch_image_names_for_json.append(f"{subfolder}_{filename}")
            batch_original_filenames.append(filename)
        except FileNotFoundError:
            logging.warning(f"警告：文件未找到 {file_path}，已跳过。")
        except Exception as e:
            logging.warning(f"警告：编码文件时出错 {file_path}: {e}，已跳过。")

    if not encoded_images:
        logging.warning(f"警告：批次 {subfolder_name} (起始于 {batch_data[0][1]}) 没有可处理的图片，已跳过API请求。")
        return None # 表明此批次无效

    # 创建组合的图片名称字符串 (用于 JSON)
    image_name_str_json = ", ".join(batch_image_names_for_json)
    # 创建组合的原始文件名字符串 (用于 Prompt)
    original_filename_str_prompt = ", ".join(batch_original_filenames)

    # --- 构建 Prompt 和 API 请求数据 ---
    prompt_text = (
        f"请描述以下 {len(encoded_images)} 张连续图片序列的主要内容或场景变化。这些图片来自视频片段 '{subfolder_name}'，"
        f"文件名为: '{original_filename_str_prompt}'。\n"
        f"约束条件: 1. 描述应连贯概括整个序列。 2. 描述尽量简洁，最好在100字以内。"
    )

    data = {
        "model": model_name,
        "prompt": prompt_text,
        "images": encoded_images, # 包含批次中所有图片的 Base64 列表
        "stream": False,          # 设置为 False 以获取单个完整响应
        "options": model_options  # 将模型参数放在 options 字典中
    }

    headers = {"Content-Type": "application/json"}
    response_text = ""
    logging.debug(f"正在为批次发送请求: {image_name_str_json}")

    try:
        response = requests.post(api_url, json=data, headers=headers, timeout=180) # 增加超时时间
        response.raise_for_status() # 检查 HTTP 错误状态码 (4xx 或 5xx)
        response_data = response.json()
        response_text = response_data.get("response", "").strip()
        logging.debug(f"收到批次响应: {image_name_str_json}")

    except requests.exceptions.Timeout:
        logging.error(f"API 请求超时 (批次: {image_name_str_json})")
        response_text = "错误：API 请求超时"
    except requests.exceptions.RequestException as e:
        logging.error(f"API 请求失败 (批次: {image_name_str_json}): {e}")
        response_text = f"错误：API 请求失败 - {e}"
    except json.JSONDecodeError as e:
        logging.error(f"解析 API 响应失败 (批次: {image_name_str_json}): {e}")
        logging.error(f"原始响应内容: {response.text}")
        response_text = f"错误：无法解析 API 响应 - {e}"
    except Exception as e:
        logging.error(f"处理批次时发生意外错误 {image_name_str_json}: {e}")
        response_text = f"错误：发生意外错误 - {e}"

    return {
        "image_name": image_name_str_json,
        "prompt": prompt_text,
        "response": response_text
    }

# --- 主程序 ---
if __name__ == "__main__":
    # 步骤 1: 分组和排序图片
    images_by_subfolder, total_image_count = group_and_sort_images(ROOT_FOLDER)

    if total_image_count == 0:
        logging.info("未找到任何图片，程序退出。")
        exit()

    qa_records = []
    processed_image_count = 0

    # 步骤 2: 按批次处理图片
    # 使用 tqdm 显示基于图片总数的进度条
    with tqdm(total=total_image_count, desc="处理图片进度", unit="张") as pbar:
        # 遍历每个子文件夹（视频）
        for subfolder, image_data_list in images_by_subfolder.items():
            if not image_data_list:
                continue # 跳过空子文件夹

            # 在子文件夹内按 BATCH_SIZE 分批
            for i in range(0, len(image_data_list), BATCH_SIZE):
                batch = image_data_list[i : i + BATCH_SIZE] # 获取当前批次

                # 处理批次并获取结果
                batch_result = process_batch(batch, MODEL_NAME, API_URL, MODEL_OPTIONS)

                if batch_result: # 如果批次处理成功
                    qa_records.append(batch_result)

                # 更新进度条（即使批次处理失败，也认为这些图片“尝试过处理”）
                pbar.update(len(batch))
                processed_image_count += len(batch) # 统计实际处理（或尝试处理）的图片数

    # 步骤 3: 写入 JSON 文件
    logging.info(f"总共处理（或尝试处理）了 {processed_image_count} 张图片。")
    if qa_records:
        try:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(qa_records, f, ensure_ascii=False, indent=4)
            logging.info(f"问答记录已成功保存到: {OUTPUT_FILE}")
        except IOError as e:
            logging.error(f"错误：写入输出文件失败 {OUTPUT_FILE}: {e}")
        except Exception as e:
            logging.error(f"错误：写入 JSON 文件时发生意外错误: {e}")
    else:
        logging.warning("没有成功处理的批次，未生成 JSON 文件。")