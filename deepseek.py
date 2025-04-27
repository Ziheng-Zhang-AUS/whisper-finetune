# # Please install OpenAI SDK first: `pip3 install openai`

# from openai import OpenAI

# client = OpenAI(api_key="sk-fa9ccc0f5fc64eef99caf0f617ea8e33", base_url="https://api.deepseek.com")

# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=False
# )

# print(response.choices[0].message.content)

#!/usr/bin/env python3
# segment_paragraphs.py

"""
从指定 JSONL 文件读取音频路径和翻译文本，分段、校验时长并拼接音频与翻译。
超参数在脚本顶部设置，无需命令行参数。
"""

#!/usr/bin/env python3
# segment_paragraphs.py

"""
从指定 JSONL 文件读取音频路径和翻译文本，实时分段、校验时长并拼接音频与翻译。
超参数在脚本顶部设置，无需命令行参数。
"""
import json
import logging
import re
import wave
from pathlib import Path
from typing import List

# ---------- 安装并导入 OpenAI SDK ----------
# Please install OpenAI SDK first: `pip3 install openai`
from openai import OpenAI
from pydub import AudioSegment

# ---------- 超参数配置 ----------
# 输入 JSONL 路径，请修改为实际路径
INPUT_JSONL = "/home/ziheng/whisper-finetune/dataset/translate_sentences/train/data.jsonl"
# 输出目录，请修改为实际路径；拼接的 wav 和 output.jsonl 都会生成在此目录下
OUTPUT_DIR = "/home/ziheng/whisper-finetune/dataset/translate_ts/train/"
# 每次调用 DeepSeek 的行数上限
CHUNK_SIZE = 50
# 每段最大音频时长（秒）
MAX_DURATION = 28.0
# DeepSeek API 配置
DEEPSEEK_API_KEY = "sk-fa9ccc0f5fc64eef99caf0f617ea8e33"
DEEPSEEK_URL = "https://api.deepseek.com"

# ---------- 配置日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------- DeepSeek 客户端初始化 ----------
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_URL)

# ---------- DeepSeek 调用 & 解析 ----------
def call_deepseek(prompt: str) -> str:
    """
    调用 DeepSeek-chat 模型完成分段，返回原始文本响应
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return response.choices[0].message.content


def extract_json(text: str) -> dict:
    """
    从 DeepSeek 返回文本中提取 JSON 块并解析，
    支持剥离 ```json```、''' 等包裹
    """
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"(\{.*\})", text, flags=re.S)
    if not m:
        raise ValueError("无法从 DeepSeek 返回中抽取到 JSON 块")
    return json.loads(m.group(1))

# ---------- 音频时长计算 ----------
def get_wav_duration(path: Path) -> float:
    """返回 wav 时长(秒)"""
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

# ---------- 主流程 ----------
def main():
    # 准备输出目录和 JSONL
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = out_dir / "output.jsonl"
    fw = output_jsonl.open("w", encoding="utf-8")

    # 1. 读取 JSONL
    logger.info("加载输入文件：%s", INPUT_JSONL)
    records = []
    for i, line in enumerate(Path(INPUT_JSONL).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        obj = json.loads(line)
        records.append({"index": i, **obj})
    logger.info("共读取 %d 条记录", len(records))

    # 统计全局段计数
    segment_counter = 0

    # 2. 分批调用 DeepSeek 并实时处理
    for offset in range(0, len(records), CHUNK_SIZE):
        chunk = records[offset: offset + CHUNK_SIZE]
        indices = [r["index"] for r in chunk]
        texts = [r["text"] for r in chunk]
        logger.info("DeepSeek 分段：处理 lines %d–%d", indices[0], indices[-1])

        prompt = (
            "我有一系列翻译片段，其原始行号为 %s，文本为：\n```\n%s\n```"
            "请基于语义判断哪些相邻文本可组成完整段落，返回 JSON："
            "{\"paragraphs\":[{\"paragraph_id\":1,\"lines\":[0,1,2]},...]},"
            "不要考虑时长。"
        ) % (indices, texts)

        try:
            raw = call_deepseek(prompt)
            parsed = extract_json(raw)
            paragraphs = parsed.get("paragraphs", [])
            logger.info("本块得到 %d 段初步分段", len(paragraphs))
        except Exception as e:
            logger.error("DeepSeek 解析失败：%s", e)
            # 回退到每行独立为段
            paragraphs = [{"paragraph_id": None, "lines": [r["index"]]} for r in chunk]
            logger.info("退回：每行单独成段，共 %d 段", len(paragraphs))

        # 对每个初步段做时长拆分 & 拼接输出
        for p in paragraphs:
            lines = [indices[i] for i in p["lines"]] if p.get("lines") else p["lines"]
            # 3. 时长校验 + 拆分成不超限的子段
            cum = 0.0
            current = []
            for idx in lines:
                dur = get_wav_duration(Path(records[idx]["audio"]))
                if cum + dur > MAX_DURATION and current:
                    # 生成子段
                    segment_counter += 1
                    _process_and_write(records, current, out_dir, fw, segment_counter)
                    current = []
                    cum = 0.0
                current.append(idx)
                cum += dur
            if current:
                segment_counter += 1
                _process_and_write(records, current, out_dir, fw, segment_counter)

    fw.close()
    logger.info("处理完毕，共输出 %d 段，文件保存在 %s", segment_counter, OUTPUT_DIR)


def _process_and_write(records: List[dict], lines: List[int], out_dir: Path, fw, seg_id: int):
    """
    拼接指定行的音频与文本，并写入 JSONL（增加时间戳和 task 字段）
    """
    combined = None
    # 用于记录拼接前的累计偏移时间（秒），并四舍五入到 0.02 倍数
    offset = 0.0
    texts_with_ts = []

    for idx in lines:
        wav_path = Path(records[idx]["audio"])
        logger.debug("  加入音频：%s", wav_path)
        piece = AudioSegment.from_wav(str(wav_path))
        combined = piece if combined is None else combined + piece

        # 获取当前片段时长（秒），并四舍五入到最接近的 0.02 倍数
        dur = get_wav_duration(wav_path)
        dur = round(dur / 0.02) * 0.02
        # 格式化偏移时间为两位小数
        ts = f"<|{offset:.2f}|>"
        texts_with_ts.append(ts + records[idx]["text"])
        offset += dur

    # 在文件名前增加 ts_ 前缀
    out_wav = out_dir / f"ts_segment_{seg_id}.wav"
    combined.export(str(out_wav), format="wav")
    # 拼接所有带时间戳的文本部分
    txt = "".join(texts_with_ts)
    entry = {"audio": str(out_wav), "text": txt, "task": "translate"}
    fw.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("输出 段 %d → %s (%d files, %.1f s)",
                seg_id, out_wav.name, len(lines), len(combined) / 1000.0)

if __name__ == "__main__":
    main()
