import os
import json
from pydub import AudioSegment

# ———— 超参数 ————
# key 可以随意命名，value 是包含 data.jsonl 的文件夹路径
folder_paths = {
    "train": "/home/ziheng/whisper-finetune/dataset/union/train",
    "val": "/home/ziheng/whisper-finetune/dataset/union/val",
    "test": "/home/ziheng/whisper-finetune/dataset/union/test",
    # … 如有更多目录，继续添加
}

# 只保留时长 > 30 秒的音频
THRESHOLD_SECONDS = 30.0

# 输出目录（请修改为你需要写入 tc.txt/ts.txt 的位置）
output_dir = "/home/ziheng/whisper-finetune/stat"
os.makedirs(output_dir, exist_ok=True)


# —— 1) 按 task 收集所有 audio 路径 —— #
audio_by_task = {
    "transcribe": [],
    "translate": [],
}

for folder in folder_paths.values():
    jsonl_path = os.path.join(folder, "data.jsonl")
    if not os.path.isfile(jsonl_path):
        print(f"⚠️ Warning: `{folder}` 下未找到 data.jsonl，已跳过。")
        continue

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            audio_path = obj.get("audio")
            task      = obj.get("task")
            if audio_path and task in audio_by_task:
                audio_by_task[task].append(audio_path)


# —— 2) 计算时长并筛选 —— #
# 最终结果：键 'tc' 对应 transcribe，'ts' 对应 translate
filtered = {"tc": [], "ts": []}

for task, paths in audio_by_task.items():
    key = "tc" if task == "transcribe" else "ts"
    for audio in paths:
        try:
            seg = AudioSegment.from_file(audio)
            dur = seg.duration_seconds
        except Exception as e:
            print(f"❗ 无法读取 `{audio}`：{e}")
            continue

        if dur < THRESHOLD_SECONDS:
            # 如果你想同时记录文件路径，可改为： filtered[key].append((audio, dur))
            filtered[key].append(dur)


# —— 3) 写入 txt —— #
for key, durations in filtered.items():
    out_path = os.path.join(output_dir, f"{key}.txt")
    with open(out_path, "w", encoding="utf-8") as fout:
        for dur in durations:
            fout.write(f"{dur:.2f}\n")
    print(f"✅ 已写入 {len(durations)} 条记录到 {out_path}")
