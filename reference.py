import json
import re
import torch
import whisper
from jiwer import wer as wer_fn
import sacrebleu
from sacrebleu import sentence_bleu

# ---------- 超参数 (请根据实际情况修改) ----------
JSONL_PATH = "/home/ziheng/whisper-finetune/dataset/union/test/data.jsonl"
MODEL_WEIGHTS_PATH = "/home/ziheng/whisper-finetune/src/whisper_finetune/scripts/20250427_183033_my_whisper_run/best_model.pt"
# -----------------------------------------------------------------

def clean_timestamps(text: str) -> str:
    """
    Remove timestamp tokens of the form <|xx.xx|> or <|x.xx|> from the text.
    """
    return re.sub(r'<\|\d+\.\d+\|>', '', text)

def main():
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 Whisper small 模型
    model = whisper.load_model("small", device=device)

    # 加载自定义权重
    checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict.pop("dims", None)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    # 用于汇总的容器
    transcribe_scores = []
    translate_refs = []
    translate_hyps = []

    # 读取并处理 JSONL
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            item = json.loads(line)
            audio_path = item["audio"]
            raw_ref = item["text"]
            task = item["task"]

            # 清理 reference 中的时间戳
            reference = clean_timestamps(raw_ref).strip()

            # 调用 Whisper
            result = model.transcribe(audio_path, task=task, language="su", verbose=True)
            hypothesis = result["text"].strip()

            if task == "transcribe":
                # 清理 hypothesis 并计算 WER
                hyp_clean = clean_timestamps(hypothesis)
                score = wer_fn(reference, hyp_clean)
                transcribe_scores.append(score)
                print(f"[{idx}] Transcribe | WER: {score:.3f}")
                print(f"      Ref: {reference}")
                print(f"      Hyp: {hyp_clean}\n")
            elif task == "translate":
                translate_refs.append(reference)
                translate_hyps.append(hypothesis)
                # 单句 BLEU
                sent_bleu = sentence_bleu(hypothesis, [reference])
                print(f"[{idx}] Translate  | Sent BLEU: {sent_bleu.score:.2f}")
                print(f"      Ref: {reference}")
                print(f"      Hyp: {hypothesis}\n")
            else:
                print(f"[{idx}] Warning: Unknown task '{task}'\n")

    # 打印汇总指标
    if transcribe_scores:
        avg_wer = sum(transcribe_scores) / len(transcribe_scores)
        print(f"Average WER over {len(transcribe_scores)} samples: {avg_wer:.3f}")
    if translate_refs:
        bleu = sacrebleu.corpus_bleu(translate_hyps, [translate_refs])
        print(f"Corpus BLEU over {len(translate_refs)} samples: {bleu.score:.2f}")

if __name__ == "__main__":
    main()