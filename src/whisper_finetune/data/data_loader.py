import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import random

import numpy as np
import torch
import torchaudio.transforms as T
from datasets import Dataset as HU_Dataset
from numpy import ndarray
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch_audiomentations import AddColoredNoise, HighPassFilter, LowPassFilter
from whisper.audio import CHUNK_LENGTH, N_FRAMES, N_SAMPLES, log_mel_spectrogram
from whisper.tokenizer import Tokenizer

from whisper_finetune.data.utils import TimeWarpAugmenter, pad_or_trim


@dataclass
class Record:
    """
    A single training instance for Whisper.
    `text` can include timestamps in the format of <|0.00|>.
    """

    audio_array: ndarray
    text: str  # text including timestamps
    language: str = ""
    prompt: str = ""  # previous text including timestamps


class AudioDataset(Dataset):
    def __init__(
        self,
        hu_dataset: HU_Dataset,
        tokenizer: Tokenizer,
        device: Optional[torch.device] = None,  # CUDA does not allow for multiprocessing.
        no_timestamp_training: bool = False,
        n_mels: int = 80,
        max_prompt_length: int = 223,  # The maximum number of tokens to use for the prompt
        prompt_use_rate: float = 0.5,
        no_timestamps_rate: float = 0.5,
        spec_augment: bool = False,
        spec_augment_params: Optional[dict] = None,
        audio_aug: bool = False,
        audio_augment_params: Optional[dict] = None,
    ) -> None:
        """
        Initializes the class with the given parameters.

        Args:
            hu_dataset (HU_Dataset): The dataset to use.
            tokenizer (Tokenizer): The tokenizer to use.
            device (Optional[torch.device], optional): The device to use. Defaults to None.
            no_timestamp_training (bool, optional): Whether to use no timestamps for training. Defaults to False.
            n_mels (int, optional): The number of mel filters to use. Defaults to 80.
            max_prompt_length (int, optional): The maximum number of tokens to use for the prompt. Defaults to 223.
            prompt_use_rate (float, optional): The rate at which to use prompts. Defaults to 0.5.
            no_timestamps_rate (float, optional): The rate at which to use no timestamps. Defaults to 0.5.
            spec_augment (bool, optional): Whether to use spectrogram augmentation. Defaults to False.
            spec_augment_params (Optional[dict], optional): The parameters for spectrogram augmentation. Defaults to None.
            audio_aug (bool, optional): Whether to use audio augmentation, such as noise, high-pass filter, and low-pass filter. Defaults to False.
            audio_augment_params (Optional[dict], optional): The parameters for audio augmentation. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If the dataset does not contain the required columns.

        """
        self.hu_dataset = hu_dataset
        self.tokenizer = tokenizer
        self.n_mels = n_mels
        self.device = device
        self.no_timestamp_training = no_timestamp_training
        self.max_prompt_length = max_prompt_length
        self.prompt_use_rate = prompt_use_rate
        self.no_timestamps_rate = no_timestamps_rate
        self.spec_augment = spec_augment
        self.audio_aug = audio_aug
        self.no_timestamps_true_count = 0
        self.total_samples_count = 0
        
        # print(f"[DEBUG INIT] no_timestamp_training={no_timestamp_training}, no_timestamps_rate={no_timestamps_rate}")



        if spec_augment:
            self.time_masking = T.TimeMasking(time_mask_param=spec_augment_params["time_mask_param"])
            self.freq_masking = T.FrequencyMasking(freq_mask_param=spec_augment_params["freq_mask_param"])
            self.time_warping = TimeWarpAugmenter(W=spec_augment_params["time_warp_w"])
        if self.audio_aug:
            self.acn = AddColoredNoise(**audio_augment_params["acn"])
            self.lpf = LowPassFilter(**audio_augment_params["lpf"])
            self.hpf = HighPassFilter(**audio_augment_params["hpf"])

        self.num_frames_per_second = N_FRAMES / CHUNK_LENGTH
        # timestamps tokens are from <|0.00|> to <|30.00|> with a step of 0.02
        self.timestamp_pattern = re.compile(r"(<\|[123]?[0-9]\.[0-9][0-9]\|>)")
        self.model_n_text_ctx = 448
        self.hu_dataset = self.hu_dataset.with_format(type="torch")

        # Some checks
        # assert np.intersect1d(self.hu_dataset.column_names, ["audio", "text", "language"]).size == 3
        assert np.intersect1d(self.hu_dataset.column_names, ["audio", "text", "language", "task"]).size == 4


    def __len__(self) -> int:
        return len(self.hu_dataset)

    def _get_prompt_tokens(self, record: Record, no_timestamps: bool) -> List[int]:
        if torch.rand(1).item() < self.prompt_use_rate and len(record["prompt"]) > 0:
            if no_timestamps:
                prompt_tokens = self._encode_text_without_timestamps(record["prompt"])[-self.max_prompt_length :]
            else:
                prompt_tokens = self._encode_text_with_timestamps(record["prompt"])[-self.max_prompt_length :]
            prompt_tokens = [self.tokenizer.sot_prev] + prompt_tokens
        else:
            prompt_tokens = []

        return prompt_tokens

    def _get_special_tokens(self, is_text_empty: bool, language: str, no_timestamps: bool, task: str) -> List[int]:

        if is_text_empty:
            special_tokens = [self.tokenizer.sot, self.tokenizer.no_speech]
        else:
            special_tokens = [
                self.tokenizer.sot,
                self.tokenizer.special_tokens[f"<|{language}|>"],
                self.tokenizer.special_tokens[f"<|{task}|>"],
            ]
            if no_timestamps:
                special_tokens.append(self.tokenizer.no_timestamps)

        return special_tokens

    def _encode_text_without_timestamps(self, text: str) -> List[int]:
        """Encode text without timestamps by removing timestamps."""
        parts = self.timestamp_pattern.split(text)
        parts = [token for token in parts if token != ""]
        tokens = []
        for part in parts:
            if self.timestamp_pattern.fullmatch(part) is not None:
                timestamp = float(part[2:-2])

                # timestamp must be in the range [0, 30] and be a multiple of 0.02 seconds
                if timestamp < 0 or timestamp > 30 or round(timestamp * 100) % 2 != 0:
                    raise ValueError(f"Invalid timestamp: {timestamp}")
                continue
            else:
                tokens.extend(self.tokenizer.encode(part))

        return tokens

    def _encode_text_with_timestamps(self, text: str) -> List[int]:
        parts = self.timestamp_pattern.split(text)
        parts = [token for token in parts if token != ""]
        tokens = []
        for part in parts:
            if self.timestamp_pattern.fullmatch(part) is not None:
                timestamp = float(part[2:-2])

                # timestamp must be in the range [0, 30] and be a multiple of 0.02 seconds
                if timestamp < 0 or timestamp > 30 or round(timestamp * 100) % 2 != 0:
                    raise ValueError(f"Invalid timestamp: {timestamp}")

                token = self.tokenizer.timestamp_begin + round(timestamp * 100) // 2
                tokens.append(token)
            else:
                tokens.extend(self.tokenizer.encode(part))

        return tokens

    def _get_partial_segment_start(self, tokens: List[int]) -> Optional[float]:
        """If at the end there are two timestamps, use the last one to cut the audio.
        And then zero pad it in the audio-dimension, so that the model learns about silence."""
        if (
            len(tokens) >= 2
            and tokens[-2] >= self.tokenizer.timestamp_begin
            and tokens[-1] >= self.tokenizer.timestamp_begin
        ):  # if the last token is a start time token
            return (tokens[-1] - self.tokenizer.timestamp_begin) * 0.02
        else:
            return None

    def _get_text_tokens(self, text: str, no_timestamps: bool) -> Tuple[List[int], Optional[float]]:
        text_tokens = self._encode_text_with_timestamps(text)
        next_partial_segment_start = self._get_partial_segment_start(text_tokens)
        if no_timestamps:
            text_tokens = list(filter(lambda x: x < self.tokenizer.timestamp_begin, text_tokens))

        return text_tokens, next_partial_segment_start

    def _calculate_mel(
        self, audio_array: ndarray, next_partial_segment_start: Optional[float], no_timestamps: bool
    ) -> torch.Tensor:
        if self.audio_aug:
            audio_array = torch.tensor(audio_array).unsqueeze(0).unsqueeze(0)
            audio_array = self.acn(audio_array)
            audio_array = self.lpf(audio_array)
            audio_array = self.hpf(audio_array)
            audio_array = audio_array.squeeze(0).squeeze(0).numpy()
        mel = log_mel_spectrogram(audio_array, n_mels=self.n_mels, device=self.device)
        if no_timestamps and next_partial_segment_start is not None:
            mel = mel[:, : int(next_partial_segment_start * self.num_frames_per_second)]
        if mel.shape[1] != N_FRAMES:
            mel = pad_or_trim(mel, N_FRAMES)

        if self.spec_augment:
            mel = self.time_warping(mel)
            mel = self.time_masking(mel)
            mel = self.freq_masking(mel)

        return mel

    def _construct_decoder_output(
        self, prompt_tokens: List[int], special_tokens: List[int], text_tokens: List[int]
    ) -> List[int]:
        if len(prompt_tokens) == 0:
            decoder_output = special_tokens[1:] + text_tokens + [self.tokenizer.eot]
        else:
            decoder_output = (
                # Mask out the training loss for predicting the prompt tokens. We use "-100" as the
                # default value for the `ignore_index` parameter in
                # `torch.nn.functional.cross_entropy()`. However, we do not mask out the loss for
                # predicting the sot token because our experiment indicates that the original
                # Whisper model assigns a high probability to the sot token after prompt tokens.
                [-100] * (len(prompt_tokens) - 1)
                + special_tokens
                + text_tokens
                + [self.tokenizer.eot]
            )
        return decoder_output

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.hu_dataset[index]
        no_timestamps = self.no_timestamp_training or torch.rand(1).item() < self.no_timestamps_rate
        
        # print(f"no_timestamps: {no_timestamps}")
        
        self.total_samples_count += 1

        if no_timestamps:
            self.no_timestamps_true_count += 1
            
        # print(f"[DEBUG] total_samples_count so far: {self.total_samples_count}")
        # print(f"[DEBUG] no_timestamps_true_count so far: {self.no_timestamps_true_count}")


        prompt_tokens = self._get_prompt_tokens(record, no_timestamps)
        text_tokens, next_partial_segment_start = self._get_text_tokens(record["text"], no_timestamps)
        is_text_empty = len(text_tokens) == 0
        task = record["task"]
        special_tokens = self._get_special_tokens(is_text_empty, record["language"], no_timestamps, record["task"])


        decoder_input = prompt_tokens + special_tokens + text_tokens
        if len(decoder_input) > self.model_n_text_ctx:
            raise ValueError(f"Input is too long: {record} (length: {len(decoder_input)})")

        decoder_output = self._construct_decoder_output(prompt_tokens, special_tokens, text_tokens)
        audio_arr = record["audio"]["array"]
        # del record

        # Pad in audio domain, not spectrogram domain.
        # https://github.com/openai/whisper/discussions/838#discussioncomment-5233715

        # replace this line with code that can also processs longer than 30s
        # audio_arr = np.pad(audio_arr, (0, N_SAMPLES - audio_arr.shape[0]), "constant")

        if audio_arr.shape[0] > N_SAMPLES:
            audio_arr = audio_arr[:N_SAMPLES]
        else:
            audio_arr = np.pad(audio_arr, (0, N_SAMPLES - audio_arr.shape[0]), "constant")

        mel = self._calculate_mel(audio_arr, next_partial_segment_start, no_timestamps)
        
        # === DEBUG 打印 special tokens 和 timestamp状态 ===
        if random.random() < 0.001:  # 每1000个样本随机打印1个，防止太多
            print("\n[Debug Sample]")
            print(f"Task: {record['task']}")
            print(f"Language: {record['language']}")
            print(f"no_timestamps (是否去掉时间戳token): {no_timestamps}")
            print(f"Special tokens added: {[self.tokenizer.decode([tok]) for tok in special_tokens]}")
            print(f"Prompt tokens length: {len(prompt_tokens)}")
            print(f"Text tokens length: {len(text_tokens)}")
            print(f"Decoder input IDs (first 20): {decoder_input[:20]}")
            print(f"Decoder output IDs (first 20): {decoder_output[:20]}")


        return (
            mel,
            torch.tensor(decoder_input, dtype=torch.int64),
            torch.tensor(decoder_output, dtype=torch.int64),
            record["task"], 
        )


def collate_fn(data):
    # x, y_in, y_out = zip(*data)
    x, y_in, y_out, task = zip(*data)
    x = pad_sequence(x, batch_first=True, padding_value=0)
    y_in = pad_sequence(y_in, batch_first=True, padding_value=0)
    y_out = pad_sequence(y_out, batch_first=True, padding_value=-100)
    return x, y_in, y_out, task


def get_dataloader(
    hu_dataset: HU_Dataset,
    tokenizer: Tokenizer,
    batch_size: int = 1,
    n_mels: int = 80,
    sampler: Optional[torch.utils.data.sampler.Sampler] = None,
    device: Optional[torch.device] = None,  # Does not allow for multiprocessing.
    no_timestamp_training: bool = False,
    max_prompt_length: int = 223,
    prompt_use_rate: float = 0.5,
    no_timestamps_rate: float = 0.5,
    shuffle: bool = True,
    num_workers: int = 0,
    spec_augment: bool = False,
    spec_augment_params: Optional[dict] = None,
    audio_aug: bool = False,
    audio_augment_params: Optional[dict] = None,
) -> DataLoader:
    print(f"Found {len(hu_dataset)} records in the dataset.")
    # print(f"[DEBUG get_dataloader] no_timestamp_training={no_timestamp_training}, no_timestamps_rate={no_timestamps_rate}")

    dataset = AudioDataset(
        hu_dataset,
        tokenizer,
        device=device,
        no_timestamp_training=no_timestamp_training,
        n_mels=n_mels,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=prompt_use_rate,
        no_timestamps_rate=no_timestamps_rate,
        spec_augment=spec_augment,
        spec_augment_params=spec_augment_params,
        audio_aug=audio_aug,
        audio_augment_params=audio_augment_params,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
