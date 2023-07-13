import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from text import text_to_sequence
import hparams as hp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FastSpeechDataset(Dataset):
    """LJSpeech"""

    def __init__(self, dataset_path=hp.dataset_path):
        self.dataset_path = dataset_path
        self.text_path = os.path.join(self.dataset_path, "train.txt")
        self.text = process_text(self.text_path)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        index = idx + 1
        mel_name = os.path.join(
            self.dataset_path, "ljspeech-mel-%05d.npy" % index
        )
        mel_np = np.load(mel_name)

        character = self.text[idx]
        character = text_to_sequence(character, hp.hparams.text_cleaners)
        character = np.array(character)

        if not hp.pre_target:
            return {"text": character, "mel": mel_np}
        else:
            alignment = np.load(
                os.path.join(hp.alignment_target_path, str(idx) + ".npy")
            )

            return {"text": character, "mel": mel_np, "alignment": alignment}


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        inx = 0
        txt = []
        for line in f.readlines():
            cnt = 0
            for index, ele in enumerate(line):
                if ele == "|":
                    cnt = cnt + 1
                    if cnt == 2:
                        inx = index
                        end = len(line)
                        txt.append(line[inx + 1 : end - 1])
                        break

        return txt


def collate_fn(batch):
    texts = [d["text"] for d in batch]
    mels = [d["mel"] for d in batch]

    if not hp.pre_target:
        texts, pos_padded = pad_text(texts)
        mels = pad_mel(mels)

        return {"texts": texts, "pos": pos_padded, "mels": mels}
    else:
        alignment_target = [d["alignment"] for d in batch]

        texts, pos_padded = pad_text(texts)
        alignment_target = pad_alignment(alignment_target)
        mels = pad_mel(mels)

        return {
            "texts": texts,
            "pos": pos_padded,
            "mels": mels,
            "alignment": alignment_target,
        }


def pad_text(inputs):
    def pad_data(x, length):
        pad = 0
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=pad
        )
        pos_padded = np.pad(
            np.array([(i + 1) for i in range(np.shape(x)[0])]),
            (0, length - x.shape[0]),
            mode="constant",
            constant_values=pad,
        )

        return x_padded, pos_padded

    max_len = max((len(x) for x in inputs))

    text_padded = np.stack([pad_data(x, max_len)[0] for x in inputs])
    pos_padded = np.stack([pad_data(x, max_len)[1] for x in inputs])

    return text_padded, pos_padded


def pad_alignment(alignment):
    def pad_data(x, length):
        pad = 0
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=pad
        )

        return x_padded

    max_len = max((len(x) for x in alignment))

    alignment_padded = np.stack([pad_data(x, max_len) for x in alignment])

    return alignment_padded


def pad_mel(inputs):
    def pad(x, max_len):
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x = np.pad(
            x,
            (0, max_len - np.shape(x)[0]),
            mode="constant",
            constant_values=0,
        )

        return x[:, :s]

    max_len = max(np.shape(x)[0] for x in inputs)
    mel_output = np.stack([pad(x, max_len) for x in inputs])

    return mel_output


class data_prefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            data = next(self.loader)
            if hp.pre_target:
                self.next_texts = data["texts"]
                self.next_pos = data["pos"]
                self.next_mels = data["mels"]
                self.next_alignment = data["alignment"]
            else:
                self.next_texts = data["texts"]
                self.next_pos = data["pos"]
                self.next_mels = data["mels"]
        except StopIteration:
            if hp.pre_target:
                self.next_texts = None
                self.next_pos = None
                self.next_mels = None
                self.next_alignment = None
            else:
                self.next_texts = None
                self.next_pos = None
                self.next_mels = None
            return
        with torch.cuda.stream(self.stream):
            if hp.pre_target:
                self.next_texts = (
                    torch.from_numpy(self.next_texts).long().to(device)
                )
                self.next_pos = (
                    torch.from_numpy(self.next_pos).long().to(device)
                )
                self.next_mels = (
                    torch.from_numpy(self.next_mels).float().to(device)
                )
                self.next_alignment = (
                    torch.from_numpy(self.next_alignment).float().to(device)
                )
            else:
                self.next_texts = (
                    torch.from_numpy(self.next_texts).long().to(device)
                )
                self.next_pos = (
                    torch.from_numpy(self.next_pos).long().to(device)
                )
                self.next_mels = (
                    torch.from_numpy(self.next_mels).float().to(device)
                )

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if hp.pre_target:
            texts = self.next_texts
            pos = self.next_pos
            mels = self.next_mels
            alignment = self.next_alignment
            self.preload()
            return texts, pos, mels, alignment
        else:
            texts = self.next_texts
            pos = self.next_pos
            mels = self.next_mels
            self.preload()
            return texts, pos, mels
