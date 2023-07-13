import matplotlib

matplotlib.use('Agg')
import os
import sys
import audio
import torch
import getopt

import numpy as np
import hparams as hp
import torch.nn as nn
import matplotlib.pylab as plt

from WER import WERCER
from FastSpeech import FastSpeech
from text.text import text_to_sequence
from utils import plot_mel_spectrogram
from synthesis import synthesis_griffin_lim, synthesis_waveglow
from inference import mels_to_wavs_GL, generate_mels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generateMelsWithTextModel(sentence, tts_model):
    text_seq = text_to_sequence(sentence, hp.hparams.text_cleaners)
    text_seq = text_seq + [0]
    text_seq = np.stack([np.array(text_seq)])
    text_seq = torch.from_numpy(text_seq).cuda().to(device)

    position_seq = torch.stack(
        [torch.Tensor([i + 1 for i in range(text_seq.size(1))])])
    position_seq = position_seq.long().to(device)

    tts_model.eval()
    with torch.no_grad():
        mel, mel_postnet = tts_model(text_seq, position_seq, alpha=1.0)

    return mel, mel_postnet


def synthesisAudioFromMels(mels, isGriffin=True):
    if (isGriffin):
        wavs = synthesis_griffin_lim(mels)
    else:
        wavs = synthesis_waveglow(mels)

    return wavs


def evaluate(wavs, text):
    clean_text = snake_case_name.replace("_", " ")
    wer, _, _, cer, _, _ = WERCER([file_path], [str(clean_text)])
    return wer, cer


def help():
    print("help usage")
    print("-t : text (korean)")
    print("-s : ckpt_step")
    print("-w : check wer and cer")
    return


if __name__ == "__main__":
    # Test
    useMetric = False
    argExist = 1

    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:s:w:h",
                                   ["text=", "step=", "wer=", "help"])
    except getopt.GetoptError as err:
        print(str(err))
        help()
        sys.exit(1)

    for opt, arg in opts:
        if (opt == "-t"):
            sentence = str(arg)
        if (opt == "-s"):
            ckpt_step = int(arg)
        if (opt == "-w"):
            if int(arg) == argExist:
                useMetric = True

    fastspeech_model = nn.DataParallel(FastSpeech()).to(device)

    ckpt_path = os.path.join(hp.checkpoint_path,
                             'checkpoint_%d.pth.tar' % ckpt_step)
    ckpt = torch.load(ckpt_path)['model']
    fastspeech_model.load_state_dict(ckpt)

    if (torch.cuda.device_count() > 1):
        fastspeech_model = fastspeech_model.module
    print("FastSpeech Model Have Been Loaded.\n")

    mels, mel_postnet = generateMelsWithTextModel(sentence, fastspeech_model)
    plot_mel_spectrogram([mel, mel_postnet], file_name="mels.png")

    audio = synthesisAudioFromMels([mels, mel_postnet])
    audio.save_wav(wavs, file_name)

    if useMetric:
        wer, cer = evaluate(audio)
        print("#######################################")
        print("Total Result of WER and CER")
        print("WER : ", wer)
        print("CER : ", cer)
