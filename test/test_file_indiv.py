import matplotlib
matplotlib.use('Agg')
import torch
import getopt
import torch.nn as nn
import numpy as np
import matplotlib.pylab as plt
import os
import sys

import time
import audio
from FastSpeech import FastSpeech
import hparams as hp
from text.text import text_to_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_data(data, figsize=(12, 4), file_name = ""):
    _, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto',
                       origin='bottom', interpolation='none')
    file_name = file_name.replace(" ", "_")
    plt.savefig(os.path.join("img", "%s.jpg" % file_name))


    os.system("gsutil cp img/%s.jpg gs://nm-voice-intern/plot_img" %file_name)

def get_waveglow():
    waveglow_path = os.path.join(hp.waveglow_path, 'waveglow_256channels.pt')
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()

    return waveglow


def synthesis_griffin_lim(text_seq, model, alpha=1.0, mode="", num = 100):
    text = text_to_sequence(text_seq, hp.hparams.text_cleaners)
    text = text + [0]
    text = np.stack([np.array(text)])
    text = torch.from_numpy(text).long().to(device)

    sequence = np.array(text_to_sequence(text_seq, hp.hparams.text_cleaners))[None, 1]
    print("sequence size", np.shape(sequence))

    
    
    pos = torch.stack([torch.Tensor([i+1 for i in range(text.size(1))])])
    pos = pos.long().to(device)

    start = int(round(time.time() * 1000))
    model.eval()


    with torch.no_grad():
        mel, mel_postnet = model(text, pos, alpha=alpha)

    end = int(round(time.time() * 1000))
    t = end - start
    print("Total mel-spectrogram :%d ms\n" %t)

    mel = mel[0].cpu().numpy().T
    mel_postnet = mel_postnet[0].cpu().numpy().T
    plot_data([mel, mel_postnet], file_name = text_seq)

    wav = audio.inv_mel_spectrogram(mel_postnet)
    print("\nWav Have Been Synthesized.")

    if not os.path.exists("results_fastspeech"):
        os.mkdir("results_fastspeech")
    new_name = text_seq.replace(" ","_")
    audio.save_wav(wav, os.path.join("results_fastspeech", new_name + str(num) + mode + ".wav"))
    return new_name

def synthesis_waveglow(text_seq, model, waveglow, alpha=1.0, mode=""):
    text = text_to_sequence(text_seq, hp.hparams.text_cleaners)
    text = text + [0]
    text = np.stack([np.array(text)])
    text = torch.from_numpy(text).long().to(device)

    pos = torch.stack([torch.Tensor([i+1 for i in range(text.size(1))])])
    pos = pos.long().to(device)

    model.eval()
    with torch.no_grad():
        _, mel_postnet = model(text, pos, alpha=alpha)
    with torch.no_grad():
        wav = waveglow.infer(mel_postnet, sigma=0.666)
    print("Wav Have Been Synthesized.")

    if not os.path.exists("results"):
        os.mkdir("results")
    audio.save_wav(wav[0].data.cpu().numpy(), os.path.join(
        "results", text_seq + mode + ".wav"))

def help():
    print("help usage")
    print("-t : text (korean)")
    print("-s : step_num")
    return


if __name__ == "__main__":
    # Test

    upload = False
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:s:h", ["text=", "step=", "help"])
    except getopt.GetoptError as err:
        print(str(err))
        help()
        sys.exit(1)

    for opt, arg in opts:
        if (opt == "-t"):
            words = str(arg)
        if (opt == "-s"):
            step_num = int(arg)

    model = nn.DataParallel(FastSpeech()).to(device)
    checkpoint = torch.load(os.path.join(
        hp.checkpoint_path, 'checkpoint_%d.pth.tar' %step_num))
    model.load_state_dict(checkpoint['model'])
    if (torch.cuda.device_count()>1):
        model = model.module
    print("Model Have Been Loaded.\n")

    file_name = synthesis_griffin_lim(words, model, alpha=1.0, mode="normal",num = step_num)

