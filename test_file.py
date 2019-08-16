#-*- coding: utf-8 -*-
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

from WER import waveToText
from WER import WERCER

os.environ['GOOGLE_APPLICATION_CREDITIALS'] = "magellan-voice-ui-86932243453a.json"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_data(data, figsize=(12, 4), file_name = ""):
    _, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto',
                       origin='bottom', interpolation='none')
    file_name = file_name.replace(" ", "_")
    plt.savefig(os.path.join("img", "%s.jpg" % file_name))


    os.system("gsutil cp img/%s.jpg gs://nm-voice-intern/results_kor_0730_file/results_kor_0730_nam_95000/plot_img/" %file_name)

def get_waveglow():
    waveglow_path = os.path.join(hp.waveglow_path, 'waveglow_256channels.pt')
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()

    return waveglow


def synthesis_griffin_lim(text_seq, model, alpha=1.0, mode="", num = 100, check=True):
    text_seq = text_seq[:-1]
    text = text_to_sequence(text_seq, hp.hparams.text_cleaners)
    text = text + [0]
    text = np.stack([np.array(text)])
    text = torch.from_numpy(text).long().to(device)

    sequence = np.array(text_to_sequence(text_seq, hp.hparams.text_cleaners))[None, 1]
    pos = torch.stack([torch.Tensor([i+1 for i in range(text.size(1))])])
    pos = pos.long().to(device)

    model.eval()


    with torch.no_grad():
        mel, mel_postnet = model(text, pos, alpha=alpha)
    
    if not os.path.exists("results_kor_0730_nam_95000"):
        os.mkdir("results_kor_0730_nam_95000")
    new_name = text_seq.replace(" ","_")
    new_name = new_name.replace("?","_")
    
    new_name = new_name[:-1]
    new_name2 = new_name + str(num) + mode + ".wav"
    new_name3 = "results_kor_0730_nam_95000/" + new_name2

    mel = mel[0].cpu().numpy().T
    mel_postnet = mel_postnet[0].cpu().numpy().T
    plot_data([mel, mel_postnet], file_name = new_name)

    start = int(round(time.time() * 1000))
    wav = audio.inv_mel_spectrogram(mel_postnet)
    end = int(round(time.time() * 1000))
    audio.save_wav(wav, os.path.join("results_kor_0730_nam_95000", new_name2))
    clean_text = new_name.replace("_", " ")
    if check:
        x, _, _, y, _, _ = WERCER([new_name3], [str(clean_text)])
    else:
        x = 0
        y = 0
    print("Total time : ", end - start)
    print()
    return new_name, x, y

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

    if not os.path.exists("results_48000"):
        os.mkdir("results_48000")
    audio.save_wav(wav[0].data.cpu().numpy(), os.path.join(
        "results_48000", text_seq + mode + ".wav"))

def help():
    print("help usage")
    print("-t : text (korean)")
    print("-s : step_num")
    print("-u : upload / true - upload to gs bucket")
    print("-w : True/ False")
    return


if __name__ == "__main__":
    # Test

    upload = False
    check = False
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:s:u:w:h", ["text=", "step=","upload=", "wer=", "help"])
    except getopt.GetoptError as err:
        print(str(err))
        help()
        sys.exit(1)

    for opt, arg in opts:
        if (opt == "-t"):
            words = str(arg)
        if (opt == "-s"):
            step_num = int(arg)
        if (opt == "-u"):
            tmp = int(arg)
            if tmp == 1:
                upload = True
        if (opt == "-w"):
           tmp = int(arg)
           if tmp == 1:
               check = True

    model = nn.DataParallel(FastSpeech()).to(device)
    #step_num = 134500
    checkpoint = torch.load(os.path.join(
        hp.checkpoint_path, 'checkpoint_%d.pth.tar' %step_num))
    model.load_state_dict(checkpoint['model'])
    if (torch.cuda.device_count()>1):
        model = model.module
    print("Model Have Been Loaded.\n")
    wer = 0
    cer = 0
    i = 0
    #words = "만나서 반가워."
    f = open(words, "rb")
    while True:
        line = f.readline()
        line = line.decode('utf-8')
        if not line: break
        i += 1
        file_name, x, y = synthesis_griffin_lim(line, model, alpha=1.0, mode="normal", num=step_num, check=check)
        wer += x
        cer += y

    avg_wer = wer / i
    avg_cer = cer / i
    if check:
        print()
        print("##############################################")
        print("Total Result of WER and CER")
        print("Average WER :", avg_wer)
        print("Average CER : ", avg_cer)
    f.close()
       


    #file_name = synthesis_griffin_lim(words, model, alpha=1.0, mode="normal",num = step_num)
    #synthesis_griffin_lim(words, model, alpha=1.5, mode="slow")
    #synthesis_griffin_lim(words, model, alpha=0.5, mode="quick")
    # print("Synthesized.\n")

    #waveglow = get_waveglow()
    #synthesis_waveglow(words, model, waveglow,
     #                  alpha=1.0, mode="waveglow_normal")
    #print("Synthesized by Waveglow.")


    if upload:
        save_name = words.replace(" ", "_")
        file_name = file_name  + str(step_num) +"normal"+ "." + "wav"
        os.system("gsutil cp -r results_kor_0730_nam_95000/ gs://nm-voice-intern/results_kor_0730_file/")


