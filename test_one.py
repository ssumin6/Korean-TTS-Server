import torch
import getopt
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import os
import sys
import timeit
import time


import audio
from FastSpeech import FastSpeech
import hparams as hp
from text.text import text_to_sequence


from WER import waveToText
from WER import WERCER

os.environ['GOOGLE_APPLICATION_CREDITIALS'] = "magellan-voice-ui-86932243453a.json"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_data(data, figsize=(12, 4)):
    _, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto',
                       origin='bottom', interpolation='none')
    plt.savefig(os.path.join("img", "model_test.jpg"))


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

    pos = torch.stack([torch.Tensor([i+1 for i in range(text.size(1))])])
    pos = pos.long().to(device)

    #start = timeit.default_timer()
    start = int(round(time.time() * 1000))
    print("start: %d" %start)

    model.eval()
    with torch.no_grad():
        mel, mel_postnet = model(text, pos, alpha=alpha)

    end = int(round(time.time() * 1000))
    print("end : %d" %end)
    tt = end- start
    print("Total - making mel : %d" %tt)


    mel = mel[0].cpu().numpy().T
    mel_postnet = mel_postnet[0].cpu().numpy().T
    #plot_data([mel, mel_postnet])

    wav = audio.inv_mel_spectrogram(mel_postnet)
    print("Wav Have Been Synthesized.")

    if not os.path.exists("results"):
        os.mkdir("results")
    new_name = text_seq.replace(" ","_")
    new_name2 = new_name+str(num)+mode+".wav"

    audio.save_wav(wav, os.path.join("results", new_name2))

    new_name3 = "results/"+ new_name2
    WERCER([new_name3], [str(text_seq)])

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
    print("-u : upload / true - upload to gs bucket")
    return


if __name__ == "__main__":
    # Test
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:s:u:h", ["text=", "step=","upload=", "help"])
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
            upload = True


    model = nn.DataParallel(FastSpeech()).to(device)
    #step_num = 134500
    checkpoint = torch.load(os.path.join(
        hp.checkpoint_path, 'checkpoint_%d.pth.tar' % step_num))
    model.load_state_dict(checkpoint['model'])
    if (torch.cuda.device_count()>1):
        model = model.module
    print("Model Have Been Loaded.")
    
    start_time = timeit.default_timer()
    print("Start Synthesis : %d" %start_time)
    #words = "만나서 반가워."
    file_name = synthesis_griffin_lim(words, model, alpha=1.0, mode="normal",num = step_num)
    #synthesis_griffin_lim(words, model, alpha=1.5, mode="slow")
    #synthesis_griffin_lim(words, model, alpha=0.5, mode="quick")
    end_time = timeit.default_timer()
    print("End Synthesis : %d" %end_time)
    time = end_time - start_time
    #print("Synthesized.")
    print("Total Time : %d" %time)
    #waveglow = get_waveglow()
    #synthesis_waveglow(words, model, waveglow,
     #                  alpha=1.0, mode="waveglow_normal")
    #print("Synthesized by Waveglow.")

    save_name = words.replace(" ", "_")
 
    file_name = file_name  + str(step_num) +"normal"+ "." + "wav"

    if upload:
        os.system("gsutil cp results/%s gs://nm-voice-intern/result_v100_eng" %file_name)


