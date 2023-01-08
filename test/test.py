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
from inference import mels_to_wavs_GL, generate_mels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_mel_spectrogram(mels, fig_size=(12, 4), file_name = ""):
    _, axes = plt.subplots(1, len(mels), fig_size=figsize)
    for i in range(len(mels)):
        axes[i].imshow(mels[i] , aspect = 'auto',
                       origin='bottom', interpolation='none')
    file_name = file_name.replace(" ", "_")
    plt.savefig(os.path.join("img", "%s.jpg" % file_name))

    os.system("gsutil cp img/%s.jpg gs://nm-voice-intern/results_kor_0730_indiv/plot_img/" %file_name)

def get_waveglow():
    waveglow_path = os.path.join(hp.waveglow_path, 'waveglow_256channels.pt')
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()

    return waveglow

def synthesis_griffin_lim(sentence, tts_model, alpha=1.0, mode="", ckpt_step = 100, useMetric = True, isHighPitchInput = False):
    text_seq = text_to_sequence(sentence, hp.hparams.text_cleaners)
    text_seq = text_seq + [0]
    text_seq = np.stack([np.array(text_seq)])
    text_seq = torch.from_numpy(text_seq).cuda().to(device)

    position_seq = torch.stack([torch.Tensor([i+1 for i in range(text_seq.size(1))])])
    position_seq = position_seq.long().to(device)

    tts_model.eval()

    with torch.no_grad():
        mel, mel_postnet = tts_model(text_seq, position_seq, alpha=alpha)

    generated_audio_dir = "results_kor_0730_indiv"
    if not os.path.exists(generated_audio_dir):
        os.mkdir(generated_audio_dir)
        
    snake_case_name = text_seq.replace(" ","_")
    snake_case_name = snake_case_name.replace("?","_")
    if (isHighPitchInput):
        snake_case_name = snake_case_name + "_high_pitch"
    file_name = snake_case_name + str(ckpt_step) + ".wav"
    file_path = os.path.join(generated_audio_dir, file_name)

    if (isHighPitchInput):
        mel_postnet = mel_postnet[0].cpu().numpy().T
    else:
        mel_postnet = mel_postnet.data.cpu().numpy()[0].T
        mel_postnet = mel_postnet[:, :-1]
        mel_postnet = np.append(mel_postnet, np.ones((80, 0), dtype=np.float32)*-4.0, axis=1)

    mel = mel[0].cpu().numpy().T
    plot_mel_spectrogram([mel, mel_postnet], file_name = snake_case_name)

    if (isHighPitchInput):
        wav = audio.inv_mel_spectrogram(mel_postnet)
    else:
        stft = audio.taco_stft()
        wav = mels_to_wavs_GL([mel_postnet], stft)

    audio.save_wav(wav, os.path.join(generated_audio_dir, snake_case_name + str(ckpt_step) + ".wav"))
    
    if useMetric:
        clean_text = snake_case_name.replace("_", " ")
        wer, _, _, cer, _, _ = WERCER([file_path], [str(clean_text)])
    else:
        wer, cer = 0, 0

    return snake_case_name, wer, cer

def synthesis_waveglow(sentence, tts_model, waveglow, alpha=1.0, mode=""):
    text_seq = text_to_sequence(sentence, hp.hparams.text_cleaners)
    text_seq = text_seq + [0]
    text_seq = np.stack([np.array(text_seq)])
    text_seq = torch.from_numpy(text_seq).long().to(device)

    position_seq = torch.stack([torch.Tensor([i+1 for i in range(text_seq.size(1))])])
    position_seq = position_seq.long().to(device)

    tts_model.eval()
    with torch.no_grad():
        _, mel_postnet = tts_model(text_seq, position_seq, alpha=alpha)

    with torch.no_grad():
        wav = waveglow.infer(mel_postnet, sigma=0.666)
    print("Wav Have Been Synthesized.")

    generated_audio_dir = "results"
    if not os.path.exists(generated_audio_dir):
        os.mkdir(generated_audio_dir)
    audio.save_wav(wav[0].data.cpu().numpy(), os.path.join(generated_audio_dir, text_seq + mode + ".wav"))

def help():
    print("help usage")
    print("-t : text (korean)")
    print("-s : ckpt_step")
    print("-u : upload / true - upload to gs bucket")
    print("-w : check wer and cer")
    print("-c : create high_pitched_sound")
    return

if __name__ == "__main__":
    # Test
    copyToDrive = False
    useMetric = False
    isHighPitchInput = False
    
    argExist = 1

    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:s:u:w:h:c", ["text=", "step=", "upload=", "wer=", "help", "cute"])
    except getopt.GetoptError as err:
        print(str(err))
        help()
        sys.exit(1)

    for opt, arg in opts:
        if (opt == "-t"):
            sentence = str(arg)
        if (opt == "-s"):
            ckpt_step = int(arg)
        if (opt == "-u"):
            if int(arg) == argExist:
                copyToDrive = True
        if (opt == "-w"):
            if int(arg) == argExist:
                useMetric = True
        if (opt == "-c"):
            isHighPitchInput = True

    fastspeech_model = nn.DataParallel(FastSpeech()).to(device)
    checkpoint = torch.load(os.path.join(
        hp.checkpoint_path, 'checkpoint_%d.pth.tar' % ckpt_step))
    fastspeech_model.load_state_dict(checkpoint['model'])
    if (torch.cuda.device_count()>1):
        fastspeech_model = fastspeech_model.module
    print("FastSpeech Model Have Been Loaded.\n")
    
    file_name, wer, cer = synthesis_griffin_lim(sentence, fastspeech_model, alpha=1.0, mode="normal", ckpt_step = ckpt_step, useMetric = useMetric, isHighPitchInput=isHighPitchInput)

    if useMetric:
        print("#######################################")
        print("Total Result of WER and CER")
        print("WER : ", wer)
        print("CER : ", cer)

    if copyToDrive:
        file_name = file_name + str(ckpt_step) + "." + "wav"
        os.system("gsutil cp results_kor_0730_indiv/%s gs://nm-voice-intern/results_kor_0730_indiv/" %file_name)