import os
import numpy as np
from scipy.io.wavfile import read

# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from text.korean import normalize

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "magellan-voice-ui-86932243453a.json"

def waveToText(file_path):
    # Instantiates a client
    client = speech.SpeechClient()

    sr, content = read(file_path)
    #print(sr, content.min(), content.max())
    audio = types.RecognitionAudio(content=content.tobytes())
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sr,
        language_code='ko-KR')

    # Detects speech in the audio file
    response = client.recognize(config, audio)

    transcripts = ""
    confidences = []
    for result in response.results:
        transcript = result.alternatives[0].transcript
        confidence = result.alternatives[0].confidence
        transcripts += " " + transcript
        confidences.append(confidence)

    return transcripts, confidences


def wer(r, h):
    """
    Reference: https://martin-thoma.com/word-error-rate-calculation/?fbclid=IwAR3CbtlLwZjUcgrDzUHzwYzGx1fLlfw-0OLUDPT3p5U3dgL9G5LGI2zxxek

    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

def WERCER_from_metadata(metadata_path):
    f = open(metadata_path, 'r', encoding='utf-8')
    metas = f.readlines()
    f.close()

    file_paths = [m.strip().split('|')[0] for m in metas]
    org_transcripts = [m.strip().split('|')[1].replace('.', '').replace('!', '').replace('?', '') for m in metas]

    return WERCER(file_paths, org_transcripts)

def WERCER(file_paths, org_transcripts):

    recognized_transcripts = []
    for f in file_paths:
        transcripts, _ = waveToText(f)
        #print(transcripts)
        #cleaned_transcripts = normalize(transcripts)
        cleaned_transcripts = transcripts
        #print(cleaned_transcripts)
        recognized_transcripts.append(cleaned_transcripts)

    print()
    print(org_transcripts)
    print(recognized_transcripts)
    WN = len(' '.join(org_transcripts).split())
    CN = len(list(' '.join(org_transcripts).replace(' ','')))
    wer_diff = 0
    cer_diff = 0
    for i, t in enumerate(org_transcripts):
        #print(t.split(), recognized_transcripts[i], recognized_transcripts[i].split(' '))
        #print(t.replace(' ','').split(), recognized_transcripts[i], list(recognized_transcripts[i].replace(' ','')))
        wer_diff += wer(t.split(), recognized_transcripts[i].split(' '))
        cer_diff += wer(list(t.replace(' ','')), list(recognized_transcripts[i].replace(' ','')))
    print("WER : ", wer_diff/WN)
    print("CER : ", cer_diff/CN)
    return wer_diff/WN, wer_diff, WN, cer_diff/CN, cer_diff, CN

def single_aug_experiment():
    f = open('filelists/vc_test_np.txt', 'r', encoding='utf-8')
    metas = f.readlines()
    f.close()
    org_transcripts = [m.strip().split('|')[1].replace('.', '').replace('!', '').replace('?', '') for m in metas][:64]

    # template = "./data_volume/none_{}_100000/wavs/wavenet-audio-mel_{}.wav"
    template = "./64_test/0_5_aug_es_test_new/{}_0_5_{}000_test/wavs/wavenet-audio-mel_{}.wav"
    aug_policy = ['fm','fw','tla_d','tla_s','tm','tw','value']
    index = [95, 45, 90, 45, 50, 50, 30]

    template = "./0_5_aug_100K_test/{}_0_5_{}000_test/wavs/wavenet-audio-mel_{}.wav"
    aug_policy = ['fm','fw','tla_d','tla_s','tm','tw','value']
    index = [100, 100, 100, 100, 100, 100, 100]
    for k, a in enumerate(aug_policy):
        paths = []
        for i in range(64):
            paths.append(template.format(a,index[k] , i))

        print(a)
        print(WERCER(paths, org_transcripts))

def comb_aug_experiment():
    f = open('filelists/vc_test_np.txt', 'r', encoding='utf-8')
    metas = f.readlines()
    f.close()
    org_transcripts = [m.strip().split('|')[1].replace('.', '').replace('!', '').replace('?', '') for m in metas][:64]

    template = "./policy_100K_test/policy_100K_test/{}_0_5_{}000_test/wavs/wavenet-audio-mel_{}.wav"
    aug_policy = ['tla_d_tm','tla_d_tw','tla_d_value','tla_d_value','tla_s_tm','tla_s_tw','tla_s_tw_tm', 'tla_s_tw_value', 'tla_s_value', 'tw_tm']
    index = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,]

    template = "./policy_es_test/policy_es_test/{}_0_5_{}000_test/wavs/wavenet-audio-mel_{}.wav"
    aug_policy = ['tla_d_tm', 'tla_d_tw', 'tla_d_value', 'tla_s_tm', 'tla_s_tw', 'tla_s_tw_tm', 'tla_s_tw_value', 'tla_s_value', 'tw_tm']
    index = [65, 75, 85, 95, 55, 70, 60, 40, 65]
    for k, a in enumerate(aug_policy):
        paths = []
        for i in range(64):
            paths.append(template.format(a,index[k] , i))

        print(a, index[k])
        print(WERCER(paths, org_transcripts))




def datavolume_experiment():
    f = open('filelists/vc_test_np.txt', 'r', encoding='utf-8')
    metas = f.readlines()
    f.close()
    org_transcripts = [m.strip().split('|')[1].replace('.', '').replace('!', '').replace('?', '') for m in metas][:64]

    template = "./data_volume_100000/none_{}_100000_test/wavs/wavenet-audio-mel_{}.wav"
    datavolumes = ['0_5', '1', '2', '4', '8']

    #template = "./64_test/data_volume_es_test_new/none_{}_test/wavs/wavenet-audio-mel_{}.wav"
    #datavolumes = ['0_5_75000', '1_95000', '2_95000', '4_45000', '8_90000']

    template = "./additional_100k_2/none_{}_100000_test/wavs/wavenet-audio-mel_{}.wav"
    datavolumes = ['0_5', '1']

    template = "./additional_es/none_{}_test/wavs/wavenet-audio-mel_{}.wav"
    datavolumes = ['0_5_75000', '1_95000']

    for d in datavolumes:
        paths = []
        for i in range(64):
            paths.append(template.format(d, i))

        print(d)
        print(WERCER(paths, org_transcripts))


def parameter_experiment():
    f = open('filelists/vc_val.txt', 'r', encoding='utf-8')
    metas = f.readlines()
    f.close()

    num_trys = np.arange(1, 11)
    org_filepath = [m.strip().split('|')[0] for m in metas]
    file_paths = ['gl/{}.wav'.format(x) for x in range(64)]
    org_transcripts = [m.strip().split('|')[1].replace('.', '').replace('!', '').replace('?', '') for m in metas]
    print(WERCER(org_filepath, org_transcripts))
    print(WERCER(file_paths, org_transcripts))

    exps = ['TMC', 'FMC']
    params = [2, 4, 6, 8, 10, 12, 14, 16]
    for exp in exps:
        for param in params:
            for num_try in num_trys:
                print('exp: ', exp, 'param: ', param,'try id: ', num_try)
                file_paths = ['try{}/{}/{}/{}.wav'.format(num_try, exp, param, x) for x in range(64)]
                print(WERCER(file_paths, org_transcripts))

    exps = ["FWL"]
    params = [2, 4, 6, 8, 10, 12, 14, 16]
    for exp in exps:
        for param in params:
            for num_try in num_trys:
                print('exp: ', exp, 'param: ', param, 'try id: ', num_try)
                file_paths = ['try{}/{}/{}/{}.wav'.format(num_try, exp, param, x) for x in range(64)]
                print(WERCER(file_paths, org_transcripts))

    exps = ['TWLR']
    params = [2, 4, 6, 8, 10, 12, 14, 16]
    for exp in exps:
        for param in params:
            for num_try in num_trys:
                print('exp: ', exp, 'param: ', param, 'try id: ', num_try)
                file_paths = ['try{}/{}/{}/{}.wav'.format(num_try, exp, param, x) for x in range(64)]
                print(WERCER(file_paths, org_transcripts))

    exps = ['TLA']
    params = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    for exp in exps:
        for param in params:
            for num_try in num_trys:
                print('exp: ', exp, 'param: ', param, 'try id: ', num_try)
                file_paths = ['try{}/{}/{}/{}.wav'.format(num_try, exp, param, x) for x in range(64)]
                print(WERCER(file_paths, org_transcripts))

    exps = ['TMN']
    params = [(1, 8), (2, 4), (4, 2), (8, 1)]
    for exp in exps:
        for param in params:
            for num_try in num_trys:
                print('exp: ', exp, 'param: ', param, 'try id: ', num_try)
                file_paths = ['try{}/{}/{}/{}.wav'.format(num_try, exp, param, x) for x in range(64)]
                print(WERCER(file_paths, org_transcripts))

    exps = ['FMN']
    params = [(1, 6), (2, 3), (3, 2), (6, 1)]
    for exp in exps:
        for param in params:
            for num_try in num_trys:
                print('exp: ', exp, 'param: ', param, 'try id: ', num_try)
                file_paths = ['try{}/{}/{}/{}.wav'.format(num_try, exp, param, x) for x in range(64)]
                print(WERCER(file_paths, org_transcripts))

    exps = ['VAR']
    params = [2, 4, 8, 16, 32, 64]
    for exp in exps:
        for param in params:
            for num_try in num_trys:
                print('exp: ', exp, 'param: ', param, 'try id: ', num_try)
                file_paths = ['try{}/{}/{}/{}.wav'.format(num_try, exp, param, x) for x in range(64)]
                print(WERCER(file_paths, org_transcripts))

# if __name__ == "__main__":
#     transcripts, confidences = waveToText('speech.wav')
#     print(transcripts)
#     print(WERCER_from_metadata('filelists/vc_val.txt'))
#     print(waveToText('try1/FMC/2/0.wav'))

if __name__ == "__main__":
    #parameter_experiment()
    #datavolume_experiment()
    #single_aug_experiment()
    i = 0
    WE = 0
    CE = 0
    f = open('kor_friday.txt', 'rb')
    while True:
        line = f.readline()
        line = line.decode('utf-8')
        if not line: break
        i += 1
        line = line[:-1]
        path = os.path.join('voice', line)
        line = line[:-3]
        x, _, _, y, _,  _ = WERCER([path], [line])
        WE += x
        CE += y

    avg_wer = WE / i
    avg_cer = CE / i
    print()
    print("#######################################")
    print("Total results of WER and CER")
    print("Average WER : ", avg_wer)
    print("Average CER : ", avg_cer)

   # WERCER(['results_kor_0730_indiv/내_이름은_이동민입니다.15500.wav'], ['내 이름은 이동민 입니다.'])

# if __name__ == "__main__":
#     print(wer("who is there".split(), "is there".split())/3)
#     print(wer("who is there".split(), "".split())/3)
#     print(wer("".split(), "who is there".split())/3)
