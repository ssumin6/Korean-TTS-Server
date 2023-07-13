from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio
import librosa


def build_from_path(in_dir, out_dir, num_workers=16, tqdm=lambda x: x):
    '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

      Args:
        in_dir: The directory where you have downloaded the LJ Speech dataset
        out_dir: The directory to write the output into
        num_workers: Optional number of worker processes to parallelize across
        tqdm: You can optionally pass tqdm to get a nice progress bar

      Returns:
        A list of tuples describing the training examples. This should be written to train.txt
    '''

    # We use ProcessPoolExecutor to parallelize across processes. This is just an optimization and you
    # can omit it and just call _process_utterance on each input if you want.

    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1

    stft = audio.taco_stft()

    with open(os.path.join(in_dir, 'nam-h_train_filelist.txt'),
              encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, '%s' % parts[0])
            parts2 = line.strip().split('.')
            out_path = os.path.join(out_dir, "%s" % parts2[0] + "mel.npy")
            text = parts[1]
            futures.append(
                executor.submit(
                    partial(_process_utterance, out_dir, out_path, wav_path,
                            text, stft)))

            if index % 100 == 0:
                print("Done %d" % index)
            index = index + 1

    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, out_path, wav_path, text, stft):
    '''Preprocesses a single utterance audio/text pair.

    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
      out_dir: The directory to write the spectrograms into
      index: The numeric index to use in the spectrogram filenames.
      wav_path: Path to the audio file containing the speech input
      text: The text spoken in the input audio file

    Returns:
      A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
    '''

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)
    wav = wav / np.abs(wav).max() * 0.999
    #stft = audio.taco_stft()

    # delete the silence in back of the audio file.
    wav = librosa.effects.trim(wav,
                               top_db=23,
                               frame_length=1024,
                               hop_length=256)[0]

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav,
                                           stft).numpy().astype(np.float32)

    # Write the spectrograms to disk:
    # spectrogram_filename = 'ljspeech-spec-%05d.npy' % index
    parts = out_path.strip().split('/')
    mel_filename = parts[4] + parts[5] + parts[6]
    o_path = os.path.join(parts[0], parts[1], parts[4])

    #    print(o_path)
    #    mel_filename = 'nam_speech-mel-%05d.npy' % index
    #  print(out_path)

    if (not os.path.exists(o_path)):
        os.mkdir(o_path)
    o_path = os.path.join(o_path, parts[5])
    if (not os.path.exists(o_path)):
        os.mkdir(o_path)
    o_path = os.path.join(o_path, parts[6])

    np.save(o_path, mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example:
    # return (spectrogram_filename, mel_filename, n_frames, text)
    return (mel_filename, n_frames, text)
