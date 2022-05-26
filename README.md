# Korean TTS Server
Implementation of Korean TTS Server based on FastSpeech Pytorch. \
This is based on the fastspeech implementation of [xcmyz](https://github.com/xcmyz/FastSpeech).

## Screen Capture of Web Demo
\Add screen Capture 

## Performance on Korean TTS dataset
\Add Performance on

### Dependencies
- python 3.6
- CUDA 10.0
- pytorch 1.1.0
- numpy 1.16.2
- scipy 1.2.1
- librosa 0.6.3
- inflect 2.1.0
- matplotlib 2.2.2

### Prepare Dataset
1. Download and extract [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).
2. Put LJSpeech dataset in `data`.
3. Run `preprocess.py`. 
> For this implementation, our team utilize Korean Dataset which is available only in Netmarble Company. 

### Get Alignment from Tacotron2
#### Note
In the paper of FastSpeech, authors use pre-trained Transformer-TTS to provide the target of alignment. I didn't have a well-trained Transformer-TTS model so I use Tacotron2 instead.

#### Calculate Alignment during Training (slow)
Change `pre_target = False` in `hparam.py`

#### Calculate Alignment before Training
1. Download the pre-trained Tacotron2 model published by NVIDIA [here](https://drive.google.com/uc?export=download&confirm=XAHL&id=1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA).
2. Put the pre-trained Tacotron2 model in `Tacotron2/pre_trained_model`
3. Run `alignment.py`, it will spend 7 hours training on NVIDIA RTX2080ti.

#### Use Calculated Alignment (quick)
I provide LJSpeech's alignments calculated by Tacotron2 in `alignment_targets.zip`. If you want to use it, just unzip it.

## Run (Support Data Parallel)
### Note
In the turbo mode, a prefetcher prefetches training data and this operation may cost more memory.

### Normal Mode
Run `train.py`.

### Turbo Mode
Run `train_accelerated.py`.

## Test
### Synthesize
Run `test.py -t text_sentence -s checkpoint_step -w 1'

### Results
- The examples of audio are in `results`. The sentence for synthesizing is "I am very happy to see you again.". `results/normal.wav` was synthesized when `alpha = 1.0`, `results/slow.wav` was synthesized when `alpha = 1.5` and `results/quick.wav` was synthesized when `alpha = 0.5`.

## Notes
- The output of LengthRegulator's last linear layer passes through the ReLU activation function in order to remove negative values. It is the outputs of this module. During the inference, the output of LengthRegulator pass through `torch.exp()` and subtract one, as the multiple for expanding encoder output. During the training stage, duration targets add one and pass through `torch.log()` and then calculate loss. For example:
```python
duration_predictor_target = duration_predictor_target + 1
duration_predictor_target = torch.log(duration_predictor_target)

duration_predictor_output = torch.exp(duration_predictor_output)
duration_predictor_output = duration_predictor_output - 1
```


## Reference
- [The Implementation of Tacotron Based on Tensorflow](https://github.com/keithito/tacotron)
- [The Implementation of Transformer Based on Pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
- [The Implementation of Transformer-TTS Based on Pytorch](https://github.com/xcmyz/Transformer-TTS)
- [The Implementation of Tacotron2 Based on Pytorch](https://github.com/NVIDIA/tacotron2)
- [Implementation of Korean Embedding](https://github.com/Yeongtae/tacotron2)
- [Original Implementation of FastSpeech](https://github.com/xcmyz/FastSpeech)
