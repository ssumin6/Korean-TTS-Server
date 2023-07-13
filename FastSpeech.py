import torch.nn as nn

from transformer.Models import Encoder, Decoder
from transformer.Layers import Linear, PostNet
from Networks import LengthRegulator
import hparams as hp


class FastSpeech(nn.Module):
    """FastSpeech"""

    def __init__(self):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder()
        self.length_regulator = LengthRegulator()
        self.decoder = Decoder()

        self.mel_linear = Linear(hp.decoder_output_size, hp.num_mels)
        self.postnet = PostNet()

    def forward(
        self,
        src_seq,
        src_pos,
        mel_max_length=None,
        length_target=None,
        alpha=1.0,
    ):
        encoder_output, encoder_mask = self.encoder(src_seq, src_pos)

        if self.training:
            (
                length_regulator_output,
                decoder_pos,
                duration_predictor_output,
            ) = self.length_regulator(
                encoder_output,
                encoder_mask,
                length_target,
                alpha,
                mel_max_length,
            )
            decoder_output = self.decoder(length_regulator_output, decoder_pos)

            mel_output = self.mel_linear(decoder_output)
            mel_output_postnet = self.postnet(mel_output) + mel_output
            """Here, we should conduct mel-spectrogram normalization."""

            return mel_output, mel_output_postnet, duration_predictor_output
        else:
            length_regulator_output, decoder_pos = self.length_regulator(
                encoder_output, encoder_mask, alpha=alpha
            )

            decoder_output = self.decoder(length_regulator_output, decoder_pos)

            mel_output = self.mel_linear(decoder_output)
            mel_output_postnet = self.postnet(mel_output) + mel_output

            return mel_output, mel_output_postnet
