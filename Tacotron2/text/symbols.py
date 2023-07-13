'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''
from . import cmudict
from .korean import ALL_SYMBOLS_1

_pad = '_'
_eos = '~'
_special = '-'
_punctuation = '!\'(),.:;? '
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
eng_symbols = [
    _pad
] + list(_special) + list(_punctuation) + list(_characters) + _arpabet
kor_symbols = ALL_SYMBOLS_1
#print(len(kor_symbols));
