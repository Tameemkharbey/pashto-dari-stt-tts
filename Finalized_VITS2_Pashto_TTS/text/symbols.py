'''
Defines the set of symbols used in text input to the model.
Configured for Pashto language.
'''

_pad = '_'

# Pashto punctuation
_punctuation = '!،؛؟۔ '

# Pashto letters (Arabic base + Pashto-specific)
_letters = (
    'آئابتثجحخدذرزسشصضطظعغـفقكلمنهوىي'  # Arabic base
    'ټپځڅچډړږژښکګگڼیۍې'  # Pashto-specific
)

# Pashto diacritics (tanwin, kasra, shadda)
_letters_ipa = 'ًِّ'

# Export all symbols
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Special symbol ids
SPACE_ID = symbols.index(' ')
