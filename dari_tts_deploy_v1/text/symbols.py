"""
Dari (Afghan Persian) character symbols for MB-iSTFT-VITS2
Custom character set — no phonemizer dependency
Following friend's Pashto approach (59 chars → we use 52 Dari chars)
"""

_pad = '_'
_punctuation = '،.؟!؛: '
_letters = 'ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیآأإؤئءةً'
_special = '\u200c'  # ZWNJ (zero-width non-joiner, common in Dari)

# Export all symbols
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_special)

# Special symbol ids
SPACE_ID = symbols.index(" ")
