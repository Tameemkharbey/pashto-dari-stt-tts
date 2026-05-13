"""
Dari text cleaners for MB-iSTFT-VITS2
Simple character-level processing — no phonemizer dependency
Following friend's Pashto approach (custom cleaners)
"""
import re
import unicodedata

_whitespace_re = re.compile(r'\s+')

# Dari number words
_ONES = {
    '۰': 'صفر', '۱': 'یک', '۲': 'دو', '۳': 'سه', '۴': 'چهار',
    '۵': 'پنج', '۶': 'شش', '۷': 'هفت', '۸': 'هشت', '۹': 'نه',
    '0': 'صفر', '1': 'یک', '2': 'دو', '3': 'سه', '4': 'چهار',
    '5': 'پنج', '6': 'شش', '7': 'هفت', '8': 'هشت', '9': 'نه',
}
_TEENS = {
    10: 'ده', 11: 'یازده', 12: 'دوازده', 13: 'سیزده', 14: 'چهارده',
    15: 'پانزده', 16: 'شانزده', 17: 'هفده', 18: 'هجده', 19: 'نزده',
}
_TENS = {
    2: 'بیست', 3: 'سی', 4: 'چهل', 5: 'پنجاه',
    6: 'شصت', 7: 'هفتاد', 8: 'هشتاد', 9: 'نود',
}
_HUNDREDS = {
    1: 'صد', 2: 'دوصد', 3: 'سه\u200cصد', 4: 'چهارصد', 5: 'پنج\u200cصد',
    6: 'شش\u200cصد', 7: 'هفت\u200cصد', 8: 'هشت\u200cصد', 9: 'نه\u200cصد',
}

def _num_to_dari(n):
    """Convert integer 0-9999 to Dari words."""
    if n < 0 or n > 9999:
        return str(n)
    if n == 0:
        return 'صفر'
    parts = []
    if n >= 1000:
        t = n // 1000
        parts.append(('هزار' if t == 1 else _ONES.get(str(t), str(t)) + ' هزار'))
        n %= 1000
    if n >= 100:
        parts.append(_HUNDREDS[n // 100])
        n %= 100
    if n >= 10 and n <= 19:
        parts.append(_TEENS[n])
        n = 0
    elif n >= 20:
        parts.append(_TENS[n // 10])
        n %= 10
    if n >= 1:
        parts.append(_ONES[str(n)])
    return ' و '.join(parts)

def _replace_numbers(text):
    """Replace digit sequences with Dari words."""
    def _repl(m):
        s = m.group()
        # Normalize Persian digits to ASCII for parsing
        normalized = s.translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789'))
        try:
            return _num_to_dari(int(normalized))
        except (ValueError, KeyError):
            return s
    return re.sub(r'[۰-۹0-9]+', _repl, text)

# Dari character normalization mappings
_CHAR_MAP = {
    '\u0643': '\u06a9',  # Arabic Kaf → Farsi Kaf
    '\u064a': '\u06cc',  # Arabic Yeh → Farsi Yeh
    '\u0649': '\u06cc',  # Alef Maksura → Farsi Yeh
    '\u0640': '',         # Tatweel (kashida) → remove
}

# Diacritics to strip (except tanwin which is in our vocab)
_DIACRITICS = re.compile(r'[\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u0659\u065a\u065b\u065c\u065d\u065e\u065f]')

# Private Use Area characters
_PUA = re.compile(r'[\uf000-\uf8ff]')


def dari_cleaners(text):
    """Clean and normalize Dari text for TTS training."""
    # Normalize unicode
    text = unicodedata.normalize('NFC', text)

    # Convert numbers to Dari words
    text = _replace_numbers(text)

    # Apply character mappings
    for src, dst in _CHAR_MAP.items():
        text = text.replace(src, dst)

    # Remove diacritics (except tanwin ً which is in vocab)
    text = _DIACRITICS.sub('', text)

    # Remove PUA characters
    text = _PUA.sub('', text)

    # Normalize whitespace
    text = _whitespace_re.sub(' ', text)

    # Strip
    text = text.strip()

    return text
