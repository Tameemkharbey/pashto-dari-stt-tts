"""
Text cleaners for Pashto TTS.
Pashto text is already normalized in Phase 2 preprocessing,
so the cleaner just collapses whitespace.
"""
import re

_whitespace_re = re.compile(r'\s+')


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def pashto_cleaners(text):
    """Simple cleaner for pre-normalized Pashto text."""
    text = collapse_whitespace(text)
    text = text.strip()
    return text
