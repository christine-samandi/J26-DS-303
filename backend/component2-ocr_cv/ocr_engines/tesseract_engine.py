"""Tesseract wrapper returning word tokens with text, confidence and bbox.

Why: every OCR engine (Tesseract now, PaddleOCR next) must return the SAME
token format, so preprocessing, extraction and benchmarking never care
which engine produced the text.
"""
from dataclasses import dataclass, asdict
from typing import List

import pytesseract
from pytesseract import Output


@dataclass
class Token:
    text: str
    confidence: float      # 0-1
    bbox: List[int]        # [x, y, w, h] in pixels of the image passed in
    line_id: str           # block_paragraph_line, groups words into lines

    def to_dict(self):
        return asdict(self)


def run_tesseract(image, lang: str = "sin+tam+eng", psm: int = 6) -> List[Token]:
    """image: numpy array (BGR or gray) or PIL image."""
    data = pytesseract.image_to_data(image, lang=lang, config=f"--psm {psm}", output_type=Output.DICT)
    tokens = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:          # -1 = layout element, not a word
            continue
        tokens.append(Token(
            text=text,
            confidence=round(conf / 100, 4),
            bbox=[int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])],
            line_id=f'{data["block_num"][i]}_{data["par_num"][i]}_{data["line_num"][i]}',
        ))
    return tokens


def tokens_to_text(tokens: List[Token]) -> str:
    """Rebuild plain text line by line (used for CER/WER)."""
    lines, current, out = {}, None, []
    for t in tokens:
        lines.setdefault(t.line_id, []).append(t.text)
    return "\n".join(" ".join(words) for words in lines.values())
