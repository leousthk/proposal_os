#!/usr/bin/env python3
"""Same Words, Different Voice: single-file code for the technical appendix.

AI-enabled measurement of text-voice emotional (in)congruence in earnings conference calls. This file
merges the repository's modules, its verification suite, and its command-line scripts. Each section
below corresponds to one file of the repository and keeps its names, so the module.function references
in the appendix (e.g. align.align_sentences) work here as well.

Requirements
    Core and tests: Python 3.10+, numpy, scipy, pandas, pymupdf.
    Pipeline:       torch, transformers, faster-whisper, soundfile, huggingface_hub, matplotlib.
                    Tested with Python 3.14.7, torch 2.11.0+cu128, transformers 5.21.0 and
                    faster-whisper 1.2.1 on one NVIDIA RTX 5070 (12 GB) and, more slowly, on CPU.
    Models (open weights, downloaded on first use): Whisper large-v3-turbo (faster-whisper conversion),
    waiv/FinVoc2Vec, audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim, yiyanghkust/finbert-tone,
    Qwen/Qwen3-4B-Instruct-2507.

Transcripts
    Either an S&P Global Market Intelligence transcript PDF, or a turn table (.csv, .tsv, .json, .jsonl)
    with one row per speaker turn and the columns speaker and text (see parse_turn_table). The turn table
    is the route for openly licensed corpora such as Earnings-21 and Earnings-22, so that the measurement
    can be reproduced without licensed transcripts; examples/make_turn_table.py converts them.

Usage
    python voice_text_congruence.py --test                                 # verification suite (no GPU)
    python voice_text_congruence.py --call CALL.mp3 CALL.pdf --out DIR     # one call
    python voice_text_congruence.py --sample ROOT --out DIR [--limit N] [--match Earnings_Call]
                                                                           # every call under ROOT
    python voice_text_congruence.py --manifest calls.csv --out DIR         # every call listed in a CSV
    python voice_text_congruence.py --llm --out DIR                        # second textual rater (Qwen3)
    python voice_text_congruence.py --report --out DIR                     # tables, figures, summary.json
"""
from __future__ import annotations

import argparse
import difflib
import glob
import gzip
import itertools
import json
import os
import platform
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pandas as pd


# ==================================================================================================
# 1. sentences  (repository: src/vtc/sentences.py)
# ==================================================================================================
# Sentence segmentation of speaker turns and token normalisation shared by text and ASR.
#
# split_sentences   rule-based splitter: a sentence ends at . ! ? (optionally followed by a closing quote or
#                   bracket) when the next character starts a new sentence, except after common
#                   abbreviations (Mr., Inc., U.S., Q1., No.) and inside numbers (3.5%).
# normalize_tokens  lower-case word tokens used to match transcript words to ASR words: % -> "percent",
#                   & -> "and", currency symbols and thousands separators dropped, then split on anything
#                   that is not a letter, digit, or apostrophe. Both sides are normalised identically, so
#                   "$189.9 million" and "189.9 million" give the same tokens.

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "inc", "co", "corp", "ltd", "llc", "plc", "no", "vs",
    "etc", "e.g", "i.e", "u.s", "u.k", "u.s.a", "approx", "est", "dept", "fig", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "mt", "ft", "vol", "op", "cf",
}
BRACKETED = re.compile(r"\[[^\]]*\]|\((?:operator instructions|inaudible|indiscernible|technical difficulty)[^)]*\)",
                       re.I)                           # [Operator Instructions], (Operator instructions), [indiscernible]
SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*\s+")


def clean_text(text: str) -> str:
    return " ".join(BRACKETED.sub(" ", text).split())


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    out, start = [], 0
    for m in SENTENCE_END.finditer(text):
        end = m.end()
        before = text[start:m.start() + 1]
        nxt = text[end:end + 1]
        word = re.search(r"([A-Za-z][A-Za-z.]*)\.$", before)
        if word and word.group(1).lower().rstrip(".") in ABBREVIATIONS:
            continue
        if re.search(r"\b[A-Z]\.$", before) and nxt[:1].isupper():   # initials: "John Q. Public"
            continue
        if nxt and not (nxt.isupper() or nxt.isdigit() or nxt in "\"'([$"):
            continue
        out.append(text[start:end].strip())
        start = end
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return [s for s in out if re.search(r"[A-Za-z0-9]", s)]


def normalize_tokens(text: str) -> list[str]:
    t = text.lower().replace("%", " percent ").replace("&", " and ")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)            # 1,000 -> 1000
    t = re.sub(r"[$€£¥]", " ", t)
    t = t.replace("’", "'")
    return [w.strip("'") for w in re.split(r"[^a-z0-9']+", t) if w.strip("'")]


# ==================================================================================================
# 2. transcripts  (repository: src/vtc/transcripts.py)
# ==================================================================================================
# Parse S&P Global Market Intelligence (Capital IQ) earnings-call transcript PDFs into speaker turns.
#
# The transcripts use a fixed typography, and the parser relies on it rather than on text patterns:
#
#   section header      bold, >= 16 pt       "Call Participants", "Presentation", "Question and Answer"
#   participant group   bold, grey, upper    "EXECUTIVES", "ANALYSTS" (participants page only)
#   speaker name        bold, 10 pt          "Jane Roe", "Operator", "Unknown Analyst"
#   title/affiliation   italic, 10 pt        one or two lines directly under the name
#   spoken text         regular, 10 pt
#   running header      bold, grey, contains "EARNINGS CALL" (kept only as call metadata)
#   page furniture      < 9.5 pt: copyright line, page number, disclaimer (skipped)
# The rules use size, weight, and colour rather than font names, so they cover the current template (Arial,
# 24 pt headers), the older S&P Global template (Verdana, 24 pt), and S&P Capital IQ reprints (Verdana, 18 pt).
#
# A speaker name set in a regular fallback font (names with accented letters) is recognised from its position
# directly above an italic title, or from the participant list. When a transcript has no "Question and Answer"
# header, the session starts at the first operator turn that announces a question after the executives spoke.
# Text that follows a section header before any speaker name (e.g. "[Operator Instructions] Our first
# question comes from ...") is attributed to the Operator when it reads like an operator line, otherwise
# to the previous speaker.

SECTION_NAMES = {"presentation": "presentation", "question and answer": "qa", "call participants": "participants"}
OPERATOR_CUE = re.compile(r"operator instructions|question (comes|will come|is coming) from|next question|"
                          r"our (first|last|final) question", re.I)


@dataclass
class Turn:
    section: str                 # "presentation" or "qa"
    speaker: str
    title: str = ""
    group: str = ""              # "executives", "analysts", or "" (from the participants page)
    role: str = ""               # CEO, CFO, IR, other_executive, analyst, operator, other
    text: str = ""
    order: int = 0


@dataclass
class Transcript:
    path: str
    header: str = ""                                  # e.g. "ACME CORPORATION FQ1 2022 EARNINGS CALL ..."
    participants: dict = field(default_factory=dict)  # name -> (group, title)
    turns: list = field(default_factory=list)


def classify_role(name: str, title: str, group: str) -> str:
    n, t = name.lower(), (title or "").lower()
    if n == "operator":
        return "operator"
    if group == "analysts" or "analyst" in n or "research division" in t:
        return "analyst"
    if re.search(r"\bco-?ceo\b|\bceo\b|chief executive", t):
        return "CEO"
    if re.search(r"\bcfo\b|chief financial", t):
        return "CFO"
    if "investor relations" in t or re.search(r"\bir\b", t):
        return "IR"
    if group == "executives" or "executive" in n or "company representative" in n:
        return "other_executive"
    return "other"


def line_kind(font: str, size: float, flags: int, color: int, text: str) -> str | None:
    if size < 9.5:
        return None                                   # copyright line, page number, disclaimer
    bold = bool(flags & 16) or "Bold" in font
    italic = bool(flags & 2) or "Italic" in font or "Oblique" in font
    if bold and color != 0 and "EARNINGS CALL" in text.upper():
        return "running_header"                       # grey page header: company, quarter, date
    if size >= 16 and bold:
        return "header"                               # 24 pt (S&P Global) or 18 pt (S&P Capital IQ reprints)
    if bold:
        return "group" if (color != 0 and text.isupper()) else "speaker"
    if italic:
        return "title"
    return "body"


def read_lines(path: str):
    """Typed text lines of the PDF in reading order: [(kind, text), ...]."""
    import pymupdf

    out = []
    with pymupdf.open(path) as doc:
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans = [s for s in line["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = " ".join("".join(s["text"] for s in line["spans"]).split())
                    s0 = spans[0]
                    kind = line_kind(s0["font"], s0["size"], s0["flags"], s0["color"], text)
                    if kind is not None:
                        out.append((kind, text))
    return out


NAME_PARTICLES = {"de", "da", "del", "della", "di", "du", "la", "le", "van", "von", "der", "den", "y", "bin", "al"}


def looks_like_name(text: str) -> bool:
    """A short run of capitalised words without sentence punctuation ("Marta Ibáñez-Ruiz", "John Q. Public")."""
    toks = text.split()
    if not 1 <= len(toks) <= 6 or text[-1] in ",;:?!" or (text.endswith(".") and not re.search(r"\b[JS]r\.$", text)):
        return False
    return all(t[0].isupper() or t.lower() in NAME_PARTICLES for t in toks)


def _join(text: str, line: str) -> str:
    """Append a wrapped line; keep hyphenated words that were split across lines ("pre-" + "pandemic")."""
    if not text:
        return line
    if text.endswith("-") and line[:1].islower():
        return text + line
    return text + " " + line


def parse_lines(lines, path: str = "") -> Transcript:
    tr = Transcript(path=path)
    mode, group, member = "front", "", None
    cur: Turn | None = None
    title_lines: list[str] = []

    def close():
        nonlocal cur, title_lines
        if cur is not None:
            if title_lines:
                cur.title = " ".join(title_lines)
            if cur.text.strip():
                cur.text = cur.text.strip()
                cur.order = len(tr.turns)
                tr.turns.append(cur)
        cur, title_lines = None, []

    # Names with accented letters can be set in a regular fallback font (ArialUnicodeMS) instead of bold; such a
    # line still sits directly above the speaker's italic title, so the layout identifies it as a speaker name.
    lines = list(lines)
    lines = [("speaker", text) if kind == "body" and i + 1 < len(lines) and lines[i + 1][0] == "title"
             and looks_like_name(text) else (kind, text) for i, (kind, text) in enumerate(lines)]
    for kind, text in lines:
        if kind == "body" and mode in ("presentation", "qa") and text.strip() in tr.participants:
            kind = "speaker"                          # a listed participant's name line without a title
        if kind == "running_header":
            tr.header = tr.header or text
            continue
        if kind == "header":
            close()
            mode, member = SECTION_NAMES.get(text.strip().lower(), "front"), None
            continue
        if mode == "participants":
            if kind == "group":
                group = text.strip().lower()
            elif kind == "speaker":
                member = text.strip()
                tr.participants[member] = (group, "")
            elif kind == "title" and member:
                g, t = tr.participants[member]
                tr.participants[member] = (g, (t + " " + text).strip())
            continue
        if mode not in ("presentation", "qa"):
            continue
        if kind == "speaker":
            close()
            name = text.strip()
            g, t = tr.participants.get(name, ("", ""))
            cur = Turn(section=mode, speaker=name, title=t, group=g)
        elif kind == "title":
            if cur is not None and not cur.text:
                title_lines.append(text)
        elif kind == "body":
            if cur is None:
                prev = tr.turns[-1] if tr.turns else None
                if prev is None or OPERATOR_CUE.search(text):
                    cur = Turn(section=mode, speaker="Operator")
                else:
                    cur = Turn(section=mode, speaker=prev.speaker, title=prev.title, group=prev.group)
            cur.text = _join(cur.text, text)
    close()
    infer_qa(tr)
    for t in tr.turns:
        g, title = tr.participants.get(t.speaker, ("", ""))
        t.title = t.title or title
        t.group = t.group or g
        t.role = classify_role(t.speaker, t.title, t.group)
    return tr


def infer_qa(tr: Transcript) -> None:
    """Some transcripts lack the "Question and Answer" header: the session then starts with the first
    operator turn that announces a question after the executives have spoken."""
    if tr.turns and not any(t.section == "qa" for t in tr.turns):
        spoke = False
        for i, t in enumerate(tr.turns):
            if t.speaker.lower() != "operator":
                spoke = True
            elif spoke and OPERATOR_CUE.search(t.text):
                for u in tr.turns[i:]:
                    u.section = "qa"
                break


def parse_transcript_pdf(path: str) -> Transcript:
    return parse_lines(read_lines(path), path)


# Transcripts from other sources: one row per speaker turn, in speaking order. This is the route for
# openly licensed corpora (Earnings-21, Earnings-22) and for transcripts of other vendors, so that the
# instrument can be reproduced without the licensed PDFs it was developed on.

TABLE_EXTENSIONS = (".csv", ".tsv", ".json", ".jsonl")
TRANSCRIPT_EXTENSIONS = (".pdf",) + TABLE_EXTENSIONS
SECTION_ALIASES = {"presentation": "presentation", "prepared remarks": "presentation", "remarks": "presentation",
                   "management discussion": "presentation", "qa": "qa", "q&a": "qa", "q and a": "qa",
                   "question and answer": "qa", "questions and answers": "qa", "question": "qa", "answer": "qa"}
ROLES = ("CEO", "CFO", "IR", "other_executive", "analyst", "operator", "other")
EXECUTIVE_TITLE = re.compile(r"\bchief\b|\bofficer\b|president|\bchair|\bdirector\b|treasurer|controller|counsel|"
                             r"secretary|\bhead of\b|founder|investor relations|\badvis[eo]r\b|\bprincipal\b|"
                             r"\b[cs]?e?vp\b|\b(ceo|cfo|coo|cto)\b", re.I)
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "cfa", "phd"}


def _cell(row: dict, key: str) -> str:
    v = row.get(key)
    return "" if v is None or (isinstance(v, float) and v != v) else str(v).strip()


def parse_turn_table(path: str) -> Transcript:
    """Speaker turns from a table (.csv, .tsv, .json records, or .jsonl), one row per turn in speaking order.

    Required columns: speaker, text. Optional: section (presentation or qa; "Prepared remarks", "Q&A",
    "Question and Answer" and similar labels are recognised), title, group (executives or analysts), role
    (CEO, CFO, IR, other_executive, analyst, operator, other), order (sort key). Missing roles follow from
    title and group as for PDF transcripts (classify_role); without a group, an executive title
    (EXECUTIVE_TITLE) marks an executive and any other title is read as an analyst's affiliation; a speaker
    without a title counts as an executive when speaking in the prepared remarks and as an analyst when the
    operator announces a question from that speaker (fallback_group). Without a section column the
    question-and-answer session is inferred from the operator's first question cue (infer_qa)."""
    import pandas as pd

    ext = os.path.splitext(path)[1].lower()
    if ext in (".json", ".jsonl"):
        df = pd.read_json(path, lines=ext == ".jsonl", dtype=False)
    else:
        df = pd.read_csv(path, sep="\t" if ext == ".tsv" else ",", dtype=str, keep_default_na=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = {"speaker", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"turn table {os.path.basename(path)} lacks the column(s) {sorted(missing)}")
    if "order" in df.columns:
        df = df.assign(_o=pd.to_numeric(df["order"], errors="coerce")).sort_values("_o", kind="stable")
    tr = Transcript(path=path)
    roles = {r.lower(): r for r in ROLES}
    for r in df.to_dict("records"):
        text = " ".join(_cell(r, "text").split())
        if not text:
            continue
        sec = SECTION_ALIASES.get(_cell(r, "section").lower(), "presentation")
        t = Turn(section=sec, speaker=_cell(r, "speaker") or "Unknown", title=_cell(r, "title"),
                 group=_cell(r, "group").lower(), text=text, order=len(tr.turns))
        t.role = roles.get(_cell(r, "role").lower().replace(" ", "_"), "")
        tr.turns.append(t)
    if "section" not in df.columns:
        infer_qa(tr)
    fallback_group(tr)
    for t in tr.turns:
        t.role = t.role or classify_role(t.speaker, t.title, t.group)
    return tr


def fallback_group(tr: Transcript) -> None:
    """Executives or analysts for turns that come without role and group (see parse_turn_table)."""
    cues = " ".join(t.text for t in tr.turns
                    if t.speaker.lower() == "operator" and OPERATOR_CUE.search(t.text)).lower()
    presenters = {t.speaker for t in tr.turns if t.section == "presentation" and t.speaker.lower() != "operator"}
    for t in tr.turns:
        if t.role or t.group or t.speaker.lower() == "operator":
            continue
        if t.title:
            t.group = "executives" if EXECUTIVE_TITLE.search(t.title) else "analysts"
            continue
        names = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]+", t.speaker) if w.lower() not in NAME_SUFFIXES]
        if t.speaker in presenters:
            t.group = "executives"
        elif names and len(names[-1]) >= 3 and re.search(rf"\b{re.escape(names[-1].lower())}\b", cues):
            t.group = "analysts"


def read_manifest(path: str) -> list:
    """Calls listed in a CSV with the columns audio and transcript, one row per call.

    Paths may be absolute or relative to the manifest. Any further columns are ignored here, so a manifest
    that also carries identifiers travels with the data without changing the pipeline. A manifest is the
    route for corpora that are not laid out as one folder per company, such as Earnings-21 and Earnings-22."""
    import pandas as pd

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = {"audio", "transcript"} - set(df.columns)
    if missing:
        raise ValueError(f"manifest {os.path.basename(path)} lacks the column(s) {sorted(missing)}")
    base = os.path.dirname(os.path.abspath(path))
    calls = []
    for r in df.to_dict("records"):
        a, t = _cell(r, "audio"), _cell(r, "transcript")
        if not a or not t:
            continue
        calls.append((a if os.path.isabs(a) else os.path.normpath(os.path.join(base, a)),
                      t if os.path.isabs(t) else os.path.normpath(os.path.join(base, t))))
    return calls


def load_transcript(path: str) -> Transcript:
    """Speaker turns from an S&P Global Market Intelligence transcript PDF or from a turn table."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return parse_transcript_pdf(path)
    if ext in TABLE_EXTENSIONS:
        return parse_turn_table(path)
    raise ValueError(f"unsupported transcript format: {os.path.basename(path)}")


# ==================================================================================================
# 3. audio  (repository: src/vtc/audio.py)
# ==================================================================================================
# Audio input: decode a call recording to 16 kHz mono float32 and cut sentence clips.
#
# MP3 is decoded with libsndfile (soundfile >= 0.12 bundles an MP3-capable libsndfile); PyAV is used as a
# fallback. Resampling uses a polyphase filter (scipy.signal.resample_poly) with the exact rational ratio.

SR = 16000


def default_device() -> str:
    """"cuda" when a CUDA device is actually usable, else "cpu".

    The verification suite never needs a GPU. The model passes run on CPU as well, roughly an order of
    magnitude more slowly, so the default adapts to the machine instead of failing on it."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:                                                   # noqa: BLE001
        return "cpu"


def _resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    if sr_in == sr_out:
        return x
    from scipy.signal import resample_poly

    f = Fraction(sr_out, sr_in).limit_denominator(1000)
    return resample_poly(x, f.numerator, f.denominator).astype(np.float32)


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    try:
        import soundfile as sf

        data, sr_in = sf.read(path, dtype="float32", always_2d=True)
        x = data.mean(axis=1)
    except Exception:                                  # noqa: BLE001  fall back to PyAV (bundled FFmpeg)
        import av

        with av.open(path) as container:
            stream = container.streams.audio[0]
            sr_in = stream.rate
            frames = [f.to_ndarray().astype(np.float32) for f in container.decode(stream)]
        data = np.concatenate([f if f.ndim == 2 else f[None, :] for f in frames], axis=1)
        x = data.mean(axis=0)
        if np.abs(x).max() > 1.5:                      # integer PCM
            x = x / 32768.0
    return _resample(np.ascontiguousarray(x, dtype=np.float32), int(sr_in), sr)


def clip(audio: np.ndarray, start: float, end: float, sr: int = SR, pad: float = 0.05) -> np.ndarray:
    a = max(0, int((start - pad) * sr))
    b = min(len(audio), int((end + pad) * sr))
    return audio[a:b]


# ==================================================================================================
# 4. asr  (repository: src/vtc/asr.py)
# ==================================================================================================
# Automatic speech recognition with word time stamps (Whisper via faster-whisper / CTranslate2).
#
# Whisper (Radford et al. 2023) is an encoder-decoder transformer trained on 680,000 hours of weakly
# supervised speech. faster-whisper runs it with CTranslate2; BatchedInferencePipeline first splits the
# recording into speech chunks of at most 30 s with the Silero voice-activity detector and then decodes
# the chunks in batches. Word time stamps come from dynamic time warping over the decoder's cross-attention
# weights on the alignment heads. The ASR text is used only to place the vendor transcript in time
# (align.py); every analysed word comes from the vendor transcript.
#
# Results are cached per call (gzip JSON), so a re-run never repeats recognition.

DEFAULT_MODEL = "large-v3-turbo"
WHISPER_REPOS = {"large-v3-turbo": ("mobiuslabsgmbh/faster-whisper-large-v3-turbo",
                                    "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf")}


def load_whisper(name: str = DEFAULT_MODEL, device: str | None = None, compute_type: str | None = None):
    device = device or default_device()
    compute_type = compute_type or ("float16" if device == "cuda" else "int8")
    if device == "cuda" and os.name == "nt":
        # CTranslate2 needs cuBLAS/cuDNN; on Windows reuse the CUDA libraries shipped with PyTorch
        import torch

        os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    if name in WHISPER_REPOS:                         # pinned conversion of the published checkpoint
        from huggingface_hub import snapshot_download

        name = snapshot_download(*WHISPER_REPOS[name][:1], revision=WHISPER_REPOS[name][1])
    model = WhisperModel(name, device=device, compute_type=compute_type)
    return BatchedInferencePipeline(model=model)


def transcribe_words(pipe, audio, batch_size: int = 16, language: str = "en", beam_size: int = 1) -> dict:
    # Greedy decoding suffices: the recognised words only anchor the vendor transcript in time.
    segments, info = pipe.transcribe(audio, language=language, batch_size=batch_size, beam_size=beam_size,
                                     word_timestamps=True, condition_on_previous_text=False)
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"w": w.word.strip(), "s": round(float(w.start), 3), "e": round(float(w.end), 3),
                          "p": round(float(w.probability), 3)})
    return {"language": info.language, "duration": round(float(info.duration), 2), "words": words}


def cached_transcription(cache_path: str, pipe, audio, **kw) -> dict:
    if os.path.exists(cache_path):
        with gzip.open(cache_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    out = transcribe_words(pipe, audio, **kw)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with gzip.open(cache_path, "wt", encoding="utf-8") as f:
        json.dump(out, f)
    return out


# ==================================================================================================
# 5. align  (repository: src/vtc/align.py)
# ==================================================================================================
# Place every transcript sentence in time by matching transcript words to time-stamped ASR words.
#
# The vendor transcript (who said what, in which section) and the ASR output (when each word was spoken)
# are two token sequences of the same call. A global sequence alignment (difflib.SequenceMatcher, a
# refinement of Ratcliff and Obershelp's "gestalt pattern matching": find the longest common block, then
# recurse on the unmatched left and right parts) maps transcript tokens to ASR words monotonically.
#
# For each sentence k with tokens i in I_k and matched ASR words m(i):
#     start_k = time of the first matched word, moved back over the sentence's unmatched leading tokens
#     end_k   = time of the last matched word, moved forward over its unmatched trailing tokens
#     coverage_k = |{i in I_k : m(i) exists}| / |I_k|
# A sentence is "timed" when coverage >= 0.6 and its speaking rate lies between 1 and 7 tokens per second.

MIN_COVERAGE = 0.6
RATE_RANGE = (1.0, 7.0)


def asr_tokens(words: list[dict]):
    toks, owner = [], []
    for j, w in enumerate(words):
        for t in normalize_tokens(w["w"]):
            toks.append(t)
            owner.append(j)
    return toks, owner


def align_sentences(sent_tokens: list[list[str]], words: list[dict]) -> list[dict]:
    t_tok, t_sid = [], []
    for k, toks in enumerate(sent_tokens):
        t_tok += toks
        t_sid += [k] * len(toks)
    a_tok, a_owner = asr_tokens(words)
    match = [-1] * len(t_tok)
    sm = difflib.SequenceMatcher(None, t_tok, a_tok, autojunk=False)
    for a, b, size in sm.get_matching_blocks():
        for d in range(size):
            match[a + d] = a_owner[b + d]

    out, pos, prev_last = [], 0, -1
    for k, toks in enumerate(sent_tokens):
        idx = list(range(pos, pos + len(toks)))
        pos += len(toks)
        m = [match[i] for i in idx]
        hit = [j for j in m if j >= 0]
        rec = {"n_tokens": len(toks), "n_matched": len(hit), "coverage": len(hit) / max(len(toks), 1),
               "start": None, "end": None, "w_first": None, "w_last": None}
        if hit:
            lead = next(i for i, j in enumerate(m) if j >= 0)                  # unmatched leading tokens
            trail = next(i for i, j in enumerate(reversed(m)) if j >= 0)       # unmatched trailing tokens
            first = max(prev_last + 1, min(hit) - lead)
            nxt = next((match[i] for i in range(pos, len(match)) if match[i] >= 0), len(words))
            last = min(nxt - 1, max(hit) + trail, len(words) - 1)
            first = min(first, min(hit))
            last = max(last, max(hit))
            rec.update(start=words[first]["s"], end=words[last]["e"], w_first=first, w_last=last)
            prev_last = max(prev_last, last)
        dur = (rec["end"] - rec["start"]) if hit else 0.0
        rate = len(toks) / dur if dur > 0 else 0.0
        rec["duration"] = round(dur, 3)
        rec["tokens_per_sec"] = round(rate, 3)
        rec["timed"] = bool(hit) and rec["coverage"] >= MIN_COVERAGE and RATE_RANGE[0] <= rate <= RATE_RANGE[1]
        rec["asr_text"] = " ".join(w["w"] for w in words[rec["w_first"]:rec["w_last"] + 1]) if hit else ""
        out.append(rec)
    return out


# ==================================================================================================
# 6. congruence  (repository: src/vtc/congruence.py)
# ==================================================================================================
# Text-voice emotional (in)congruence: the nine-cell typology and its statistics.
#
# Each sentence s has a textual tone T_s and a vocal tone V_s in {positive, neutral, negative} (the argmax
# of the model probabilities) and continuous net tones
#     text_net_s  = P_T(positive) - P_T(negative),        voice_net_s = P_V(positive) - P_V(negative),
#     gap_s       = voice_net_s - text_net_s              (> 0: the voice is more positive than the words).
# With sign(positive) = 1, sign(neutral) = 0, sign(negative) = -1, the direction of a sentence is
#     d_s = sign(V_s) - sign(T_s) in {-2, ..., 2}:
#     d = 0 aligned; d > 0 vocal lift (voice more positive than words); d < 0 vocal drag;
#     |d| = 2 polarity reversal (positive words with a negative voice, or the reverse).
#
# Association between the channels (3 x 3 table n_ij, rows = text, columns = voice):
#     E_ij = n_i. n_.j / n,   chi2 = sum (n_ij - E_ij)^2 / E_ij,   Cramer's V = sqrt(chi2 / (n (k - 1)))
#     adjusted residual r_ij = (n_ij - E_ij) / sqrt(E_ij (1 - n_i./n) (1 - n_.j/n))    (Agresti 2013)
#     Cohen's kappa = (p_o - p_e) / (1 - p_e),  p_o = sum_i n_ii / n,  p_e = sum_i n_i. n_.i / n^2
# r_ij > 1.96 marks a combination that occurs more often than if words and voice were independent.
# Uncertainty is computed by resampling whole calls (cluster bootstrap), because sentences of the same
# call share speaker, recording, and news.

LABELS = ("positive", "neutral", "negative")
SIGN = {"positive": 1, "neutral": 0, "negative": -1}

# Names state the communicative situation each cell makes observable: authenticity and emotional labor
# (Grandey 2000), dramatization and impression management (Goffman 1959), nonverbal leakage (Ekman and Friesen 1969).
CELLS = {
    ("positive", "positive"): ("C1", "Authentic optimism", "upbeat words, upbeat voice"),
    ("positive", "neutral"): ("C2", "Scripted optimism", "upbeat words, neutral voice"),
    ("positive", "negative"): ("C3", "Leaked doubt", "upbeat words, negative voice"),
    ("neutral", "positive"): ("C4", "Dramatized facts", "neutral words, upbeat voice"),
    ("neutral", "neutral"): ("C5", "Factual disclosure", "neutral words, neutral voice"),
    ("neutral", "negative"): ("C6", "Unspoken concern", "neutral words, negative voice"),
    ("negative", "positive"): ("C7", "Reassuring spin", "negative words, upbeat voice"),
    ("negative", "neutral"): ("C8", "Stoic bad news", "negative words, neutral voice"),
    ("negative", "negative"): ("C9", "Candid concern", "negative words, negative voice"),
}


def argmax_label(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [f"{prefix}{lab}" for lab in LABELS]
    return df[cols].idxmax(axis=1).str.replace(prefix, "", regex=False)


def add_congruence(df: pd.DataFrame, text: str = "tx_", voice: str = "fv_") -> pd.DataFrame:
    d = df.copy()
    d["text_label"] = argmax_label(d, text)
    d["voice_label"] = argmax_label(d, voice)
    d["text_net"] = d[f"{text}positive"] - d[f"{text}negative"]
    d["voice_net"] = d[f"{voice}positive"] - d[f"{voice}negative"]
    d["gap"] = d["voice_net"] - d["text_net"]
    d["direction"] = d["voice_label"].map(SIGN) - d["text_label"].map(SIGN)
    d["aligned"] = d["direction"] == 0
    d["reversal"] = d["direction"].abs() == 2
    d["cell"] = [CELLS[(t, v)][0] for t, v in zip(d["text_label"], d["voice_label"])]
    d["cell_name"] = [CELLS[(t, v)][1] for t, v in zip(d["text_label"], d["voice_label"])]
    return d


def crosstab3(d: pd.DataFrame, rows: str = "text_label", cols: str = "voice_label") -> pd.DataFrame:
    return pd.crosstab(d[rows], d[cols]).reindex(index=LABELS, columns=LABELS, fill_value=0)


def independence(table: pd.DataFrame) -> dict:
    n = table.to_numpy(float)
    tot = n.sum()
    r, c = n.sum(1, keepdims=True), n.sum(0, keepdims=True)
    e = r @ c / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((n - e) ** 2 / e)
        adj = (n - e) / np.sqrt(e * (1 - r / tot) * (1 - c / tot))
    k = min(n.shape) - 1
    from scipy.stats import chi2 as chi2_dist

    dof = (n.shape[0] - 1) * (n.shape[1] - 1)
    po = np.trace(n) / tot
    pe = float((r.ravel() * c.ravel()).sum() / tot ** 2)
    return {"n": int(tot), "chi2": float(chi2), "dof": dof, "p": float(chi2_dist.sf(chi2, dof)),
            "cramers_v": float(np.sqrt(chi2 / (tot * k))), "kappa": float((po - pe) / (1 - pe)),
            "agreement": float(po),
            "adj_residuals": pd.DataFrame(adj, index=table.index, columns=table.columns)}


def cluster_bootstrap(values: pd.Series, clusters: pd.Series, reps: int = 2000, seed: int = 0):
    """Percentile 95% interval of a sample mean when whole clusters (calls) are resampled.

    With per-cluster sums S_c and counts N_c, each replicate draws clusters c_1..c_G with replacement
    and computes sum_g S_{c_g} / sum_g N_{c_g}.
    """
    g = pd.DataFrame({"v": values.astype(float).to_numpy(), "c": clusters.to_numpy()}).groupby("c")["v"]
    s, n = g.sum().to_numpy(), g.count().to_numpy()
    if len(s) < 2:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(s), size=(reps, len(s)))
    est = s[idx].sum(1) / n[idx].sum(1)
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def icc1(values, groups) -> float:
    """Share of variance lying between groups: one-way ANOVA estimator of ICC(1) for unequal group sizes,
        ICC = (MSB - MSW) / (MSB + (n0 - 1) MSW),   n0 = (N - sum_g n_g^2 / N) / (G - 1)."""
    d = pd.DataFrame({"v": np.asarray(values, float), "g": np.asarray(groups)}).dropna()
    st = d.groupby("g")["v"].agg(["size", "mean", "var"])
    k, n = len(st), len(d)
    if k < 2 or n <= k:
        return float("nan")
    msb = (st["size"] * (st["mean"] - d["v"].mean()) ** 2).sum() / (k - 1)
    msw = (st["var"].fillna(0) * (st["size"] - 1)).sum() / (n - k)
    n0 = (n - (st["size"] ** 2).sum() / n) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def eta2(values, groups) -> float:
    """Share of the variance of a score accounted for by a grouping (between-group sum of squares / total)."""
    d = pd.DataFrame({"v": np.asarray(values, float), "g": np.asarray(groups)}).dropna()
    dev = d["v"] - d["v"].mean()
    return float(((d.groupby("g")["v"].transform("mean") - d["v"].mean()) ** 2).sum() / (dev ** 2).sum())


def speaker_relative(d: pd.DataFrame, by=("call_id", "speaker"), z: float = 0.5) -> pd.Series:
    """Vocal tone relative to the speaker's own mean in the call: + / 0 / - at |z| > 0.5 SD."""
    g = d.groupby(list(by))["voice_net"]
    zz = (d["voice_net"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return pd.Series(np.select([zz > z, zz < -z], ["positive", "negative"], "neutral"), index=d.index)


def same_text_dispersion(d: pd.DataFrame, min_count: int = 5) -> tuple[pd.DataFrame, dict]:
    """Identical sentences said many times: how much does their vocal tone vary although the words do not?"""
    key = d["text"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.split().str.join(" ")
    d = d.assign(_key=key)
    counts = d["_key"].value_counts()
    rep = d[d["_key"].isin(counts[counts >= min_count].index)]
    if rep.empty:
        return pd.DataFrame(), {}
    tab = rep.groupby("_key").agg(text=("text", "first"), n=("voice_net", "size"), voice_net_mean=("voice_net", "mean"),
                                  voice_net_sd=("voice_net", "std"),
                                  share_voice_pos=("voice_label", lambda s: (s == "positive").mean()),
                                  share_voice_neg=("voice_label", lambda s: (s == "negative").mean()),
                                  n_speakers=("speaker", "nunique")).sort_values("n", ascending=False)
    total_var = rep["voice_net"].var()
    within = rep.groupby("_key")["voice_net"].var().mul(rep.groupby("_key").size() - 1).sum() / (len(rep) - len(tab))
    # the same person saying the same words on different occasions (test-retest within speaker)
    g = rep.groupby(["_key", "speaker"])["voice_net"]
    n_g = g.size()
    rep_g = n_g[n_g >= 2]
    within_sp = (g.var()[rep_g.index].mul(rep_g - 1).sum() / (rep_g.sum() - len(rep_g))) if len(rep_g) else np.nan
    summary = {"n_sentences": int(len(rep)), "n_distinct_texts": int(len(tab)),
               "within_text_share_of_voice_variance": float(within / total_var) if total_var > 0 else np.nan,
               "n_text_speaker_repeats": int(rep_g.sum()), "n_text_speaker_groups": int(len(rep_g)),
               "within_text_speaker_share_of_voice_variance": float(within_sp / total_var) if total_var > 0 else np.nan,
               "voice_net_sd_total": float(np.sqrt(total_var)),
               "voice_net_sd_within_text": float(np.sqrt(within)),
               "voice_net_sd_within_text_speaker": float(np.sqrt(within_sp))}
    return tab, summary


# ==================================================================================================
# 7. text_models  (repository: src/vtc/text_models.py)
# ==================================================================================================
# Textual emotion of each sentence from two open-weight language models.
#
# 1. FinBERT (Huang, Wang, and Yang 2023; yiyanghkust/finbert-tone). BERT-base further pre-trained on
#    corporate reports, earnings-call transcripts, and analyst reports, fine-tuned on 10,000 analyst-report
#    sentences labelled positive / neutral / negative. WordPiece tokens -> 12-layer encoder -> [CLS] ->
#    linear layer -> softmax over the three tones. Labels are read from the model's id2label.
#
# 2. An instruction-tuned large language model (Qwen3-4B-Instruct-2507; Qwen Team 2025; Apache-2.0) used
#    zero-shot as a second rater. The prompt asks for the sentiment of the wording only. Instead of
#    sampling an answer, the scorer reads the model's next-token distribution at the answer position and
#    aggregates the probability of the label words:
#        p(label) = sum_{v in variants(label)} P(first token of v | prompt),  renormalised over the 3 labels
#    so the output is deterministic and probabilistic, like FinBERT's. The share of probability mass that
#    falls on the label words is kept (llm_mass) as a check that the model answered in the label space.

FINBERT = "yiyanghkust/finbert-tone"
LLM = "Qwen/Qwen3-4B-Instruct-2507"
TEXT_REVISIONS = {LLM: "cdbee75f17c01a7cc42f958dc650907174af0554",
                  FINBERT: "4921590d3c0c3832c0efea24c8381ce0bda7844b"}
TONES = ("positive", "neutral", "negative")

LLM_SYSTEM = "You classify the sentiment of sentences spoken by participants in corporate earnings conference calls."
LLM_PROMPT = ('Sentence: "{s}"\n\nJudge only the wording of the sentence, not how it might have been spoken. '
              "Is the sentiment it expresses about the company positive, neutral, or negative? "
              "Answer with one word: positive, neutral, or negative.")


class FinBERTScorer:
    """FinBERT's vocabulary (FinVocab) has its own special-token order ([PAD]=0, [EOS]=1, [UNK]=2, [CLS]=3,
    [SEP]=4), and recent transformers releases cannot build a tokenizer from the bare vocab.txt, so the
    uncased WordPiece tokenizer is built directly with the tokenizers library and checked on load."""

    def __init__(self, device: str | None = None):
        device = device or default_device()
        import torch
        from huggingface_hub import snapshot_download
        from tokenizers import BertWordPieceTokenizer
        from transformers import BertForSequenceClassification

        rev = TEXT_REVISIONS[FINBERT]
        path = snapshot_download(FINBERT, revision=rev)
        self.tok = BertWordPieceTokenizer(os.path.join(path, "vocab.txt"), lowercase=True)
        self.tok.enable_truncation(max_length=256)
        self.tok.enable_padding(pad_id=self.tok.token_to_id("[PAD]"), pad_token="[PAD]")
        probe = self.tok.encode("revenue grew strongly, margins declined.")
        if self.tok.token_to_id("[UNK]") in probe.ids[1:-1]:
            raise RuntimeError(f"FinBERT vocabulary not loaded correctly: {probe.tokens}")
        self.model = BertForSequenceClassification.from_pretrained(path).to(device).eval()
        self.revision = rev
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.device, self.torch = device, torch

    def __call__(self, sentences: list[str], batch_size: int = 64) -> list[dict]:
        torch, out = self.torch, []
        for i in range(0, len(sentences), batch_size):
            enc = self.tok.encode_batch(sentences[i:i + batch_size])
            ids = torch.tensor([e.ids for e in enc], device=self.device)
            mask = torch.tensor([e.attention_mask for e in enc], device=self.device)
            types = torch.tensor([e.type_ids for e in enc], device=self.device)
            with torch.inference_mode():
                logits = self.model(input_ids=ids, attention_mask=mask, token_type_ids=types).logits
                p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            for row in p:
                out.append({f"tx_{self.id2label[j]}": float(row[j]) for j in range(len(row))})
        return out


class LLMScorer:
    def __init__(self, device: str | None = None, model_name: str = LLM):
        device = device or default_device()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        rev = TEXT_REVISIONS.get(model_name)
        self.tok = AutoTokenizer.from_pretrained(model_name, revision=rev)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, revision=rev, dtype=torch.bfloat16).to(device).eval()
        self.device, self.torch = device, torch
        self.label_ids = {}
        for lab in TONES:
            ids = set()
            for v in (lab, lab.capitalize(), " " + lab, " " + lab.capitalize()):
                ids.add(self.tok.encode(v, add_special_tokens=False)[0])
            self.label_ids[lab] = sorted(ids)

    def prompt(self, s: str) -> str:
        msgs = [{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": LLM_PROMPT.format(s=s)}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def __call__(self, sentences: list[str], batch_size: int = 32) -> list[dict]:
        torch, out = self.torch, []
        for i in range(0, len(sentences), batch_size):
            enc = self.tok([self.prompt(s) for s in sentences[i:i + batch_size]], padding=True,
                           return_tensors="pt").to(self.device)
            with torch.inference_mode():
                logits = self.model(**enc).logits[:, -1, :].float()
            prob = torch.softmax(logits, dim=-1).cpu().numpy()
            for row in prob:
                mass = {lab: float(row[ids].sum()) for lab, ids in self.label_ids.items()}
                tot = sum(mass.values())
                rec = {f"llm_{lab}": (m / tot if tot > 0 else np.nan) for lab, m in mass.items()}
                rec["llm_mass"] = tot
                out.append(rec)
        return out


# ==================================================================================================
# 8. voice_models  (repository: src/vtc/voice_models.py)
# ==================================================================================================
# Vocal emotion of each sentence clip from two open-weight speech foundation models.
#
# Both models start from wav2vec 2.0 (Baevski et al. 2020): a convolutional encoder turns the 16 kHz
# waveform into 20 ms frames, and a transformer (24 layers, hidden size 1,024) contextualises them. The
# input is normalised per clip, x~ = (x - mean(x)) / sqrt(var(x) + 1e-7), as in Wav2Vec2FeatureExtractor.
#
# 1. FinVoc2Vec (Ewertz et al. 2025; waiv/FinVoc2Vec, CC BY-NC 4.0). wav2vec2-large-xlsr-53, further
#    pre-trained on 500,000 unlabelled conference-call sentences and fine-tuned on 5,000 linguistically
#    neutral conference-call sentences hand-labelled for vocal tone (negative / neutral / positive).
#        h_t = wav2vec2(x~)_t,   h = (1/T) sum_t h_t,   logits = W2 tanh(W1 h + b1) + b2,   p = softmax(logits)
#    The class is re-implemented below (it mirrors finvoc2vec_model.py in the model repository) and loaded
#    from the published safetensors weights, so no remote code is executed. Clips are truncated to 20 s,
#    as in the model card.
#
# 2. Dimensional speech emotion (Wagner et al. 2023; audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim,
#    CC BY-NC-SA 4.0). wav2vec2-large-robust pruned to 12 layers, fine-tuned on MSP-Podcast; regression head
#    on the time-averaged hidden state gives arousal, dominance, and valence (approximately 0 to 1).
#
# Model revisions are pinned (VOICE_REVISIONS) or recorded at run time (the snapshot's commit hash), so that
# every score can be reproduced with the same weights.

FINVOC = "waiv/FinVoc2Vec"
AUDEERING = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
VOICE_REVISIONS = {FINVOC: "63bac642b8c42a899d3f315730f995018fd1760e",
                   AUDEERING: "6eba34a2485ea31cb03600241787c3a5edab8626"}
MAX_SECONDS = 20.0


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / np.sqrt(x.var() + 1e-7)


def _snapshot(repo: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=VOICE_REVISIONS.get(repo))


class FinVoc2VecScorer:
    """p(negative), p(neutral), p(positive) of the speaker's vocal tone for one clip."""

    def __init__(self, device: str | None = None):
        device = device or default_device()
        import torch
        from torch import nn
        from transformers import Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2PreTrainedModel

        path = _snapshot(FINVOC)
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            cfg_dict = json.load(f)
        if cfg_dict.get("pooling_mode", "mean") != "mean":
            raise ValueError(f"unexpected pooling mode {cfg_dict.get('pooling_mode')}")
        cfg = Wav2Vec2Config(**{k: v for k, v in cfg_dict.items() if k not in ("model_type", "architectures", "auto_map")})

        class FinVoc2Vec(Wav2Vec2PreTrainedModel):
            config_class = Wav2Vec2Config

            def __init__(self, config):
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
                self.post_init()

            def forward(self, input_values):
                h = self.wav2vec2(input_values)[0]           # (batch, frames, hidden)
                pooled = h.mean(dim=1)                       # pooling_mode "mean"
                return self.out_proj(self.dropout(torch.tanh(self.dense(pooled))))

        self.model = FinVoc2Vec.from_pretrained(path, config=cfg).to(device).eval()
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.device, self.torch = device, torch
        self.revision = os.path.basename(path)

    def __call__(self, clip: np.ndarray) -> dict:
        torch = self.torch
        x = _normalize(clip[: int(MAX_SECONDS * 16000)])
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(x)[None].to(self.device))
            p = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()
        return {f"fv_{self.id2label[i]}": float(p[i]) for i in range(len(p))}


class DimensionalSER:
    """Arousal, dominance, valence (approximately 0-1) from the audEERING MSP-Podcast model."""

    def __init__(self, device: str | None = None):
        device = device or default_device()
        import torch
        from torch import nn
        from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

        class RegressionHead(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, features):
                x = self.dropout(features)
                x = torch.tanh(self.dense(x))
                x = self.dropout(x)
                return self.out_proj(x)

        class EmotionModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.config = config
                self.wav2vec2 = Wav2Vec2Model(config)
                self.classifier = RegressionHead(config)
                self.post_init()

            def forward(self, input_values):
                hidden = self.wav2vec2(input_values)[0]
                pooled = torch.mean(hidden, dim=1)
                return pooled, self.classifier(pooled)

        path = _snapshot(AUDEERING)
        self.model = EmotionModel.from_pretrained(path).to(device).eval()
        self.labels = [self.model.config.id2label[i].lower() for i in range(self.model.config.num_labels)]
        self.device, self.torch = device, torch
        self.revision = os.path.basename(path)

    def __call__(self, clip: np.ndarray) -> dict:
        torch = self.torch
        x = _normalize(clip[: int(MAX_SECONDS * 16000)])
        with torch.inference_mode():
            _, y = self.model(torch.from_numpy(x)[None].to(self.device))
        y = y[0].float().cpu().numpy()                       # labels from config: arousal, dominance, valence
        return {f"ser_{lab}": float(v) for lab, v in zip(self.labels, y)}


# ==================================================================================================
# 9. pitch  (repository: src/vtc/pitch.py)
# ==================================================================================================
# F0 tracking by Boersma's (1993) autocorrelation method (Praat "To Pitch (ac)"), vectorised over frames.
#
# This is the tracker of the earlier prototype, verified against Praat (identical frame grid and voicing
# decisions, mean absolute F0 difference 0.005 Hz), rewritten so that all frames of a clip are windowed and
# transformed at once. It replaces a Praat dependency whose binary can be blocked by Windows application-
# control policies.
#
#   1. Subtract the global mean; record the global absolute peak.
#   2. Frames are 3 periods of the pitch floor long (Praat's Hanning window); time step 0.75 / floor.
#   3. In each frame subtract the local mean, window, and compute the normalised autocorrelation r_xw(tau);
#      divide by the window's own autocorrelation:  r_x(tau) ~= r_xw(tau) / r_w(tau)   (Boersma's key step).
#   4. Local maxima of r_x between 1/ceiling and 1/floor above 0.5 * voicing_threshold are voiced
#      candidates, refined by parabolic interpolation; heights above 1 are reflected (r -> 1/r).
#   5. Unvoiced candidate:  U = v + max(0, 2 - (local_peak/global_peak) / (s / (1 + v)));
#      voiced candidates:   R = r - octave_cost * log2(ceiling / f).
#   6. A Viterbi path maximises sum(R) minus transition costs, octave_jump_cost * |log2(f1/f2)| between
#      voiced frames and voiced_unvoiced_cost at voicing changes, both times 0.01 / time_step.
# Defaults are Praat's: silence 0.03, voicing 0.45, octave cost 0.01, octave-jump 0.35, voiced/unvoiced 0.14.

def _next_pow2(n: int) -> int:
    return 1 << int(np.ceil(np.log2(max(1, n))))


def pitch_ac(x: np.ndarray, sr: int, time_step: float | None = None, pitch_floor: float = 75.0,
             pitch_ceiling: float = 600.0, max_candidates: int = 15, silence_threshold: float = 0.03,
             voicing_threshold: float = 0.45, octave_cost: float = 0.01, octave_jump_cost: float = 0.35,
             voiced_unvoiced_cost: float = 0.14):
    """Returns (times, f0) with f0 = 0 for unvoiced frames."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return np.empty(0), np.empty(0)
    x = x - x.mean()
    global_peak = float(np.max(np.abs(x)))
    time_step = time_step if time_step and time_step > 0 else 0.75 / pitch_floor
    win_dur = 3.0 / pitch_floor
    n_win = int(round(win_dur * sr))
    n_win += (n_win + 1) % 2                           # odd length: a true centre sample
    half = n_win // 2
    window = np.hanning(n_win + 2)[1:-1]               # Praat's Hanning: no zero end points
    nfft = _next_pow2(2 * n_win)
    rw = np.fft.irfft(np.abs(np.fft.rfft(window, nfft)) ** 2, nfft)[:n_win]
    rw /= rw[0]
    duration = len(x) / sr
    n_frames = int(np.floor((duration - win_dur) / time_step + 1e-9)) + 1
    if n_frames < 1 or global_peak == 0.0:
        return np.empty(0), np.empty(0)
    times = 0.5 * (duration - (n_frames - 1) * time_step) + time_step * np.arange(n_frames)

    xp = np.pad(x, (half, half + 1))                   # zero-pad so every frame has n_win samples
    centres = np.round(times * sr).astype(int)
    frames = xp[centres[:, None] + np.arange(n_win)[None, :]]
    frames = frames - frames.mean(axis=1, keepdims=True)
    local_peak = np.abs(frames).max(axis=1)
    intensity = np.minimum(local_peak / global_peak, 1.0)
    ac = np.fft.irfft(np.abs(np.fft.rfft(frames * window, nfft, axis=1)) ** 2, nfft, axis=1)[:, :n_win]

    min_lag = int(np.floor(sr / pitch_ceiling))
    max_lag = min(int(np.ceil(sr / pitch_floor)), n_win - 3)
    valid = (local_peak > 0) & (ac[:, 0] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(valid[:, None], ac[:, :max_lag + 2] / ac[:, :1], 0.0) / rw[:max_lag + 2]
        lags = np.arange(max(min_lag, 2), max_lag + 1)
        rm, r0, rp = r[:, lags - 1], r[:, lags], r[:, lags + 1]
        peak = valid[:, None] & (r0 > 0.5 * voicing_threshold) & (r0 > rm) & (r0 >= rp)
        dr = 0.5 * (rp - rm)
        d2r = 2.0 * r0 - rm - rp
        shift = np.where(d2r > 0, dr / d2r, 0.0)
        height = r0 + 0.5 * dr * shift
        height = np.where(height > 1.0, 1.0 / height, height)
        freq = sr / (lags + shift)
    peak &= (freq >= pitch_floor) & (freq <= pitch_ceiling)

    tsc = 0.01 / time_step
    ojc, vuc = octave_jump_cost * tsc, voiced_unvoiced_cost * tsc
    cand_f, delta = [], []
    for k in range(n_frames):
        f, h = freq[k, peak[k]], height[k, peak[k]]
        if len(f) > max_candidates - 1:
            keep = np.argsort(h)[::-1][: max_candidates - 1]
            f, h = f[keep], h[keep]
        unvoiced = voicing_threshold + max(0.0, 2.0 - intensity[k] / (silence_threshold / (1.0 + voicing_threshold)))
        cand_f.append(np.concatenate([[0.0], f]))
        delta.append(np.concatenate([[unvoiced], h - octave_cost * np.log2(pitch_ceiling / f)]))

    score, back = delta[0].copy(), []
    for k in range(1, n_frames):                       # Viterbi; candidate 0 = unvoiced
        pf, cf = cand_f[k - 1], cand_f[k]
        pv, cv = pf > 0, cf > 0
        trans = np.zeros((len(pf), len(cf)))
        both = pv[:, None] & cv[None, :]
        if both.any():
            ratio = np.abs(np.log2(np.where(both, pf[:, None], 1.0) / np.where(both, cf[None, :], 1.0)))
            trans[both] = ojc * ratio[both]
        trans[pv[:, None] ^ cv[None, :]] = vuc
        total = score[:, None] - trans + delta[k][None, :]
        back.append(np.argmax(total, axis=0))
        score = np.max(total, axis=0)
    path = np.zeros(n_frames, dtype=int)
    path[-1] = int(np.argmax(score))
    for k in range(n_frames - 1, 0, -1):
        path[k - 1] = back[k - 1][path[k]]
    return times, np.array([cand_f[k][path[k]] for k in range(n_frames)])


# ==================================================================================================
# 10. acoustics  (repository: src/vtc/acoustics.py)
# ==================================================================================================
# Glass-box prosodic descriptors of each sentence clip.
#
# These interpretable measures show what the speech models respond to; they are not the emotion measures.
#   f0_median_st, f0_sd_st   median and SD of F0 in semitones re 100 Hz over voiced frames
#                            (Boersma's autocorrelation tracker, pitch.py; floor 75 Hz, ceiling 500 Hz, 10 ms step)
#   intensity_sd_db          SD of frame level (dB; 25 ms frames, 10 ms hop) over frames within 25 dB of the
#                            clip's loudest frame
#   voiced_frac              share of 10 ms frames with a voiced F0 estimate
#   tokens_per_sec           transcript tokens / aligned duration (from align.py)
# Absolute levels depend on the recording channel, so analyses use them within speaker and call.

def prosody(clip: np.ndarray, sr: int = 16000) -> dict:
    out = {"f0_median_st": np.nan, "f0_sd_st": np.nan, "intensity_sd_db": np.nan, "voiced_frac": np.nan}
    if len(clip) < int(0.3 * sr):
        return out
    _, f0 = pitch_ac(clip, sr, time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0)
    voiced = f0[f0 > 0]
    out["voiced_frac"] = float(len(voiced) / max(len(f0), 1))
    if len(voiced) >= 5:
        st = 12.0 * np.log2(voiced / 100.0)
        out["f0_median_st"] = float(np.median(st))
        out["f0_sd_st"] = float(np.std(st))
    n, hop = int(0.025 * sr), int(0.010 * sr)
    if len(clip) >= n:
        idx = np.arange(0, len(clip) - n + 1, hop)[:, None] + np.arange(n)[None, :]
        rms = np.sqrt(np.mean(clip.astype(np.float64)[idx] ** 2, axis=1))
        db = 20.0 * np.log10(rms + 1e-10)
        db = db[db >= db.max() - 25.0]
        out["intensity_sd_db"] = float(np.std(db))
    return out


# ==================================================================================================
# 11. pipeline  (repository: src/vtc/pipeline.py)
# ==================================================================================================
# End-to-end processing of one earnings call: transcript PDF + recording -> one row per sentence.
#
#   1. parse the vendor transcript into speaker turns (transcripts.py) and sentences (sentences.py)
#   2. decode the recording (audio.py) and recognise it with word time stamps (asr.py, cached)
#   3. place every transcript sentence in time (align.py)
#   4. score each timed sentence clip: vocal tone (FinVoc2Vec), arousal/dominance/valence (dimensional
#      SER), prosody (Praat)                                                   (voice_models.py, acoustics.py)
#   5. score each sentence's wording: FinBERT tone                            (text_models.py)
#   6. write sentences/<call_id>.csv.gz and meta/<call_id>.json
# The instruction-tuned LLM rater is a separate pass over the saved tables (scripts/score_llm.py), so the
# speech models and the 4-billion-parameter LLM never compete for GPU memory.

MIN_CLIP_SECONDS = 1.0
CALL_NAME = re.compile(r"_Q([1-4])_(\d{4})_Earnings?_Call_([A-Za-z]{3})[_-](\d{1,2})[_-](\d{4})")


class Models:
    """Loads each model once, on first use."""

    def __init__(self, device: str | None = None, whisper: str = DEFAULT_MODEL):
        device = device or default_device()
        self.device, self.whisper_name, self._m = device, whisper, {}

    def get(self, key: str):
        if key not in self._m:
            if key == "whisper":
                self._m[key] = load_whisper(self.whisper_name, device=self.device)
            elif key == "finvoc":
                self._m[key] = FinVoc2VecScorer(self.device)
            elif key == "ser":
                self._m[key] = DimensionalSER(self.device)
            elif key == "finbert":
                self._m[key] = FinBERTScorer(self.device)
        return self._m[key]

    def versions(self) -> dict:
        import faster_whisper
        import torch
        import transformers

        out = {"python": platform.python_version(), "torch": torch.__version__,
               "transformers": transformers.__version__, "faster_whisper": faster_whisper.__version__,
               "whisper_model": self.whisper_name,
               "whisper_revision": WHISPER_REPOS.get(self.whisper_name, (None, None))[1]}
        for k in ("finvoc", "ser", "finbert"):
            if k in self._m:
                out[f"{k}_revision"] = self._m[k].revision
        return out


def call_info(mp3_path: str) -> dict:
    stem = os.path.splitext(os.path.basename(mp3_path))[0]
    m = CALL_NAME.search(stem)
    info = {"call_id": stem, "company": os.path.basename(os.path.dirname(mp3_path))}
    if m:
        info.update(fiscal_quarter=int(m.group(1)), fiscal_year=int(m.group(2)),
                    call_date=pd.to_datetime(f"{m.group(4)} {m.group(3)} {m.group(5)}", format="%d %b %Y").date().isoformat())
    return info


def sentence_rows(tr) -> list[dict]:
    rows = []
    for t in tr.turns:
        for k, s in enumerate(split_sentences(t.text)):
            toks = normalize_tokens(s)
            if not toks:
                continue
            rows.append({"turn": t.order, "sent_in_turn": k, "section": t.section, "speaker": t.speaker,
                         "title": t.title, "role": t.role, "text": s, "tokens": toks})
    return rows


def process_call(mp3: str, transcript: str, out_dir: str, models: Models, log=print) -> str | None:
    info = call_info(mp3)
    cid = info["call_id"]
    out_csv = os.path.join(out_dir, "sentences", f"{cid}.csv.gz")
    if os.path.exists(out_csv):
        return out_csv
    t0 = time.time()
    tr = load_transcript(transcript)
    rows = sentence_rows(tr)
    if not rows:
        raise ValueError("no sentences parsed from transcript")
    wav = load_audio(mp3)
    rec = cached_transcription(os.path.join(out_dir, "asr", f"{cid}.json.gz"), models.get("whisper"), wav)
    t_asr = time.time()
    spans = align_sentences([r["tokens"] for r in rows], rec["words"])
    for r, sp in zip(rows, spans):
        r.update(sp)
        r["align_sim"] = round(difflib.SequenceMatcher(None, r["tokens"], normalize_tokens(sp["asr_text"]),
                                                       autojunk=False).ratio(), 3) if sp["asr_text"] else 0.0
        r["n_words"] = len(r.pop("tokens"))

    fv, ser = models.get("finvoc"), models.get("ser")
    for r in rows:
        if r["timed"] and r["duration"] >= MIN_CLIP_SECONDS:
            c = clip(wav, r["start"], r["end"])
            r.update(fv(c))
            r.update(ser(c))
            r.update(prosody(c))
            r["truncated"] = r["duration"] > 20.0
    texts = [r["text"] for r in rows]
    for r, s in zip(rows, models.get("finbert")(texts)):
        r.update(s)

    df = pd.DataFrame(rows)
    for k, v in info.items():
        df.insert(0, k, v)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv + ".tmp", index=False, compression="gzip")
    os.replace(out_csv + ".tmp", out_csv)                    # never leave a half-written table
    meta = dict(info, transcript_header=tr.header, n_turns=len(tr.turns), n_sentences=len(df),
                n_timed=int(df["timed"].sum()), n_voice_scored=int(df.get("fv_positive", pd.Series(dtype=float)).notna().sum()),
                median_coverage=float(df["coverage"].median()), audio_seconds=round(len(wav) / SR, 1),
                asr_seconds=round(t_asr - t0, 1), total_seconds=round(time.time() - t0, 1),
                participants={k: list(v) for k, v in tr.participants.items()}, versions=models.versions())
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)
    with open(os.path.join(out_dir, "meta", f"{cid}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    log(f"{cid}: {meta['n_sentences']} sentences, {meta['n_voice_scored']} voice-scored, "
        f"coverage {meta['median_coverage']:.2f}, audio {meta['audio_seconds']:.0f}s, "
        f"ASR {meta['asr_seconds']:.0f}s, total {meta['total_seconds']:.0f}s")
    return out_csv


# ==================================================================================================
# 12. report  (repository: src/vtc/report.py)
# ==================================================================================================
# Sample-level results: the text x voice table, its statistics, comparisons, validity checks, examples.
#
# Calls in which fewer than 20% of the transcript sentences can be placed in the recording are excluded
# (listed in calls_alignment.csv and summary.json). The recogniser's word rate tells the two reasons apart: a
# recording of a different event is recognised at a normal rate (>= 100 words per minute) but does not match
# the transcript, whereas a degraded recording is recognised only in fragments. Sentence-level alignment
# failure in retained calls has the same source, so results are also reported for well-aligned calls only.
# Analysis sample: sentences that are timed (align.py), at least 1 s long, scored by both channels, and
# spoken by executives (CEO, CFO, other executives; investor-relations officers are reported separately
# because they mostly read safe-harbour statements). Analysts are the comparison group.
# All intervals are cluster-bootstrap 95% intervals over calls (congruence.cluster_bootstrap).

EXECUTIVES = ("CEO", "CFO", "other_executive")
CELL_CODES = tuple(code for code, _, _ in CELLS.values())

MIN_CALL_TIMED = 0.2        # calls where fewer sentences align are excluded (mismatched or unreadable recordings)
MIN_ASR_WPM = 100           # below this recognised-word rate a recording counts as degraded
HIGH_QUALITY_TIMED = 0.8    # robustness subsample: calls in which at least 80% of sentences align


def load(out_dir: str, return_calls: bool = False):
    files = sorted(glob.glob(os.path.join(out_dir, "sentences", "*.csv.gz")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    calls = df.groupby("call_id").agg(company=("company", "first"), n_sentences=("timed", "size"),
                                      share_timed=("timed", "mean"))
    calls["matched"] = calls["share_timed"] >= MIN_CALL_TIMED
    keep = df["call_id"].map(calls["matched"])
    ok = keep & df["timed"].astype(bool) & (df["duration"] >= 1.0) & df["fv_positive"].notna() & df["tx_positive"].notna()
    out = add_congruence(df[ok].reset_index(drop=True))
    return (out, calls) if return_calls else out


def shares(d: pd.DataFrame) -> pd.DataFrame:
    t = crosstab3(d)
    return t / t.to_numpy().sum()


def table_with_ci(d: pd.DataFrame, reps: int) -> pd.DataFrame:
    rows = []
    for (t, v), (code, name, desc) in CELLS.items():
        hit = (d["text_label"] == t) & (d["voice_label"] == v)
        lo, hi = cluster_bootstrap(hit, d["call_id"], reps=reps)
        in_row = d["text_label"] == t
        rlo, rhi = cluster_bootstrap(hit[in_row], d.loc[in_row, "call_id"], reps=reps)
        rows.append({"cell": code, "name": name, "description": desc, "text": t, "voice": v,
                     "n": int(hit.sum()), "share": float(hit.mean()), "ci_low": lo, "ci_high": hi,
                     "share_within_text_row": float(hit[in_row].mean()) if in_row.any() else np.nan,
                     "row_ci_low": rlo, "row_ci_high": rhi})
    return pd.DataFrame(rows)


def summary_measures(d: pd.DataFrame) -> dict:
    return {"n": int(len(d)), "aligned": float(d["aligned"].mean()), "vocal_lift": float((d["direction"] > 0).mean()),
            "vocal_drag": float((d["direction"] < 0).mean()), "reversal": float(d["reversal"].mean()),
            "mean_abs_gap": float(d["gap"].abs().mean()),
            "corr_text_voice": float(d[["text_net", "voice_net"]].corr(method="spearman").iloc[0, 1])}


def compare(d: pd.DataFrame, col: str, levels, reps: int) -> pd.DataFrame:
    rows = []
    for lev in levels:
        sub = d[d[col] == lev]
        if sub.empty:
            continue
        s = summary_measures(sub)
        s["aligned_ci"] = cluster_bootstrap(sub["aligned"], sub["call_id"], reps=reps)
        s["drag_ci"] = cluster_bootstrap(sub["direction"] < 0, sub["call_id"], reps=reps)
        s["lift_ci"] = cluster_bootstrap(sub["direction"] > 0, sub["call_id"], reps=reps)
        s["reversal_ci"] = cluster_bootstrap(sub["reversal"], sub["call_id"], reps=reps)
        for (t, v), (code, _, _) in CELLS.items():
            s[code] = float(((sub["text_label"] == t) & (sub["voice_label"] == v)).mean())
        rows.append(dict(group=lev, **s))
    return pd.DataFrame(rows)


def within_call_difference(d: pd.DataFrame, measure: str, reps: int, col: str = "section", a: str = "qa",
                           b: str = "presentation") -> dict:
    """Group a minus group b (default: Q&A minus presentation), computed within each call that has both and
    then averaged over calls; the interval resamples calls."""
    g = d.groupby(["call_id", col])[measure].mean().unstack()
    if not {a, b} <= set(g.columns):
        return {}
    g = g.dropna(subset=[a, b])
    if g.empty:
        return {}
    diff = (g[a] - g[b]).to_numpy(float)
    rng = np.random.default_rng(0)
    boot = rng.choice(diff, size=(reps, len(diff))).mean(1)
    return {"measure": measure, "n_calls": int(len(diff)), "mean_a": float(g[a].mean()), "mean_b": float(g[b].mean()),
            "mean_diff": float(diff.mean()), "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5))}


def asr_density(out_dir: str, call_ids) -> pd.Series:
    """Recognised words per minute of recording (about 130-160 when the recogniser can read the audio)."""
    out = {}
    for cid in call_ids:
        p = os.path.join(out_dir, "asr", f"{cid}.json.gz")
        if os.path.exists(p):
            with gzip.open(p, "rt", encoding="utf-8") as f:
                rec = json.load(f)
            out[cid] = 60 * len(rec["words"]) / rec["duration"] if rec.get("duration") else np.nan
    return pd.Series(out, dtype=float)


def heatmap(table: pd.DataFrame, path: str, title: str, counts: pd.DataFrame | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    vals = table.to_numpy(float) * 100
    ax.imshow(vals, cmap="Greys", vmin=0, vmax=max(vals.max(), 1) * 1.15)
    for i in range(3):
        for j in range(3):
            code = CELLS[(LABELS[i], LABELS[j])][0]
            txt = f"{code}\n{vals[i, j]:.1f}%" + (f"\n(n={int(counts.iloc[i, j]):,})" if counts is not None else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color="white" if vals[i, j] > vals.max() * 0.6 else "black", family="Times New Roman")
    ax.set_xticks(range(3), [f"{l}" for l in LABELS], family="Times New Roman")
    ax.set_yticks(range(3), [f"{l}" for l in LABELS], family="Times New Roman")
    ax.set_xlabel("Vocal tone (FinVoc2Vec)", family="Times New Roman")
    ax.set_ylabel("Textual tone (FinBERT)", family="Times New Roman")
    ax.set_title(title, family="Times New Roman", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def examples(d: pd.DataFrame, per_cell: int = 3) -> pd.DataFrame:
    """The most confident instances of each cell: P_T(text label) x P_V(voice label)."""
    pos = {lab: i for i, lab in enumerate(LABELS)}
    tx = d[[f"tx_{lab}" for lab in LABELS]].to_numpy()
    fv = d[[f"fv_{lab}" for lab in LABELS]].to_numpy()
    rows = np.arange(len(d))
    conf = tx[rows, d["text_label"].map(pos).to_numpy()] * fv[rows, d["voice_label"].map(pos).to_numpy()]
    d = d.assign(conf=conf)
    keep = d[(d["n_words"] >= 6) & (d["n_words"] <= 30)]
    out = keep.sort_values("conf", ascending=False).groupby("cell").head(per_cell)
    return out[["cell", "cell_name", "call_id", "section", "role", "start", "end", "text", "text_net",
                "voice_net", "conf"]].sort_values(["cell", "conf"], ascending=[True, False])


def run(out_dir: str, reps: int = 1000, name: str = "report") -> dict:
    rep_dir = os.path.join(out_dir, name)
    os.makedirs(rep_dir, exist_ok=True)
    allrows, calls = load(out_dir, return_calls=True)
    calls["asr_wpm"] = asr_density(out_dir, calls.index)
    calls["exclusion"] = np.where(calls["matched"], "", np.where(calls["asr_wpm"] < MIN_ASR_WPM, "degraded recording",
                                                                  "recording does not match transcript"))
    calls.to_csv(os.path.join(rep_dir, "calls_alignment.csv"))
    ex = allrows[allrows["role"].isin(EXECUTIVES)].copy()
    for code in CELL_CODES:
        ex[code] = ex["cell"] == code
    ex["drag"], ex["lift"] = ex["direction"] < 0, ex["direction"] > 0
    res = {"n_calls_processed": int(len(calls)), "n_calls_mismatched": int((~calls["matched"]).sum()),
           "mismatched_calls": calls.index[~calls["matched"]].tolist(),
           "exclusion_reasons": calls.loc[~calls["matched"], "exclusion"].value_counts().to_dict(),
           "n_calls": int(allrows["call_id"].nunique()), "n_companies": int(allrows["company"].nunique()),
           "n_sentences_all_speakers": int(len(allrows)), "n_executive_sentences": int(len(ex))}

    tab = crosstab3(ex)
    stats = independence(tab)
    tab.to_csv(os.path.join(rep_dir, "table_counts_executives.csv"))
    shares(ex).to_csv(os.path.join(rep_dir, "table_shares_executives.csv"))
    stats.pop("adj_residuals").to_csv(os.path.join(rep_dir, "adjusted_residuals_executives.csv"))
    res["independence_executives"] = stats
    cells = table_with_ci(ex, reps)
    cells.to_csv(os.path.join(rep_dir, "cells_executives.csv"), index=False)
    res["summary_executives"] = summary_measures(ex)
    heatmap(shares(ex), os.path.join(rep_dir, "fig_heatmap_executives.png"),
            "Executives: textual x vocal tone", counts=tab)

    by_section = compare(ex, "section", ["presentation", "qa"], reps)
    by_section.to_csv(os.path.join(rep_dir, "by_section.csv"), index=False)
    by_role = compare(allrows, "role", ["CEO", "CFO", "other_executive", "IR", "analyst", "operator"], reps)
    by_role.to_csv(os.path.join(rep_dir, "by_role.csv"), index=False)
    measures = ("aligned", "drag", "lift", "reversal", "gap", "voice_net", "text_net") + CELL_CODES
    for key, kw in (("qa_minus_presentation", {}), ("ceo_minus_cfo", {"col": "role", "a": "CEO", "b": "CFO"})):
        diffs = pd.DataFrame([within_call_difference(ex, m, reps, **kw) for m in measures])
        diffs.to_csv(os.path.join(rep_dir, f"{key}.csv"), index=False)
        res[key] = diffs.to_dict("records")
    for sec in ("presentation", "qa"):
        sub = ex[ex["section"] == sec]
        if len(sub):
            heatmap(shares(sub), os.path.join(rep_dir, f"fig_heatmap_{sec}.png"),
                    f"Executives, {'prepared remarks' if sec == 'presentation' else 'Q&A'}", counts=crosstab3(sub))

    # channel independence: how much of the vocal tone is explained by the words?
    r = ex[["text_net", "voice_net"]].corr().iloc[0, 1]
    res["voice_variance_explained_by_text_r2"] = float(r ** 2)
    # same words, different voice
    same_tab, same_sum = same_text_dispersion(allrows[allrows["role"] != "operator"], min_count=5)
    if len(same_tab):
        same_tab.to_csv(os.path.join(rep_dir, "same_text_dispersion.csv"))
    res["same_text"] = same_sum
    # robustness: speaker-relative vocal tone
    rel = ex.assign(voice_label=speaker_relative(ex))
    rel_stats = independence(crosstab3(rel))
    rel_stats.pop("adj_residuals")
    res["speaker_relative"] = {"independence": rel_stats, "shares": shares(rel).round(4).to_dict()}
    # robustness: second textual rater (LLM)
    if "llm_positive" in ex.columns and ex["llm_positive"].notna().any():
        llm = add_congruence(ex[ex["llm_positive"].notna()], text="llm_")
        xtab = pd.crosstab(llm["text_label"], ex.loc[llm.index, "text_label"]).reindex(
            index=LABELS, columns=LABELS, fill_value=0)              # rows: LLM label, columns: FinBERT label
        xtab.to_csv(os.path.join(rep_dir, "llm_vs_finbert.csv"))
        k = independence(xtab)
        li = independence(crosstab3(llm))
        li.pop("adj_residuals")
        res["llm"] = {"n": int(len(llm)), "kappa_llm_vs_finbert": k["kappa"], "agreement": k["agreement"],
                      "mean_label_mass": float(ex["llm_mass"].mean()), "independence": li,
                      "voice_variance_explained_by_text_r2": float(llm[["text_net", "voice_net"]].corr().iloc[0, 1] ** 2),
                      "summary": summary_measures(llm), "shares": shares(llm).round(4).to_dict()}
        shares(llm).to_csv(os.path.join(rep_dir, "table_shares_executives_llm.csv"))
    # robustness: calls whose recordings the recogniser reads well
    good = calls.index[calls["matched"] & (calls["share_timed"] >= HIGH_QUALITY_TIMED)]
    hq = ex[ex["call_id"].isin(good)]
    if len(hq):
        hs = independence(crosstab3(hq))
        hs.pop("adj_residuals")
        res["high_quality_calls"] = {
            "min_share_timed": HIGH_QUALITY_TIMED, "n_calls": int(hq["call_id"].nunique()), "independence": hs,
            "summary": summary_measures(hq), "shares": shares(hq).round(4).to_dict(),
            "voice_variance_explained_by_text_r2": float(hq[["text_net", "voice_net"]].corr().iloc[0, 1] ** 2),
            "qa_minus_presentation": [within_call_difference(hq, m, reps) for m in ("aligned", "reversal", "C3", "C6")]}
    # where does the variation sit: between companies, between calls, between speakers within calls?
    spk = ex["call_id"] + "|" + ex["speaker"].astype(str)
    lev = []
    for m in ("aligned", "drag", "lift", "reversal", "gap", "voice_net", "text_net"):
        cm = ex.groupby("call_id")[m].mean().astype(float)
        lev.append({"measure": m, "mean_of_calls": cm.mean(), "sd_across_calls": cm.std(), "p10": cm.quantile(0.1),
                    "p90": cm.quantile(0.9), "icc_company": icc1(ex[m], ex["company"]),
                    "icc_call": icc1(ex[m], ex["call_id"]), "icc_speaker_in_call": icc1(ex[m], spk)})
    lev = pd.DataFrame(lev)
    lev.to_csv(os.path.join(rep_dir, "variance_levels.csv"), index=False)
    res["variance_levels"] = lev.round(4).to_dict("records")
    # convergent validity of the vocal measure
    cols = ["voice_net", "ser_valence", "ser_arousal", "ser_dominance", "f0_sd_st", "intensity_sd_db", "tokens_per_sec"]
    ex[cols].corr(method="spearman").round(3).to_csv(os.path.join(rep_dir, "voice_correlations.csv"))
    # lexical leakage: does a vocal score move with the words (text label) or with the voice (vocal-tone label)?
    leak = pd.DataFrame({m: {"rho_text_net": ex[m].corr(ex["text_net"], method="spearman"),
                             "rho_voice_net": ex[m].corr(ex["voice_net"], method="spearman"),
                             "eta2_text_label": eta2(ex[m], ex["text_label"]),
                             "eta2_voice_label": eta2(ex[m], ex["voice_label"])}
                         for m in ("voice_net", "ser_valence", "ser_arousal", "ser_dominance", "f0_sd_st")}).T
    leak.round(4).to_csv(os.path.join(rep_dir, "lexical_leakage.csv"))
    res["lexical_leakage"] = leak.round(4).to_dict("index")
    pros = ex.groupby(["text_label", "voice_label"])[["ser_arousal", "ser_valence", "f0_sd_st", "intensity_sd_db",
                                                     "tokens_per_sec"]].mean().round(3)
    pros.to_csv(os.path.join(rep_dir, "prosody_by_cell.csv"))
    examples(ex).to_csv(os.path.join(rep_dir, "examples_by_cell.csv"), index=False)
    # alignment quality
    allsent = pd.concat([pd.read_csv(f, usecols=["call_id", "timed", "coverage", "align_sim"])
                         for f in glob.glob(os.path.join(out_dir, "sentences", "*.csv.gz"))])
    kept = allsent[allsent["call_id"].map(calls["matched"]).fillna(False).astype(bool)]
    res["alignment"] = {"share_timed": float(allsent["timed"].mean()),
                        "share_timed_retained_calls": float(kept["timed"].mean()),
                        "median_coverage": float(allsent["coverage"].median()),
                        "median_align_sim_timed": float(allsent.loc[allsent["timed"], "align_sim"].median()),
                        "corr_share_timed_asr_wpm": float(calls[["share_timed", "asr_wpm"]].corr().iloc[0, 1]),
                        "n_calls_high_quality": int((calls["matched"] & (calls["share_timed"] >= HIGH_QUALITY_TIMED)).sum()),
                        "median_asr_wpm": float(calls["asr_wpm"].median())}
    with open(os.path.join(rep_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return res


# ==================================================================================================
# Module namespaces: the appendix cites functions as module.function (e.g. align.align_sentences)
# ==================================================================================================
def _namespace(names):
    g = globals()
    return SimpleNamespace(**{n: g[n] for n in names.split()})


sentences = _namespace("ABBREVIATIONS BRACKETED SENTENCE_END clean_text split_sentences normalize_tokens")
transcripts = _namespace("read_manifest SECTION_NAMES OPERATOR_CUE Turn Transcript classify_role line_kind read_lines NAME_PARTICLES looks_like_name parse_lines infer_qa parse_transcript_pdf TABLE_EXTENSIONS TRANSCRIPT_EXTENSIONS SECTION_ALIASES ROLES EXECUTIVE_TITLE parse_turn_table fallback_group load_transcript")
audio = _namespace("SR default_device load_audio clip")
asr = _namespace("DEFAULT_MODEL WHISPER_REPOS load_whisper transcribe_words cached_transcription")
align = _namespace("MIN_COVERAGE RATE_RANGE asr_tokens align_sentences")
congruence = _namespace("LABELS SIGN CELLS argmax_label add_congruence crosstab3 independence cluster_bootstrap icc1 eta2 speaker_relative same_text_dispersion")
text_models = _namespace("FINBERT LLM TEXT_REVISIONS TONES LLM_SYSTEM LLM_PROMPT FinBERTScorer LLMScorer")
voice_models = _namespace("FINVOC AUDEERING VOICE_REVISIONS MAX_SECONDS FinVoc2VecScorer DimensionalSER")
pitch = _namespace("pitch_ac")
acoustics = _namespace("prosody")
pipeline = _namespace("MIN_CLIP_SECONDS CALL_NAME Models call_info sentence_rows process_call")
report = _namespace("EXECUTIVES CELL_CODES MIN_CALL_TIMED MIN_ASR_WPM HIGH_QUALITY_TIMED load shares table_with_ci summary_measures compare within_call_difference asr_density heatmap examples run")


# ==================================================================================================
# Verification suite  (repository: tests/test_vtc.py)
# ==================================================================================================
# Verification suite for the parts of the pipeline that do not need a GPU or model downloads.
#
# Run:  python tests/test_vtc.py          (plain runner, prints a results table)

REPORT = []


class Skip(Exception):
    pass


def record(name, value, truth, tol, note=""):
    ok = abs(value - truth) <= tol
    REPORT.append(dict(test=name, estimate=round(float(value), 5), truth=round(float(truth), 5),
                       tolerance=tol, passed=ok, note=note))
    assert ok, f"{name}: {value} vs {truth} (tol {tol})"


# ------------------------------------------------------------------ transcripts
def test_line_kinds():
    record("24 pt bold line is a section header", line_kind("Arial-BoldMT", 24, 16, 0, "Presentation") == "header", 1, 0)
    record("10 pt bold line is a speaker", line_kind("Arial-BoldMT", 10, 16, 0, "Jane Roe") == "speaker", 1, 0)
    record("grey upper-case bold line is a group", line_kind("Arial-BoldMT", 10, 16, 6710886, "ANALYSTS") == "group", 1, 0)
    record("italic line is a title", line_kind("Arial-ItalicMT", 10, 2, 0, "CEO & Director") == "title", 1, 0)
    record("running header is metadata", line_kind("Verdana-Bold", 10, 16, 1, "X CORP FQ1 2022 EARNINGS CALL") == "running_header", 1, 0)
    record("8 pt disclaimer is skipped", line_kind("ArialUnicodeMS", 8, 0, 0, "Copyright") is None, 1, 0)
    old = [line_kind("Verdana-Bold", 24, 16, 0, "Presentation"), line_kind("Verdana-Bold", 10, 16, 0, "Operator"),
           line_kind("Verdana-Italic", 10, 2, 0, "CEO & Director"), line_kind("Verdana", 10, 0, 0, "Thank you, Jane.")]
    record("older Verdana template: header, speaker, title, body", float(old == ["header", "speaker", "title", "body"]), 1, 0)


def test_parse_lines():
    lines = [("running_header", "ACME CORP FQ1 2024 EARNINGS CALL MAY 1, 2024"),
             ("header", "Call Participants"), ("group", "EXECUTIVES"), ("speaker", "Jane Roe"),
             ("title", "CEO & Director"), ("speaker", "John Doe"), ("title", "Executive VP &"), ("title", "CFO"),
             ("group", "ANALYSTS"), ("speaker", "Ann Lee"), ("title", "Big Bank, Research"), ("title", "Division"),
             ("header", "Presentation"), ("speaker", "Operator"), ("body", "Welcome to the call."),
             ("speaker", "Jane Roe"), ("title", "CEO & Director"), ("body", "We had a strong quarter. Margins were up"),
             ("body", "and costs were well con-"), ("body", "trolled."),
             ("header", "Question and Answer"), ("body", "[Operator Instructions] Our first question comes from Ann Lee."),
             ("speaker", "Ann Lee"), ("title", "Big Bank, Research Division"), ("body", "What about pricing?"),
             ("speaker", "John Doe"), ("title", "Executive VP & CFO"), ("body", "Pricing was stable.")]
    tr = parse_lines(lines)
    record("turns parsed", len(tr.turns), 5, 0)
    roles = [t.role for t in tr.turns]
    record("roles: operator, CEO, operator, analyst, CFO",
           float(roles == ["operator", "CEO", "operator", "analyst", "CFO"]), 1, 0, str(roles))
    record("sections assigned", float([t.section for t in tr.turns] == ["presentation"] * 2 + ["qa"] * 3), 1, 0)
    record("wrapped hyphenated word re-joined", float("well con-trolled." in tr.turns[1].text), 1, 0)
    record("two-line title joined", float(tr.participants["John Doe"][1] == "Executive VP & CFO"), 1, 0)


def test_name_in_regular_font():
    # accented names set in a regular fallback font: recognised from the title beneath them or the participant list
    # (all names are fictitious)
    lines = [("header", "Call Participants"), ("group", "EXECUTIVES"), ("body", "Marta Ibáñez-Ruiz"),
             ("title", "President, CEO & Director"), ("speaker", "John Q. Public"), ("title", "Investor Relations Officer"),
             ("body", "Raúl Peña-Núñez"), ("title", "Executive VP & CFO"),
             ("header", "Presentation"), ("speaker", "John Q. Public"), ("title", "Investor Relations Officer"),
             ("body", "I'd like to turn the call over to our CEO, Marta Ibáñez."), ("body", "Marta Ibáñez-Ruiz"),
             ("title", "President, CEO & Director"), ("body", "Thank you, John. It was a good quarter."),
             ("body", "Raúl Peña-Núñez"), ("body", "Thank you, Marta.")]
    tr = parse_lines(lines)
    record("accented names in a regular font: speakers and roles",
           float([(t.speaker, t.role) for t in tr.turns] == [("John Q. Public", "IR"), ("Marta Ibáñez-Ruiz", "CEO"),
                                                             ("Raúl Peña-Núñez", "CFO")]), 1, 0,
           str([(t.speaker, t.role) for t in tr.turns]))


def test_missing_qa_header():
    lines = [("header", "Presentation"), ("speaker", "Operator"),
             ("body", "Welcome. All lines are in listen-only mode. [Operator Instructions]"),
             ("speaker", "Jane Roe"), ("title", "CEO"), ("body", "We had a strong quarter."),
             ("speaker", "Operator"), ("body", "[Operator Instructions] Our first question comes from Ann Lee."),
             ("speaker", "Ann Lee"), ("title", "Big Bank, Research Division"), ("body", "What about pricing?"),
             ("speaker", "Jane Roe"), ("title", "CEO"), ("body", "Pricing was stable.")]
    secs = [t.section for t in parse_lines(lines).turns]
    record("Q&A inferred without its header", float(secs == ["presentation"] * 2 + ["qa"] * 3), 1, 0, str(secs))


def test_classify_role():
    cases = [("A", "President, CEO & Director", "executives", "CEO"), ("B", "Executive VP & CFO", "executives", "CFO"),
             ("C", "Vice President of Investor Relations", "executives", "IR"), ("D", "COO", "executives", "other_executive"),
             ("Unknown Analyst", "", "", "analyst"), ("Operator", "", "", "operator")]
    record("role rules", float(all(classify_role(n, t, g) == r for n, t, g, r in cases)), 1, 0)


def test_call_info():
    info = call_info(r"sample/acme/Acme_Corporation_Q1_2022_Earnings_Call_May_11_2022.mp3")
    record("call date parsed", float(info["call_date"] == "2022-05-11" and info["fiscal_quarter"] == 1), 1, 0)


# ------------------------------------------------------------------ sentences
def test_split_sentences():
    s = ("Thank you, Mr. Smith. Revenue grew 4.5% in the U.S. and abroad. John Q. Public will cover margins. "
         "Did we beat guidance? Yes! [Operator Instructions] Next, Q1 was strong.")
    out = split_sentences(s)
    record("sentence count", len(out), 6, 0, " | ".join(out))
    record("abbreviation and decimal kept inside sentence", float(out[1] == "Revenue grew 4.5% in the U.S. and abroad."), 1, 0)
    old = split_sentences("We will conduct a question session. (Operator instructions). We now turn the call over.")
    record("parenthesised operator note removed", float(not any("instructions" in s.lower() for s in old)), 1, 0, " | ".join(old))


def test_normalize_tokens():
    a = normalize_tokens("Revenue of $1,234.5 million, up 12% & rising; we're pleased.")
    b = normalize_tokens("revenue of 1234.5 million up 12 percent and rising we're pleased")
    record("transcript and ASR spellings normalise identically", float(a == b), 1, 0, " ".join(a))


# ------------------------------------------------------------------ alignment
def _fake_asr(sentences, rng, drop=0.05, insert=0.05, t0=2.0):
    words, t = [], t0
    for s in sentences:
        for w in s.split():
            if rng.random() < insert:
                words.append({"w": "uh", "s": t, "e": t + 0.2}); t += 0.25
            if rng.random() < drop:
                t += 0.3
                continue
            dur = 0.15 + 0.05 * len(w) / 4
            words.append({"w": w, "s": round(t, 3), "e": round(t + dur, 3)})
            t += dur + 0.05
        t += 0.6                                                     # pause between sentences
    return words


def test_alignment_recovers_times():
    rng = np.random.default_rng(3)
    sents = [" ".join(f"w{rng.integers(0, 400)}" for _ in range(rng.integers(6, 20))) for _ in range(150)]
    words = _fake_asr(sents, rng)
    spans = align_sentences([normalize_tokens(s) for s in sents], words)
    # truth: first and last surviving word of each sentence, in order
    k, starts, ends = 0, [], []
    for s in sents:
        toks = set(s.split())
        own = []
        while k < len(words) and (words[k]["w"] in toks or words[k]["w"] == "uh"):
            if words[k]["w"] != "uh":
                own.append(words[k])
            k += 1
        starts.append(own[0]["s"]); ends.append(own[-1]["e"])
    err = np.mean([abs(sp["start"] - s) + abs(sp["end"] - e) for sp, s, e in zip(spans, starts, ends)])
    record("alignment: mean boundary error (s), 5% deletions + 5% insertions", err, 0.0, 0.15)
    record("alignment: share of sentences timed", np.mean([sp["timed"] for sp in spans]), 1.0, 0.05)


def test_alignment_flags_missing_audio():
    sents = ["alpha beta gamma delta epsilon zeta", "this sentence was never spoken at all here", "eta theta iota kappa lambda mu"]
    words = _fake_asr([sents[0], sents[2]], np.random.default_rng(0), drop=0, insert=0)
    spans = align_sentences([normalize_tokens(s) for s in sents], words)
    record("unspoken sentence is not timed", float(not spans[1]["timed"]), 1, 0)
    record("neighbours stay timed", float(spans[0]["timed"] and spans[2]["timed"]), 1, 0)


def test_turn_table():
    """A turn table stands in for a transcript PDF: sections, roles and speaking order survive the round trip."""
    import tempfile

    rows = ["speaker,title,section,text",
            'Operator,,presentation,"Good morning and welcome to the first quarter earnings call."',
            'Jane Roe,Chief Executive Officer,presentation,"Revenue grew nine percent. We remain confident."',
            'Operator,,qa,"Our next question comes from Alex Doe."',
            'Alex Doe,Example Securities,qa,"How should we think about pricing?"',
            'Jane Roe,Chief Executive Officer,qa,"Pricing held up through the quarter."']
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "call.csv")
        open(p, "w", encoding="utf-8").write("\n".join(rows) + "\n")
        tr = load_transcript(p)
    roles = {t.speaker: t.role for t in tr.turns}
    record("every turn is read", len(tr.turns), 5, 0)
    record("the executive title gives CEO", float(roles.get("Jane Roe") == "CEO"), 1, 0)
    record("an affiliation gives analyst", float(roles.get("Alex Doe") == "analyst"), 1, 0)
    record("the operator is recognised", float(roles.get("Operator") == "operator"), 1, 0)
    record("Q&A turns keep their section", sum(t.section == "qa" for t in tr.turns), 3, 0)
    record("speaking order is preserved", float([t.order for t in tr.turns] == [0, 1, 2, 3, 4]), 1, 0)


def test_manifest():
    """A manifest resolves its paths against its own folder, so a corpus travels with its listing."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "audio"), exist_ok=True)
        p = os.path.join(d, "calls.csv")
        open(p, "w", encoding="utf-8").write(
            "audio,transcript,call_id\naudio/a.wav,turns/a.csv,A\n,turns/b.csv,B\n")
        calls = read_manifest(p)
    record("the incomplete row is dropped", len(calls), 1, 0)
    record("the audio path is resolved", float(calls[0][0] == os.path.normpath(os.path.join(d, "audio/a.wav"))), 1, 0)
    record("the transcript path is resolved", float(calls[0][1] == os.path.normpath(os.path.join(d, "turns/a.csv"))), 1, 0)


def test_turn_table_infers_qa():
    """Without a section column the question-and-answer session starts at the operator's first question cue."""
    import tempfile

    rows = ["speaker,title,text",
            'Operator,,"Good morning and welcome."',
            'Jane Roe,Chief Executive Officer,"Revenue grew nine percent."',
            'Operator,,"Our next question comes from Alex Doe."',
            'Alex Doe,Example Securities,"How should we think about pricing?"']
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "call.csv")
        open(p, "w", encoding="utf-8").write("\n".join(rows) + "\n")
        tr = load_transcript(p)
    record("prepared remarks end before the cue", sum(t.section == "presentation" for t in tr.turns), 2, 0)
    record("the cue opens the Q&A", sum(t.section == "qa" for t in tr.turns), 2, 0)


# ------------------------------------------------------------------ prosody
def _glide(sr=16000, dur=2.0, f_start=120.0, f_end=160.0, gap=(0.9, 1.2), seed=0):
    """Harmonic voice whose F0 glides linearly, with a silent gap and weak noise."""
    t = np.arange(int(dur * sr)) / sr
    f = f_start + (f_end - f_start) * t / dur
    phase = 2 * np.pi * np.cumsum(f) / sr
    x = sum(np.sin(k * phase) / k for k in range(1, 15))
    x[(t >= gap[0]) & (t < gap[1])] = 0.0
    return 0.3 * x / np.abs(x).max() + 1e-3 * np.random.default_rng(seed).standard_normal(len(t)), t, f


def test_pitch_harmonic():
    sr = 16000
    t = np.arange(sr) / sr
    x = sum(np.sin(2 * np.pi * 150.0 * k * t) / k for k in range(1, 15))
    _, f0 = pitch_ac(x, sr, time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0)
    record("F0 of a 150 Hz harmonic complex (Hz)", np.median(f0[f0 > 0]), 150.0, 0.75)
    record("voiced share of a periodic signal", np.mean(f0 > 0), 1.0, 0.05)


def test_pitch_step_and_silence():
    """Two segments of known F0 an octave apart, separated by silence: level accuracy, octave robustness,
    and voicing decisions, with no dependency outside this file."""
    sr = 16000
    segs = []
    for f in (110.0, 220.0):
        t = np.arange(int(0.6 * sr)) / sr
        segs.append(sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 15)))
    x = np.concatenate([segs[0], np.zeros(int(0.3 * sr)), segs[1]])
    x = 0.3 * x / np.abs(x).max()
    ts, f0 = pitch_ac(x, sr, time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0)
    lo, hi = f0[(ts < 0.55) & (f0 > 0)], f0[(ts > 0.95) & (f0 > 0)]
    record("F0 of the 110 Hz segment (Hz)", np.median(lo), 110.0, 0.75)
    record("F0 of the 220 Hz segment (Hz)", np.median(hi), 220.0, 1.5)
    record("no octave error across the step", np.median(hi) / np.median(lo), 2.0, 0.05)
    record("silent gap is unvoiced", np.mean(f0[(ts > 0.68) & (ts < 0.82)] > 0), 0.0, 0.05)


def test_pitch_vs_praat():
    try:
        import parselmouth
    except ImportError as e:
        raise Skip(f"Praat (parselmouth) unavailable: {e}")
    x, _, _ = _glide()
    _, ours = pitch_ac(x, 16000, time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0)
    praat = parselmouth.Sound(x, sampling_frequency=16000).to_pitch_ac(
        time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0).selected_array["frequency"]
    record("frame count vs Praat", len(ours), len(praat), 0)
    record("voicing agreement vs Praat", np.mean((ours > 0) == (praat > 0)), 1.0, 0.03)
    both = (ours > 0) & (praat > 0)
    record("mean |F0 - Praat F0| on a 120-160 Hz glide (Hz)", np.mean(np.abs(ours[both] - praat[both])), 0.0, 0.5)


def test_prosody_descriptors():
    x, _, f = _glide(gap=(5.0, 5.0))                    # no gap
    p = prosody(x, 16000)
    st = 12 * np.log2(f / 100.0)
    record("F0 median (st re 100 Hz) of the glide", p["f0_median_st"], np.median(st), 0.1)
    record("F0 SD (st) of the glide", p["f0_sd_st"], np.std(st), 0.1)


# ------------------------------------------------------------------ congruence
def _probs(label, strength=0.8):
    p = {lab: (1 - strength) / 2 for lab in ("positive", "neutral", "negative")}
    p[label] = strength
    return p


def test_cells_and_direction():
    rows = []
    for (t, v) in CELLS:
        r = {f"tx_{k}": x for k, x in _probs(t).items()}
        r.update({f"fv_{k}": x for k, x in _probs(v).items()})
        rows.append(r)
    d = add_congruence(pd.DataFrame(rows))
    record("nine distinct cells", d["cell"].nunique(), 9, 0)
    record("aligned only on the diagonal", float(list(d["aligned"]) == [t == v for t, v in CELLS]), 1, 0)
    rev = d.loc[d["reversal"], "cell"].tolist()
    record("polarity reversals are C3 and C7", float(sorted(rev) == ["C3", "C7"]), 1, 0)
    record("gap = voice_net - text_net for C3", d.loc[d["cell"] == "C3", "gap"].iloc[0], -1.4, 1e-9)


def test_independence_vs_scipy():
    from scipy.stats import chi2_contingency

    t = pd.DataFrame([[120, 40, 10], [300, 900, 150], [15, 60, 90]], index=["positive", "neutral", "negative"],
                     columns=["positive", "neutral", "negative"])
    ours = independence(t)
    chi2, p, dof, _ = chi2_contingency(t.to_numpy(), correction=False)
    record("chi-square vs scipy", ours["chi2"], chi2, 1e-8)
    record("degrees of freedom", ours["dof"], dof, 0)
    n = t.to_numpy(float); po = np.trace(n) / n.sum(); pe = (n.sum(1) * n.sum(0)).sum() / n.sum() ** 2
    record("Cohen's kappa", ours["kappa"], (po - pe) / (1 - pe), 1e-12)
    e = n.sum(1, keepdims=True) @ n.sum(0, keepdims=True) / n.sum()
    r00 = (n[0, 0] - e[0, 0]) / np.sqrt(e[0, 0] * (1 - n[0].sum() / n.sum()) * (1 - n[:, 0].sum() / n.sum()))
    record("adjusted residual (Agresti) for cell (1,1)", ours["adj_residuals"].iloc[0, 0], r00, 1e-10)


def test_independent_channels_give_no_association():
    rng = np.random.default_rng(1)
    n = 20000
    t = rng.choice(["positive", "neutral", "negative"], n, p=[0.3, 0.6, 0.1])
    v = rng.choice(["positive", "neutral", "negative"], n, p=[0.2, 0.7, 0.1])
    d = pd.DataFrame({"text_label": t, "voice_label": v})
    res = independence(crosstab3(d))
    record("Cramer's V near 0 when channels are independent", res["cramers_v"], 0.0, 0.03)
    record("kappa near 0 when channels are independent", res["kappa"], 0.0, 0.03)


def test_cluster_bootstrap_covers_truth():
    rng = np.random.default_rng(2)
    calls = np.repeat(np.arange(300), 50)
    p_call = rng.beta(4, 12, 300)[calls]                       # clustered binary outcome, mean 0.25
    y = rng.random(len(calls)) < p_call
    lo, hi = cluster_bootstrap(pd.Series(y), pd.Series(calls), reps=2000)
    record("cluster-bootstrap interval contains the population mean 0.25", float(lo <= 0.25 <= hi), 1, 0, f"[{lo:.3f}, {hi:.3f}]")
    naive = 1.96 * np.sqrt(y.mean() * (1 - y.mean()) / len(y))
    record("cluster interval wider than the naive interval", float((hi - lo) / 2 > naive), 1, 0)


def test_same_text_dispersion():
    rng = np.random.default_rng(4)
    texts = ["Thank you.", "Good morning, everyone.", "Thanks for the question."]
    rows = [{"text": texts[i % 3], "voice_net": rng.normal(0.1 * (i % 3), 0.3), "voice_label": "neutral",
             "speaker": f"s{i % 7}"} for i in range(300)]
    _, summ = same_text_dispersion(pd.DataFrame(rows), min_count=5)
    between = np.var([0.0, 0.1, 0.2])                       # variance of the three text means
    truth = 0.3 ** 2 / (0.3 ** 2 + between)                  # 0.931
    record("identical texts: within-text share of vocal variance", summ["within_text_share_of_voice_variance"], truth, 0.03)


def test_same_text_within_speaker():
    rng = np.random.default_rng(5)
    effects = np.linspace(-1, 1, 40)
    effects = effects / effects.std() * 0.2                   # speaker variance exactly 0.04
    rows = [{"text": t, "speaker": f"s{k}", "voice_net": effects[k] + rng.normal(0, 0.1), "voice_label": "neutral"}
            for t in ("Thank you.", "Good morning.", "Next question, please.") for k in range(40) for _ in range(5)]
    _, summ = same_text_dispersion(pd.DataFrame(rows), min_count=5)
    record("same speaker, same words: within share of vocal variance", summ["within_text_speaker_share_of_voice_variance"],
           0.01 / 0.05, 0.03)


def test_icc_and_eta2():
    rng = np.random.default_rng(6)
    sizes = rng.integers(10, 41, 400)                           # unequal group sizes
    groups = np.repeat(np.arange(400), sizes)
    v = rng.normal(0, 0.5, 400)[groups] + rng.normal(0, np.sqrt(0.75), len(groups))
    record("ICC(1): between-group share 0.25 recovered", icc1(v, groups), 0.25, 0.03)
    record("eta squared: between / total sum of squares", eta2([1, 2, 3, 4, 5, 6], list("aaabbb")), 13.5 / 17.5, 1e-9)


def test_speaker_relative():
    d = pd.DataFrame({"call_id": ["a"] * 6, "speaker": ["x"] * 6, "voice_net": [0.5, 0.5, 0.5, 0.9, 0.1, 0.5]})
    lab = speaker_relative(d).tolist()
    record("relative labels follow the speaker's own mean", float(lab[3] == "positive" and lab[4] == "negative"), 1, 0, str(lab))


# ------------------------------------------------------------------ runner
def run_tests():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    status, t0 = [], time.time()
    for name, fn in tests:
        try:
            fn()
            status.append((name, "PASS", ""))
        except Skip as s:
            status.append((name, "SKIP", str(s)))
        except AssertionError as e:
            status.append((name, "FAIL", str(e)))
        except Exception as e:                                          # noqa: BLE001
            status.append((name, "ERROR", repr(e)))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(REPORT).drop(columns="note").to_string(index=False))
    print()
    for s in status:
        print(f"{s[1]:5s} {s[0]} {s[2]}")
    n_fail = sum(s[1] in ("FAIL", "ERROR") for s in status)
    print(f"\n{len(status) - n_fail}/{len(status)} test functions passed or skipped in {time.time() - t0:.1f}s")
    return n_fail


# ==================================================================================================
# Batch runner  (repository: scripts/run_sample.py)
# ==================================================================================================
# Process every call in a sample folder (one sub-folder per company, <call>.mp3 next to <call>.pdf).
#
# Calls are visited round-robin across companies (each company's calls in date order), so that a partial
# run is a balanced sample. Finished calls are skipped, so the script can be stopped and restarted at any
# time; failures are logged and the run continues.
#
# usage: python scripts/run_sample.py --root SAMPLE_FOLDER --out outputs [--limit N] [--company NAME]
#                                     [--match REGEX]   (e.g. Earnings_Call: skip investor-conference talks)
#                                     [--shard i/n]     (n parallel workers on one GPU, worker i)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")          # Windows caches without symlinks


def discover(root: str, company: str | None = None, match: str | None = None) -> list[tuple[str, str]]:
    per_company = []
    for comp in sorted(os.listdir(root)):
        d = os.path.join(root, comp)
        if not os.path.isdir(d) or (company and comp != company):
            continue
        calls = []
        for f in os.listdir(d):
            if f.lower().endswith(".mp3") and (not match or re.search(match, f)):
                pdf = os.path.join(d, f[:-4] + ".pdf")
                if os.path.exists(pdf):
                    mp3 = os.path.join(d, f)
                    calls.append((call_info(mp3).get("call_date", ""), mp3, pdf))
        per_company.append([(mp3, pdf) for _, mp3, pdf in sorted(calls)])
    order = []
    for batch in itertools.zip_longest(*per_company):
        order += [c for c in batch if c is not None]
    return order


def run_sample(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "outputs"))
    ap.add_argument("--limit", type=int, default=0, help="stop after this many calls (0 = all)")
    ap.add_argument("--company", default=None)
    ap.add_argument("--match", default=None, help="regular expression the file name must contain")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th call starting at i (parallel workers)")
    args = ap.parse_args(argv)
    shard, n_shards = (int(x) for x in args.shard.split("/"))

    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    log_path = os.path.join(args.out, "logs", f"run_{shard}of{n_shards}.log")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    calls = discover(args.root, args.company, args.match)
    if args.limit:
        calls = calls[:args.limit]
    calls = calls[shard::n_shards]
    models = Models(device=args.device)
    done = failed = 0
    log(f"start: {len(calls)} calls")
    for k, (mp3, pdf) in enumerate(calls, 1):
        try:
            process_call(mp3, pdf, args.out, models, log=lambda m: log(f"[{k}/{len(calls)}] {m}"))
            done += 1
        except Exception as e:                                         # noqa: BLE001
            failed += 1
            log(f"[{k}/{len(calls)}] FAILED {os.path.basename(mp3)}: {e!r}")
            with open(os.path.join(args.out, "logs", f"errors_{shard}of{n_shards}.log"), "a", encoding="utf-8") as f:
                f.write(f"{mp3}\n{traceback.format_exc()}\n")
        with open(os.path.join(args.out, "logs", f"progress_{shard}of{n_shards}.json"), "w", encoding="utf-8") as f:
            json.dump({"total": len(calls), "processed": done, "failed": failed, "position": k,
                       "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
    log(f"end: {done} processed, {failed} failed")


# ==================================================================================================
# LLM rater pass  (repository: scripts/score_llm.py)
# ==================================================================================================
# Second textual rater: add Qwen3-4B-Instruct sentiment probabilities to every saved sentence table.
#
# Runs after (or between) speech passes; tables that already have llm_* columns are skipped.
# usage: python scripts/score_llm.py --out outputs

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")          # Windows caches without symlinks


def run_llm(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "outputs"))
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    args = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(args.out, "sentences", "*.csv.gz")))
    scorer = None
    for k, path in enumerate(files, 1):
        df = pd.read_csv(path)
        if "llm_positive" in df.columns:
            continue
        scorer = scorer or LLMScorer(args.device)
        t0 = time.time()
        scores = pd.DataFrame(scorer(df["text"].astype(str).tolist()))
        pd.concat([df, scores], axis=1).to_csv(path + ".tmp", index=False, compression="gzip")
        os.replace(path + ".tmp", path)
        print(f"[{k}/{len(files)}] {os.path.basename(path)}: {len(df)} sentences in {time.time() - t0:.0f}s", flush=True)


# ==================================================================================================
# Command line
# ==================================================================================================
def _cli(argv=None):
    ap = argparse.ArgumentParser(description="Same Words, Different Voice: text-voice congruence in earnings calls")
    ap.add_argument("--test", action="store_true", help="run the verification suite (no GPU needed)")
    ap.add_argument("--call", nargs=2, metavar=("AUDIO", "TRANSCRIPT"),
                    help="process one call: recording plus transcript PDF or turn table")
    ap.add_argument("--sample", metavar="ROOT", help="process every call under ROOT (one folder per company)")
    ap.add_argument("--manifest", metavar="CSV",
                    help="process every call listed in a CSV with the columns audio and transcript")
    ap.add_argument("--limit", type=int, default=0, help="with --sample: stop after this many calls")
    ap.add_argument("--match", default=None, help="with --sample: regular expression the file name must contain")
    ap.add_argument("--llm", action="store_true", help="add the LLM textual rater to all saved tables")
    ap.add_argument("--report", action="store_true", help="write tables, figures, and summary.json")
    ap.add_argument("--out", default="vtc_outputs", help="output folder (default: vtc_outputs)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    a = ap.parse_args(argv)
    a.device = a.device or default_device()
    if not (a.test or a.call or a.sample or a.manifest or a.llm or a.report):
        ap.print_help()
        return 0
    status = 0
    if a.test:
        status = run_tests()
    if a.call:
        process_call(a.call[0], a.call[1], a.out, Models(device=a.device))
    if a.sample:
        run_sample(["--root", a.sample, "--out", a.out, "--limit", str(a.limit), "--device", a.device]
                   + (["--match", a.match] if a.match else []))
    if a.manifest:
        calls = read_manifest(a.manifest)
        models = Models(device=a.device)
        for k, (audio, transcript) in enumerate(calls, 1):
            try:
                process_call(audio, transcript, a.out, models)
                print(f"[{k}/{len(calls)}] {os.path.basename(audio)}", flush=True)
            except Exception as e:                                      # noqa: BLE001
                status = 1
                print(f"[{k}/{len(calls)}] FAILED {os.path.basename(audio)}: {e!r}", flush=True)
    if a.llm:
        run_llm(["--out", a.out, "--device", a.device])
    if a.report:
        res = report.run(a.out)
        print(json.dumps({k: v for k, v in res.items() if not isinstance(v, (dict, list))}, indent=1))
    return status


if __name__ == "__main__":
    sys.exit(_cli())
