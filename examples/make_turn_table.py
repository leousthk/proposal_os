#!/usr/bin/env python3
"""Convert an Earnings-21 / Earnings-22 transcript into the turn table that the pipeline reads.

Those two corpora are openly licensed (CC BY-SA 4.0) recordings of earnings calls with human transcripts,
so they are the route to reproducing the measurement without licensed vendor PDFs:

    Earnings-21  https://github.com/revdotcom/speech-datasets/tree/main/earnings21
    Earnings-22  https://github.com/revdotcom/speech-datasets/tree/main/earnings22

Their transcripts come as token-per-line ".nlp" files with a header naming the columns; this script groups
consecutive tokens of the same speaker into turns, reattaches punctuation, and writes one row per turn.

    python examples/make_turn_table.py EARNINGS22/transcripts/4320211.nlp --out turns/4320211.csv
    python voice_text_congruence.py --call EARNINGS22/media/4320211.mp3 turns/4320211.csv --out OUT

Several files at once, then one batch run:

    python examples/make_turn_table.py EARNINGS22/transcripts/*.nlp --out-dir turns --manifest calls.csv
    python voice_text_congruence.py --manifest calls.csv --out OUT

These corpora carry no speaker titles, so the pipeline infers roles from the operator's question cues and
from who speaks in the prepared remarks. Where you know a speaker's role, add a "role" or "title" column
to the turn table: that replaces the heuristic and makes the role assignment exact.

The delimiter and column names are detected from the header, so a minor change in the corpus layout does
not break the conversion. If the required columns are absent the script says which ones it looked for.
"""
import argparse
import csv
import glob
import os
import re
import sys

# These corpora identify speakers by number, so the conference-call operator has to be recognised from
# what the speaker says. Finding the operator matters: the pipeline reads the prepared-remarks/Q&A
# boundary from the operator's first question cue, and treats the operator's own turns as procedural.
OPERATOR_CUE = re.compile(
    r"operator instructions|question (comes|will come|is coming) from|next question|"
    r"welcome to the .{0,60}(call|conference)|(ladies and gentlemen|thank you for (standing by|joining))|"
    r"press star|the floor is (now )?open|that concludes|turn the (call|floor|conference) over", re.I)

TOKEN_KEYS = ("token", "word", "text")
SPEAKER_KEYS = ("speaker", "spk", "speaker_id")
PUNCT_KEYS = ("punctuation", "punct")
# A turn boundary without a speaker change: the corpora mark segments this way in some releases.
SEGMENT_KEYS = ("segment", "turn", "utterance_id")


def _pick(fields, candidates):
    low = {f.strip().lower(): f for f in fields}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def read_nlp(path):
    """Rows of an .nlp file as dicts, with the delimiter taken from the header line."""
    with open(path, encoding="utf-8", newline="") as fh:
        head = fh.readline()
        if not head:
            raise SystemExit(f"{path}: file is empty")
        delim = max(("|", "\t", ","), key=head.count)
        if head.count(delim) == 0:
            raise SystemExit(f"{path}: cannot find a delimiter in the header line: {head[:120]!r}")
        fh.seek(0)
        return list(csv.DictReader(fh, delimiter=delim)), delim


def to_turns(rows, fields):
    tok_k = _pick(fields, TOKEN_KEYS)
    spk_k = _pick(fields, SPEAKER_KEYS)
    pun_k = _pick(fields, PUNCT_KEYS)
    seg_k = _pick(fields, SEGMENT_KEYS)
    if not tok_k or not spk_k:
        raise SystemExit(
            f"expected a token column ({' / '.join(TOKEN_KEYS)}) and a speaker column "
            f"({' / '.join(SPEAKER_KEYS)}); the file has: {', '.join(fields)}")

    turns, cur, key = [], [], None
    for r in rows:
        tok = (r.get(tok_k) or "").strip()
        if not tok:
            continue
        spk = (r.get(spk_k) or "").strip() or "Unknown"
        seg = (r.get(seg_k) or "").strip() if seg_k else ""
        k = (spk, seg)
        if k != key and cur:
            turns.append((key[0], cur))
            cur = []
        key = k
        cur.append(tok + ((r.get(pun_k) or "").strip() if pun_k else ""))
    if cur:
        turns.append((key[0], cur))
    return [(spk, " ".join(toks)) for spk, toks in turns if toks]


def find_operator(turns):
    """The speaker who reads the procedural script, or None.

    Scored by how many of their turns carry an operator cue. Requiring at least two cues, and more than
    any other speaker, keeps an executive who happens to say "next question" from being mistaken for the
    operator. Returns the raw speaker id so the caller can rename it."""
    hits = {}
    for spk, text in turns:
        if OPERATOR_CUE.search(text):
            hits[spk] = hits.get(spk, 0) + 1
    if not hits:
        return None
    best, n = max(hits.items(), key=lambda kv: kv[1])
    others = [v for k, v in hits.items() if k != best]
    return best if n >= 2 and n > max(others, default=0) else None


def speaker_name(raw, operator_id=None):
    """Numeric speaker ids become "Speaker 3"; the operator gets the name the pipeline recognises."""
    s = str(raw).strip()
    if operator_id is not None and s == str(operator_id).strip():
        return "Operator"
    if s.lower() in ("operator", "moderator", "host"):
        return "Operator"
    return f"Speaker {s}" if s.isdigit() else s


def convert(src, dst):
    rows, _ = read_nlp(src)
    if not rows:
        raise SystemExit(f"{src}: no token rows found")
    turns = to_turns(rows, list(rows[0].keys()))
    op = find_operator(turns)
    if op is None:
        print(f"  {os.path.basename(src)}: no operator identified; the Q&A boundary and the roles will "
              f"have to come from a section, role or title column", file=sys.stderr)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["order", "speaker", "text"])
        for i, (spk, text) in enumerate(turns):
            w.writerow([i, speaker_name(spk, op), text])
    return len(turns)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nlp", nargs="+", help="one or more .nlp transcripts (shell globs are expanded)")
    ap.add_argument("--out", help="output CSV, for a single input")
    ap.add_argument("--out-dir", help="output folder, for several inputs")
    ap.add_argument("--media-dir", help="folder holding the recordings, to fill in the manifest")
    ap.add_argument("--manifest", help="also write a manifest CSV listing audio and transcript per call")
    a = ap.parse_args(argv)

    paths = [p for pat in a.nlp for p in (glob.glob(pat) or [pat])]
    if not paths:
        raise SystemExit("no input files")
    if len(paths) > 1 and not a.out_dir:
        raise SystemExit("several inputs: pass --out-dir")
    if len(paths) == 1 and not (a.out or a.out_dir):
        raise SystemExit("pass --out FILE or --out-dir DIR")

    written = []
    for src in paths:
        stem = os.path.splitext(os.path.basename(src))[0]
        dst = a.out if (a.out and len(paths) == 1) else os.path.join(a.out_dir, stem + ".csv")
        n = convert(src, dst)
        written.append((stem, dst))
        print(f"{src} -> {dst}  ({n} turns)")

    if a.manifest:
        media = a.media_dir or os.path.join(os.path.dirname(os.path.abspath(paths[0])), "..", "media")
        base = os.path.dirname(os.path.abspath(a.manifest))
        with open(a.manifest, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["audio", "transcript", "call_id"])
            for stem, dst in written:
                hit = ""
                for ext in (".mp3", ".wav", ".m4a", ".flac"):
                    cand = os.path.join(media, stem + ext)
                    if os.path.exists(cand):
                        hit = os.path.relpath(cand, base)
                        break
                if not hit:
                    print(f"  no recording found for {stem}; fill in its audio path by hand", file=sys.stderr)
                w.writerow([hit, os.path.relpath(dst, base), stem])
        print(f"manifest -> {a.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
