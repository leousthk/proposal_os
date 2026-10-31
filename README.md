# Same Words, Different Voice

Sentence-level measurement of **text–voice emotional congruence** in earnings conference calls.

For every sentence an executive speaks, this pipeline measures the emotional tone of the **words** and the
emotional tone of the **voice**, independently of each other, and assigns the sentence to one of nine
text–voice configurations. It is the research artifact accompanying the Stage 1 proposal
*Same Words, Different Voice: AI-Enabled Measurement of Text–Voice Emotional Congruence in Executive
Communication.*

Everything is in one runnable file, `voice_text_congruence.py`, which contains every module, the
verification suite, and the command-line interface. All models are open-weight and pinned to published
revisions.

---

## The nine configurations

Textual tone `T` and vocal tone `V` each take the value positive (`+`), neutral (`0`), or negative (`−`).
Crossing them yields nine configurations; the diagonal is congruent.

|                    | Voice positive | Voice neutral | Voice negative |
| ------------------ | -------------- | ------------- | -------------- |
| **Words positive** | `T+V+`         | `T+V0`        | `T+V−`         |
| **Words neutral**  | `T0V+`         | `T0V0`        | `T0V−`         |
| **Words negative** | `T−V+`         | `T−V0`        | `T−V−`         |

---

## Quick start: verify the installation in under a minute

The verification suite needs no GPU, no model downloads, and no data.

```bash
python -m pip install numpy scipy pandas pymupdf
python voice_text_congruence.py --test
```

Expected output: `25/25 test functions passed or skipped`. The three direct comparisons against Praat are
skipped unless `parselmouth` is installed (`pip install praat-parselmouth`); everything else runs
everywhere.

## Full installation

```bash
python -m pip install -r requirements.txt
```

A CUDA GPU is recommended but not required: `--device` defaults to `cuda` when a CUDA device is actually
available and to `cpu` otherwise, so the pipeline runs on any machine, roughly an order of magnitude more
slowly on CPU. Models download from their published repositories at the pinned revisions on first use
(about 5 GB in total, or 13 GB with the optional second textual rater).

---

## Commands

```bash
# verification suite (no GPU, no data)
python voice_text_congruence.py --test

# one call: a recording plus a transcript PDF or a turn table
python voice_text_congruence.py --call CALL.mp3 CALL.pdf       --out OUT
python voice_text_congruence.py --call CALL.wav turns.csv      --out OUT

# every call under a folder laid out as ROOT/<company>/<call>.{mp3,pdf}
python voice_text_congruence.py --sample ROOT --match Earnings_Call --out OUT

# every call listed in a CSV with the columns audio and transcript
python voice_text_congruence.py --manifest examples/example_manifest.csv --out OUT

# optional second textual rater (Qwen3), added to every saved table
python voice_text_congruence.py --llm --out OUT

# tables, figures, and summary.json from everything processed so far
python voice_text_congruence.py --report --out OUT
```

Batch runs are resumable: a call whose sentence table already exists is skipped, so an interrupted run
continues where it stopped. `--shard i/n` splits a sample across parallel workers writing to one `--out`.

---

## Input formats

**Transcript PDF.** An S&P Global Market Intelligence transcript. The parser reads the layout rather than
font names (24 pt bold section headers, 10 pt bold speaker names, italic titles, regular body text), so it
handles the current template, the older all-Verdana template, and Capital IQ reprints.

**Turn table** — `.csv`, `.tsv`, `.json` (records) or `.jsonl`, one row per speaker turn in speaking order.
This is the route for openly licensed corpora and for transcripts from other vendors.

| Column    | Required | Meaning                                                                        |
| --------- | -------- | ------------------------------------------------------------------------------ |
| `speaker` | yes      | Speaker name, or `Operator`                                                    |
| `text`    | yes      | What was said in that turn                                                     |
| `section` | no       | `presentation` or `qa`; `Prepared remarks`, `Q&A`, `Question and Answer` etc. are recognised. Omit it and the Q&A is inferred from the operator's first question cue |
| `title`   | no       | Job title or, for an analyst, the affiliation; used to infer the role          |
| `group`   | no       | `executives` or `analysts`                                                     |
| `role`    | no       | `CEO`, `CFO`, `IR`, `other_executive`, `analyst`, `operator`, `other`          |
| `order`   | no       | Sort key, if the rows are not already in speaking order                        |

Supplying `role` or `group` where the source has them avoids the title heuristic. See
`examples/example_turn_table.csv`.

**Manifest** — a CSV with the columns `audio` and `transcript`, one row per call, paths absolute or
relative to the manifest. Further columns (`call_id`, `company`, `call_date`, `gvkey`, `permno`, `ticker`, …)
are ignored by the pipeline and travel with the data for the researcher's own merge. See
`examples/example_manifest.csv`.

Audio: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.mp4`, decoded to 16 kHz mono.

---

## Reproducing the measurement on openly licensed data

The calls the instrument was developed on are licensed and cannot be redistributed. The measurement itself
reproduces on **Earnings-21** and **Earnings-22**, two openly licensed corpora of earnings-call recordings
with human transcripts (CC BY-SA 4.0):

* Earnings-21 — <https://github.com/revdotcom/speech-datasets/tree/main/earnings21>
* Earnings-22 — <https://github.com/revdotcom/speech-datasets/tree/main/earnings22>

Their transcripts are not S&P PDFs, so convert them to turn tables first:

```bash
python examples/make_turn_table.py EARNINGS22/transcripts/4320211.nlp --out turns/4320211.csv
python voice_text_congruence.py --call EARNINGS22/media/4320211.mp3 turns/4320211.csv --out OUT
python voice_text_congruence.py --report --out OUT
```

These corpora carry no speaker titles, so roles come from the operator's question cues and from who speaks
in the prepared remarks (`fallback_group`); add a `role` or `title` column where you know them. Figures
computed on this sample will differ from those in the proposal, which uses a different set of calls — what
reproduces is the instrument and its behaviour, not the point estimates.

---

## Outputs

For each call the pipeline writes one row per sentence to `OUT/sentences/<call>.csv.gz` and a metadata
record to `OUT/meta/<call>.json` holding counts, alignment summaries, runtime, library versions, and the
model commits actually used.

| Variable group                                                      | Contents                                                                              |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `call_id, company, call_date, section, speaker, title, role`         | Call and speaker identifiers; `section` is prepared remarks or Q&A                     |
| `text, n_words`                                                     | Transcript sentence and its token count                                                |
| `start, end, duration, coverage, align_sim, timed`                  | Aligned time span in seconds, share of tokens matched, similarity, retention flag      |
| `tx_positive, tx_neutral, tx_negative`                              | FinBERT probabilities of textual tone                                                  |
| `llm_positive, llm_neutral, llm_negative, llm_mass`                 | Qwen3 probabilities of textual tone, and the probability mass on the label words       |
| `fv_positive, fv_neutral, fv_negative`                              | FinVoc2Vec probabilities of vocal tone                                                 |
| `ser_arousal, ser_dominance, ser_valence`                           | Dimensional vocal emotion, approximately 0 to 1                                        |
| `f0_median_st, f0_sd_st, intensity_sd_db, voiced_frac, tokens_per_sec` | Interpretable prosody                                                               |
| `text_label, voice_label, text_net, voice_net, gap, direction, cell` | Channel tones, net tones, gap, direction, and configuration, stored as `C1`–`C9` (`C1` = `T+V+`, `C9` = `T−V−`) |

---

## Models

| Role                          | Model                                                 | Parameters | License        |
| ----------------------------- | ----------------------------------------------------- | ---------- | -------------- |
| Speech recognition            | Whisper `large-v3-turbo` (faster-whisper conversion)  | 0.8 B      | MIT            |
| Textual tone, primary         | FinBERT (`yiyanghkust/finbert-tone`)                  | 0.11 B     | see model card |
| Textual tone, second rater    | `Qwen3-4B-Instruct-2507`                              | 4.0 B      | Apache-2.0     |
| Vocal tone, primary           | FinVoc2Vec (`waiv/FinVoc2Vec`)                        | 0.3 B      | CC BY-NC 4.0   |
| Arousal, dominance, valence   | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | 0.2 B    | CC BY-NC-SA 4.0 |

---

## Licensing and ethics

The code in this repository is released under the MIT License (see `LICENSE`).

**The models are not.** FinVoc2Vec (CC BY-NC 4.0) and the dimensional emotion model (CC BY-NC-SA 4.0)
restrict use to non-commercial research. Running the full pipeline is therefore subject to those terms,
whatever this repository's own license permits. Check each model card before use.

Transcripts and recordings from commercial vendors are used under the provider's terms and are not
redistributed here.

The instrument characterizes communication, not persons. It produces no individual diagnosis of emotion,
stress, or deception, and no named rankings of executives. Only sentence-level scores and prosodic
summaries are stored; no speaker embeddings are retained.

---

## Repository layout

```
voice_text_congruence.py      every module, the verification suite, and the command line
requirements.txt              dependencies
examples/
  example_turn_table.csv      a transcript in turn-table form
  example_manifest.csv        a batch listing
  make_turn_table.py          converts Earnings-21/22 .nlp transcripts to turn tables
LICENSE
```

Section headers inside `voice_text_congruence.py` name the repository module each part corresponds to
(`transcripts`, `sentences`, `audio`, `asr`, `align`, `text_models`, `voice_models`, `pitch`, `acoustics`,
`congruence`, `report`, `pipeline`), so the `module.function` references in the technical appendix resolve
here as well.
