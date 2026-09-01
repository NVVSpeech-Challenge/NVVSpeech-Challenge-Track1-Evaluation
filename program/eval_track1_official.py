#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVVSpeech Challenge @ ISCSLP 2026
Track 1 Official Evaluation Script

Protocol version: 1.0.0

Usage
-----
python eval_track1_official.py \
    --reference hidden_reference.jsonl \
    --submission team_submission.jsonl \
    --output-json result.json

Reference JSONL schema (one object per line)
------------------------------------------------
{
  "utt_id": "example_001",
  "language": "en",                 # exactly "en" or "zh"
  "text_with_nvvs": "That was [laugh] surprising."
}

Submission JSONL schema (one object per line)
-------------------------------------------------
{
  "utt_id": "example_001",
  "text_with_nvvs": "That was [laugh] surprising."
}

Frozen official protocol
------------------------
1. Valid NVV tags are restricted to the 16 canonical labels in VALID_TAGS.
2. English lexical units are normalized words; Mandarin lexical units are
   normalized characters. NVV tags are atomic tokens for tagged WER/CER.
3. Predicted tag positions are mapped to reference-text coordinates by a
   deterministic minimum-edit lexical alignment. A tag position is the number
   of lexical units before that tag; tags themselves never occupy lexical
   positions.
4. F1_micro pools all events within a language. A true positive requires the
   same tag and an aligned position within the frozen collar: 2 words (EN),
   5 characters (ZH). Matching is one-to-one and optimizes, in order,
   maximum TP count and minimum total absolute position error.
5. mNTD is computed per reference utterance containing at least one NVV,
   then macro-averaged within a language. For each utterance, a deterministic
   monotone minimum-cost event alignment is used. Same-tag pairs cost
   min(|p-g|/L, 1); different-tag pairs and unmatched events cost 1. The
   utterance value is divided by max(#reference events, #predicted events).
   Thus mNTD is always in [0, 1] and jointly penalizes wrong tags, missing
   tags, extra tags, wrong event order, and placement errors.
6. Tagged-transcript error is corpus-level tag-aware WER (EN) / CER (ZH):
   total edit distance divided by total number of reference tagged tokens.
7. Track1Score_lang = 100 * [0.70 * F1_micro + 0.20 * (1 - mNTD)
                              + 0.10 * (1 - min(Err_tagged, 1))]
   FinalTrack1Score = (Track1Score_ZH + Track1Score_EN) / 2.

The script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Frozen public protocol configuration
# ---------------------------------------------------------------------------
PROTOCOL_NAME = "NVVSpeech Challenge Track 1 Official Evaluation"
PROTOCOL_VERSION = "1.0.0"

ID_KEY_DEFAULT = "utt_id"
LANGUAGE_KEY_DEFAULT = "language"
TEXT_KEY_DEFAULT = "text_with_nvvs"

VALID_TAGS: Tuple[str, ...] = (
    "breath",
    "sniff",
    "laugh",
    "cry",
    "cough",
    "throat clearing",
    "sneeze",
    "sigh",
    "gasp",
    "snore",
    "yawn",
    "hum",
    "moan",
    "hiss",
    "lipsmack",
    "burp",
)
VALID_TAG_SET = frozenset(VALID_TAGS)

# Collar thresholds, frozen for protocol v1.0.0.
F1_COLLAR = {"en": 2, "zh": 5}

# Tag parser. Nested brackets are intentionally rejected during validation.
TAG_RE = re.compile(r"\[([^\[\]]*)\]")
EN_WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", flags=re.UNICODE)

# Map common typographic apostrophes to ASCII apostrophe before tokenization.
APOSTROPHE_TRANSLATION = str.maketrans({
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u201b": "'",  # single high-reversed-9 quotation mark
    "\u2032": "'",  # prime
    "\uff07": "'",  # fullwidth apostrophe
})


# ---------------------------------------------------------------------------
# Exceptions and data classes
# ---------------------------------------------------------------------------
class EvaluationError(Exception):
    """Base class for evaluator failures."""

    default_code = "EVALUATION_ERROR"

    def __init__(self, message: str, *, code: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code or self.default_code


class ValidationError(EvaluationError):
    """Raised when a reference or submission file violates the public schema."""

    default_code = "VALIDATION_ERROR"


@dataclass(frozen=True)
class Event:
    """A canonical NVV event anchored at a lexical boundary."""

    tag: str
    position: int
    order: int


@dataclass(frozen=True)
class ParsedTranscript:
    """Normalized transcript representation for one utterance."""

    lexical_tokens: Tuple[str, ...]
    tagged_tokens: Tuple[str, ...]
    events: Tuple[Event, ...]


@dataclass(frozen=True)
class ReferenceRecord:
    utt_id: str
    language: str
    text: str


@dataclass(frozen=True)
class SubmissionRecord:
    utt_id: str
    text: str


# ---------------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------------
def _read_jsonl(path: str, role: str) -> List[Tuple[int, Mapping[str, Any]]]:
    """Read JSONL safely and return (1-based line number, object) tuples."""
    records: List[Tuple[int, Mapping[str, Any]]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    # Blank lines are benign and common in hand-edited JSONL.
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValidationError(
                        f"{role} file {path!r}, line {line_no}: invalid JSON: {exc.msg}",
                        code="INVALID_JSON",
                    ) from exc
                if not isinstance(value, dict):
                    raise ValidationError(
                        f"{role} file {path!r}, line {line_no}: each JSONL line must be an object.",
                        code="INVALID_JSONL_RECORD",
                    )
                records.append((line_no, value))
    except OSError as exc:
        raise EvaluationError(
            f"Unable to read {role} file {path!r}: {type(exc).__name__}",
            code="INPUT_FILE_READ_ERROR",
        ) from exc

    if not records:
        raise ValidationError(
            f"{role} file {path!r} contains no JSONL records.",
            code="EMPTY_JSONL",
        )
    return records


def _require_string(
    record: Mapping[str, Any],
    key: str,
    role: str,
    path: str,
    line_no: int,
    *,
    nonempty: bool = False,
) -> str:
    if key not in record:
        raise ValidationError(
            f"{role} file {path!r}, line {line_no}: missing required field {key!r}.",
            code="MISSING_REQUIRED_FIELD",
        )
    value = record.get(key)
    if not isinstance(value, str):
        raise ValidationError(
            f"{role} file {path!r}, line {line_no}: field {key!r} must be a string.",
            code="INVALID_FIELD_TYPE",
        )
    if nonempty and not value.strip():
        raise ValidationError(
            f"{role} file {path!r}, line {line_no}: field {key!r} must not be empty.",
            code="EMPTY_REQUIRED_FIELD",
        )
    return value


def _normalize_language(value: str, role: str, path: str, line_no: int) -> str:
    language = value.strip().lower()
    if language not in {"en", "zh"}:
        raise ValidationError(
            f"{role} file {path!r}, line {line_no}: language must be exactly 'en' or 'zh', "
            f"got {value!r}.",
            code="INVALID_LANGUAGE",
        )
    return language


def load_reference(
    path: str,
    *,
    id_key: str,
    language_key: str,
    text_key: str,
) -> Dict[str, ReferenceRecord]:
    """Load the hidden reference JSONL and validate unique IDs/languages."""
    loaded = _read_jsonl(path, "Reference")
    result: Dict[str, ReferenceRecord] = {}

    for line_no, obj in loaded:
        utt_id = _require_string(
            obj, id_key, "Reference", path, line_no, nonempty=True
        ).strip()
        language_raw = _require_string(obj, language_key, "Reference", path, line_no)
        text = _require_string(obj, text_key, "Reference", path, line_no)
        language = _normalize_language(language_raw, "Reference", path, line_no)

        if utt_id in result:
            raise ValidationError(
                f"Reference file {path!r}, line {line_no}: duplicate utt_id detected.",
                code="DUPLICATE_UTT_ID",
            )
        # Parse once here so malformed reference labels are caught before scoring.
        parse_tagged_transcript(
            text,
            language,
            context=f"Reference record at line {line_no}",
        )
        result[utt_id] = ReferenceRecord(utt_id=utt_id, language=language, text=text)

    languages = {record.language for record in result.values()}
    missing = {"en", "zh"} - languages
    if missing:
        raise ValidationError(
            "Reference file must contain both English and Mandarin Chinese records for "
            f"bilingual ranking. Missing: {', '.join(sorted(missing))}.",
            code="REFERENCE_LANGUAGE_COVERAGE_ERROR",
        )
    return result


def load_submission(
    path: str,
    *,
    id_key: str,
    text_key: str,
) -> Dict[str, SubmissionRecord]:
    """Load a participant submission JSONL and validate unique IDs."""
    loaded = _read_jsonl(path, "Submission")
    result: Dict[str, SubmissionRecord] = {}

    for line_no, obj in loaded:
        utt_id = _require_string(
            obj, id_key, "Submission", path, line_no, nonempty=True
        ).strip()
        text = _require_string(obj, text_key, "Submission", path, line_no)
        if utt_id in result:
            raise ValidationError(
                f"Submission file {path!r}, line {line_no}: duplicate utt_id detected.",
                code="DUPLICATE_UTT_ID",
            )
        result[utt_id] = SubmissionRecord(utt_id=utt_id, text=text)
    return result


def validate_id_sets(
    reference: Mapping[str, ReferenceRecord],
    submission: Mapping[str, SubmissionRecord],
) -> None:
    """Require exactly one prediction for every reference utterance."""
    ref_ids = set(reference)
    sub_ids = set(submission)
    missing = sorted(ref_ids - sub_ids)
    extra = sorted(sub_ids - ref_ids)
    if missing or extra:
        details: List[str] = []
        if missing:
            details.append(f"missing {len(missing)} required utt_id(s)")
        if extra:
            details.append(f"unknown {len(extra)} utt_id(s)")

        if missing and extra:
            code = "ID_SET_MISMATCH"
        elif missing:
            code = "MISSING_SAMPLES"
        else:
            code = "UNKNOWN_SAMPLES"

        raise ValidationError(
            "Submission utt_id set does not exactly match reference: "
            + "; ".join(details)
            + ".",
            code=code,
        )


# ---------------------------------------------------------------------------
# Normalization and tag parsing
# ---------------------------------------------------------------------------
def _canonical_tag(raw_tag: str, context: str) -> str:
    """Normalize tag spelling while rejecting non-canonical labels."""
    tag = unicodedata.normalize("NFKC", raw_tag).lower()
    tag = re.sub(r"\s+", " ", tag).strip()
    if not tag:
        raise ValidationError(
            f"{context}: empty NVV tag '[]' is not allowed.",
            code="EMPTY_NVV_TAG",
        )
    if tag not in VALID_TAG_SET:
        valid = ", ".join(VALID_TAGS)
        raise ValidationError(
            f"{context}: unsupported NVV tag [{tag}]. Valid tags are: {valid}.",
            code="UNSUPPORTED_NVV_TAG",
        )
    return tag


def _is_ignored_char(ch: str, *, preserve_apostrophe: bool) -> bool:
    """Return whether a non-tag character should be discarded in normalization."""
    if preserve_apostrophe and ch == "'":
        return False
    category = unicodedata.category(ch)
    # P*: punctuation, S*: symbols, Z*: separators, C*: controls/formats.
    return ch.isspace() or category[0] in {"P", "S", "Z", "C"}


def _tokenize_plain_segment(segment: str, language: str) -> List[str]:
    """Tokenize one non-tag text segment using the frozen language protocol."""
    segment = unicodedata.normalize("NFKC", segment).translate(APOSTROPHE_TRANSLATION).lower()

    if language == "en":
        normalized_chars = [
            ch if not _is_ignored_char(ch, preserve_apostrophe=True) else " "
            for ch in segment
        ]
        normalized = "".join(normalized_chars)
        return EN_WORD_RE.findall(normalized)

    if language == "zh":
        # Mandarin CER uses one normalized non-punctuation character per lexical unit.
        return [
            ch
            for ch in segment
            if not _is_ignored_char(ch, preserve_apostrophe=False)
        ]

    raise ValueError(f"Unsupported language: {language!r}")


def parse_tagged_transcript(text: str, language: str, *, context: str) -> ParsedTranscript:
    """Parse, validate, normalize, and tokenize a tagged transcript.

    Tags must be explicit bracketed expressions such as ``[laugh]``. Stray or
    unmatched square brackets are rejected rather than silently treated as text.
    """
    if not isinstance(text, str):
        raise ValidationError(
            f"{context}: transcript must be a string.",
            code="INVALID_TRANSCRIPT_TYPE",
        )
    if language not in {"en", "zh"}:
        raise ValueError(f"Unsupported language: {language!r}")

    lexical_tokens: List[str] = []
    tagged_tokens: List[str] = []
    events: List[Event] = []

    cursor = 0
    event_order = 0
    for match in TAG_RE.finditer(text):
        plain_segment = text[cursor:match.start()]
        if "[" in plain_segment or "]" in plain_segment:
            raise ValidationError(
                f"{context}: malformed or nested square-bracket tag.",
                code="MALFORMED_NVV_TAG",
            )

        segment_tokens = _tokenize_plain_segment(plain_segment, language)
        lexical_tokens.extend(segment_tokens)
        tagged_tokens.extend(segment_tokens)

        tag = _canonical_tag(match.group(1), context)
        events.append(Event(tag=tag, position=len(lexical_tokens), order=event_order))
        tagged_tokens.append(f"[{tag}]")
        event_order += 1
        cursor = match.end()

    trailing_segment = text[cursor:]
    if "[" in trailing_segment or "]" in trailing_segment:
        raise ValidationError(
            f"{context}: malformed or unmatched square bracket.",
            code="MALFORMED_NVV_TAG",
        )

    segment_tokens = _tokenize_plain_segment(trailing_segment, language)
    lexical_tokens.extend(segment_tokens)
    tagged_tokens.extend(segment_tokens)

    return ParsedTranscript(
        lexical_tokens=tuple(lexical_tokens),
        tagged_tokens=tuple(tagged_tokens),
        events=tuple(events),
    )


# ---------------------------------------------------------------------------
# Edit distance and lexical alignment
# ---------------------------------------------------------------------------
def levenshtein_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Memory-efficient Levenshtein edit distance for tagged WER/CER."""
    if len(reference) < len(hypothesis):
        # Keep the working row as short as possible.
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[j] + 1,          # deletion
                    current[j - 1] + 1,       # insertion
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def lexical_alignment_boundary_map(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> List[int]:
    """Map each hypothesis lexical boundary into reference coordinates.

    The map is derived from a deterministic minimum-edit alignment. For a
    hypothesis boundary ``j`` (after ``j`` lexical tokens), the returned
    reference boundary is the first reference coordinate visited at that
    boundary. This anchors tags before any following reference-only deletions,
    so a missing earlier tag never itself shifts later tag positions.
    """
    n, m = len(reference), len(hypothesis)
    dp: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        ref_token = reference[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ref_token == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,          # delete reference token
                dp[i][j - 1] + 1,          # insert hypothesis token
                dp[i - 1][j - 1] + cost,   # match/substitute
            )

    # Backtrace with deterministic tie breaking: diagonal, delete, insert.
    reverse_ops: List[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                reverse_ops.append("M" if cost == 0 else "S")
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            reverse_ops.append("D")
            i -= 1
            continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            reverse_ops.append("I")
            j -= 1
            continue
        raise RuntimeError("Internal alignment backtrace failure.")

    operations = list(reversed(reverse_ops))
    boundary_visits: List[List[int]] = [[] for _ in range(m + 1)]
    ref_index = 0
    hyp_index = 0
    boundary_visits[0].append(0)

    for op in operations:
        if op in {"M", "S"}:
            ref_index += 1
            hyp_index += 1
            boundary_visits[hyp_index].append(ref_index)
        elif op == "D":
            ref_index += 1
            boundary_visits[hyp_index].append(ref_index)
        elif op == "I":
            hyp_index += 1
            boundary_visits[hyp_index].append(ref_index)
        else:
            raise RuntimeError(f"Internal unknown alignment operation: {op!r}")

    if hyp_index != m or ref_index != n:
        raise RuntimeError("Internal alignment path does not terminate at the expected boundary.")

    # Every hypothesis boundary must have been visited. `min` implements the
    # documented first-coordinate anchoring convention.
    if any(not visits for visits in boundary_visits):
        raise RuntimeError("Internal alignment did not visit every hypothesis boundary.")
    return [min(visits) for visits in boundary_visits]


def align_predicted_events_to_reference(
    reference: ParsedTranscript, prediction: ParsedTranscript
) -> Tuple[Event, ...]:
    """Project predicted lexical tag positions into reference coordinates."""
    boundary_map = lexical_alignment_boundary_map(
        reference.lexical_tokens,
        prediction.lexical_tokens,
    )
    aligned: List[Event] = []
    for event in prediction.events:
        if not 0 <= event.position < len(boundary_map):
            raise RuntimeError("Internal predicted event position is outside the alignment map.")
        aligned.append(
            Event(tag=event.tag, position=boundary_map[event.position], order=event.order)
        )
    return tuple(aligned)


# ---------------------------------------------------------------------------
# Event matching and metric calculation
# ---------------------------------------------------------------------------
def _better_match_plan(
    candidate: Tuple[int, float, int, str],
    incumbent: Tuple[int, float, int, str],
) -> bool:
    """Compare plans: maximize matches, minimize cost, then deterministic priority."""
    cand_matches, cand_cost, cand_priority, _ = candidate
    best_matches, best_cost, best_priority, _ = incumbent
    if cand_matches != best_matches:
        return cand_matches > best_matches
    if not math.isclose(cand_cost, best_cost, rel_tol=0.0, abs_tol=1e-12):
        return cand_cost < best_cost
    return cand_priority > best_priority


def match_same_tag_events_for_f1(
    reference_events: Sequence[Event],
    predicted_events: Sequence[Event],
    collar: int,
) -> List[Tuple[Event, Event]]:
    """One-to-one F1 matching for one tag with max-cardinality/min-distance DP."""
    refs = sorted(reference_events, key=lambda event: (event.position, event.order))
    preds = sorted(predicted_events, key=lambda event: (event.position, event.order))
    n, m = len(refs), len(preds)

    # Cells: (number of matches, total absolute distance, action priority, action)
    # Action priorities resolve exact ties: match > skip reference > skip prediction.
    dp: List[List[Tuple[int, float, int, str]]] = [
        [(0, 0.0, 0, "") for _ in range(m + 1)] for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        dp[i][0] = (0, 0.0, 2, "R")
    for j in range(1, m + 1):
        dp[0][j] = (0, 0.0, 1, "P")

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates: List[Tuple[int, float, int, str]] = []
            # Skip the current reference event.
            a = dp[i - 1][j]
            candidates.append((a[0], a[1], 2, "R"))
            # Skip the current predicted event.
            b = dp[i][j - 1]
            candidates.append((b[0], b[1], 1, "P"))
            # Match if this pair is within the fixed collar.
            distance = abs(refs[i - 1].position - preds[j - 1].position)
            if distance <= collar:
                c = dp[i - 1][j - 1]
                candidates.append((c[0] + 1, c[1] + float(distance), 3, "M"))

            best = candidates[0]
            for candidate in candidates[1:]:
                if _better_match_plan(candidate, best):
                    best = candidate
            dp[i][j] = best

    pairs: List[Tuple[Event, Event]] = []
    i, j = n, m
    while i > 0 or j > 0:
        action = dp[i][j][3]
        if action == "M":
            pairs.append((refs[i - 1], preds[j - 1]))
            i -= 1
            j -= 1
        elif action == "R":
            i -= 1
        elif action == "P":
            j -= 1
        else:
            raise RuntimeError("Internal F1 matching backtrace failure.")
    pairs.reverse()
    return pairs


def count_f1_matches(
    reference_events: Sequence[Event],
    predicted_events: Sequence[Event],
    collar: int,
) -> int:
    """Count F1 true positives after category-specific optimal event matching."""
    by_tag_ref: Dict[str, List[Event]] = {tag: [] for tag in VALID_TAGS}
    by_tag_pred: Dict[str, List[Event]] = {tag: [] for tag in VALID_TAGS}
    for event in reference_events:
        by_tag_ref[event.tag].append(event)
    for event in predicted_events:
        by_tag_pred[event.tag].append(event)

    tp = 0
    for tag in VALID_TAGS:
        tp += len(match_same_tag_events_for_f1(by_tag_ref[tag], by_tag_pred[tag], collar))
    return tp


def utterance_mntd(
    reference_events: Sequence[Event],
    predicted_events: Sequence[Event],
    reference_length: int,
) -> float:
    """Compute bounded multi-event normalized tag distance for one utterance.

    Events are sorted by reference-coordinate position. A sequence-alignment DP
    preserves event order; this prevents crossed matches from hiding a swapped
    event order. The result lies in [0, 1].
    """
    refs = sorted(reference_events, key=lambda event: (event.position, event.order))
    preds = sorted(predicted_events, key=lambda event: (event.position, event.order))
    n, m = len(refs), len(preds)

    if n == 0:
        # mNTD is not defined for no-reference-NVV utterances and is excluded
        # from the language macro average. This branch is useful for self-tests.
        return 0.0 if m == 0 else 1.0

    denominator = max(reference_length, 1)
    # DP cells: (cost, action priority, action). Lower cost is better;
    # tie priority resolves exact ties: diagonal > delete-ref > insert-pred.
    dp: List[List[Tuple[float, int, str]]] = [
        [(0.0, 0, "") for _ in range(m + 1)] for _ in range(n + 1)
    ]
    for i in range(1, n + 1):
        dp[i][0] = (float(i), 2, "R")
    for j in range(1, m + 1):
        dp[0][j] = (float(j), 1, "P")

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ref_event = refs[i - 1]
            pred_event = preds[j - 1]
            if ref_event.tag == pred_event.tag:
                pair_cost = min(abs(ref_event.position - pred_event.position) / denominator, 1.0)
            else:
                # A wrong category is a maximum event-level error.
                pair_cost = 1.0

            candidates = [
                (dp[i - 1][j][0] + 1.0, 2, "R"),
                (dp[i][j - 1][0] + 1.0, 1, "P"),
                (dp[i - 1][j - 1][0] + pair_cost, 3, "M"),
            ]
            best = candidates[0]
            for candidate in candidates[1:]:
                if candidate[0] < best[0] - 1e-12 or (
                    math.isclose(candidate[0], best[0], rel_tol=0.0, abs_tol=1e-12)
                    and candidate[1] > best[1]
                ):
                    best = candidate
            dp[i][j] = best

    normalizer = max(n, m)
    value = dp[n][m][0] / normalizer
    # Floating-point safety guard; the construction mathematically guarantees [0, 1].
    return max(0.0, min(1.0, value))


def safe_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Return (precision, recall, f1) for pooled event counts."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0.0:
        return precision, recall, 0.0
    return precision, recall, 2.0 * precision * recall / (precision + recall)


def track1_score(f1_micro: float, mntd: float, tagged_error: float) -> float:
    """Compute the frozen Track 1 ranking score for one language."""
    value = 100.0 * (
        0.70 * f1_micro
        + 0.20 * (1.0 - mntd)
        + 0.10 * (1.0 - min(tagged_error, 1.0))
    )
    # Stabilize public JSON/leaderboard values against harmless binary-float noise
    # (for example, 99.99999999999999 instead of 100.0).
    return round(max(0.0, min(100.0, value)), 12)


# ---------------------------------------------------------------------------
# Language and full-file evaluation
# ---------------------------------------------------------------------------
def evaluate_language(
    records: Sequence[Tuple[ReferenceRecord, SubmissionRecord]],
    language: str,
) -> Dict[str, Any]:
    """Evaluate English or Mandarin records and return serializable metrics."""
    if language not in {"en", "zh"}:
        raise ValueError(f"Unsupported language: {language!r}")
    if not records:
        raise ValidationError(
            f"No {language!r} records are present in the reference file.",
            code="REFERENCE_LANGUAGE_COVERAGE_ERROR",
        )

    collar = F1_COLLAR[language]
    tp = fp = fn = 0
    total_edit_distance = 0
    total_reference_tagged_tokens = 0
    per_utterance_mntd: List[float] = []

    per_tag: Dict[str, Dict[str, int]] = {
        tag: {"tp": 0, "fp": 0, "fn": 0, "reference": 0, "prediction": 0}
        for tag in VALID_TAGS
    }

    for reference_record, submission_record in records:
        # Do not expose utterance IDs in participant-visible validation logs.
        context_ref = "Reference transcript"
        context_pred = "Submission transcript"
        reference = parse_tagged_transcript(reference_record.text, language, context=context_ref)
        prediction = parse_tagged_transcript(submission_record.text, language, context=context_pred)
        aligned_predictions = align_predicted_events_to_reference(reference, prediction)

        edit_distance = levenshtein_distance(reference.tagged_tokens, prediction.tagged_tokens)
        total_edit_distance += edit_distance
        total_reference_tagged_tokens += len(reference.tagged_tokens)

        # Global and per-label F1 counts.
        by_tag_ref: Dict[str, List[Event]] = {tag: [] for tag in VALID_TAGS}
        by_tag_pred: Dict[str, List[Event]] = {tag: [] for tag in VALID_TAGS}
        for event in reference.events:
            by_tag_ref[event.tag].append(event)
        for event in aligned_predictions:
            by_tag_pred[event.tag].append(event)

        utterance_tp = 0
        for tag in VALID_TAGS:
            matched = len(match_same_tag_events_for_f1(by_tag_ref[tag], by_tag_pred[tag], collar))
            ref_count = len(by_tag_ref[tag])
            pred_count = len(by_tag_pred[tag])
            per_tag[tag]["tp"] += matched
            per_tag[tag]["fp"] += pred_count - matched
            per_tag[tag]["fn"] += ref_count - matched
            per_tag[tag]["reference"] += ref_count
            per_tag[tag]["prediction"] += pred_count
            utterance_tp += matched

        utterance_fp = len(aligned_predictions) - utterance_tp
        utterance_fn = len(reference.events) - utterance_tp
        tp += utterance_tp
        fp += utterance_fp
        fn += utterance_fn

        # mNTD is measured only when there is an expected reference NVV.
        if reference.events:
            per_utterance_mntd.append(
                utterance_mntd(
                    reference.events,
                    aligned_predictions,
                    reference_length=len(reference.lexical_tokens),
                )
            )

    if total_reference_tagged_tokens == 0:
        raise ValidationError(
            f"{language} reference records contain no normalized tagged tokens; "
            "cannot compute tagged-transcript error.",
            code="EMPTY_REFERENCE_TOKENS",
        )

    precision, recall, f1_micro = safe_f1(tp, fp, fn)
    mntd = sum(per_utterance_mntd) / len(per_utterance_mntd) if per_utterance_mntd else 0.0
    tagged_error = total_edit_distance / total_reference_tagged_tokens
    score = track1_score(f1_micro, mntd, tagged_error)

    per_tag_metrics: Dict[str, Dict[str, Any]] = {}
    for tag in VALID_TAGS:
        tag_counts = per_tag[tag]
        tag_precision, tag_recall, tag_f1 = safe_f1(
            tag_counts["tp"], tag_counts["fp"], tag_counts["fn"]
        )
        per_tag_metrics[tag] = {
            **tag_counts,
            "precision": tag_precision,
            "recall": tag_recall,
            "f1": tag_f1,
        }

    return {
        "language": language,
        "num_utterances": len(records),
        "f1_collar": collar,
        "event_counts": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "reference": tp + fn,
            "prediction": tp + fp,
        },
        "precision": precision,
        "recall": recall,
        "F1_micro": f1_micro,
        "mNTD": mntd,
        "mNTD_num_utterances": len(per_utterance_mntd),
        "tagged_transcript_error": tagged_error,
        "tagged_transcript_error_name": "WER" if language == "en" else "CER",
        "tagged_edit_distance": total_edit_distance,
        "tagged_reference_tokens": total_reference_tagged_tokens,
        "Track1Score": score,
        "per_tag": per_tag_metrics,
    }


def evaluate(
    reference: Mapping[str, ReferenceRecord],
    submission: Mapping[str, SubmissionRecord],
) -> Dict[str, Any]:
    """Evaluate a validated bilingual submission and return full results."""
    records_by_language: Dict[str, List[Tuple[ReferenceRecord, SubmissionRecord]]] = {
        "en": [],
        "zh": [],
    }
    for utt_id, ref_record in reference.items():
        records_by_language[ref_record.language].append((ref_record, submission[utt_id]))

    # Sorting makes all diagnostics deterministic even if JSONL order differs.
    for language in records_by_language:
        records_by_language[language].sort(key=lambda pair: pair[0].utt_id)

    en_result = evaluate_language(records_by_language["en"], "en")
    zh_result = evaluate_language(records_by_language["zh"], "zh")
    final_score = round((en_result["Track1Score"] + zh_result["Track1Score"]) / 2.0, 12)

    return {
        "protocol": {
            "name": PROTOCOL_NAME,
            "version": PROTOCOL_VERSION,
            "valid_tags": list(VALID_TAGS),
            "f1_collar": dict(F1_COLLAR),
            "final_score_definition": "(Track1Score_ZH + Track1Score_EN) / 2",
        },
        "scores": {
            "Track1Score_EN": en_result["Track1Score"],
            "Track1Score_ZH": zh_result["Track1Score"],
            "FinalTrack1Score": final_score,
        },
        "languages": {
            "en": en_result,
            "zh": zh_result,
        },
    }


# ---------------------------------------------------------------------------
# Reporting and self tests
# ---------------------------------------------------------------------------
def _print_language_summary(result: Mapping[str, Any]) -> None:
    name = "EN" if result["language"] == "en" else "ZH"
    error_name = result["tagged_transcript_error_name"]
    events = result["event_counts"]
    print(
        f"[{name}] utterances={result['num_utterances']} | "
        f"TP/FP/FN={events['tp']}/{events['fp']}/{events['fn']} | "
        f"F1_micro={result['F1_micro']:.6f} | "
        f"mNTD={result['mNTD']:.6f} | "
        f"{error_name}={result['tagged_transcript_error']:.6f} | "
        f"Track1Score={result['Track1Score']:.6f}"
    )


def print_summary(results: Mapping[str, Any]) -> None:
    """Print only leaderboard-relevant totals; full detail goes to JSON."""
    print("=" * 88)
    print(f"{results['protocol']['name']} | protocol v{results['protocol']['version']}")
    _print_language_summary(results["languages"]["en"])
    _print_language_summary(results["languages"]["zh"])
    print("-" * 88)
    print(f"Track1Score_EN       : {results['scores']['Track1Score_EN']:.6f}")
    print(f"Track1Score_ZH       : {results['scores']['Track1Score_ZH']:.6f}")
    print(f"FinalTrack1Score     : {results['scores']['FinalTrack1Score']:.6f}")
    print("=" * 88)


def write_json(path: str, payload: Mapping[str, Any]) -> None:
    """Write JSON results atomically enough for evaluator usage."""
    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as exc:
        raise EvaluationError(
            f"Unable to write output JSON {path!r}: {type(exc).__name__}",
            code="OUTPUT_WRITE_ERROR",
        ) from exc


def run_self_tests() -> None:
    """Small dependency-free regression suite for release validation."""
    # Normalization and canonical tag processing.
    parsed = parse_tagged_transcript(
        "I’m [Laugh], really!", "en", context="self-test"
    )
    assert parsed.lexical_tokens == ("i'm", "really")
    assert parsed.tagged_tokens == ("i'm", "[laugh]", "really")
    assert parsed.events[0].position == 1

    # Invalid website/sample typo must not silently enter official scoring.
    try:
        parse_tagged_transcript("x [sniffl] y", "en", context="self-test")
    except ValidationError:
        pass
    else:
        raise AssertionError("Invalid tag was not rejected.")

    # Alignment must not let a missing earlier NVV tag shift a later lexical position.
    reference = parse_tagged_transcript("a b c", "en", context="self-test")
    prediction = parse_tagged_transcript("a [cough] c", "en", context="self-test")
    aligned = align_predicted_events_to_reference(reference, prediction)
    assert aligned[0].position == 1, aligned

    # Perfect prediction: all components exactly optimal.
    ref = parse_tagged_transcript("a [laugh] b [cough]", "en", context="self-test")
    hyp = parse_tagged_transcript("a [laugh] b [cough]", "en", context="self-test")
    aligned_hyp = align_predicted_events_to_reference(ref, hyp)
    assert count_f1_matches(ref.events, aligned_hyp, collar=2) == 2
    assert utterance_mntd(ref.events, aligned_hyp, len(ref.lexical_tokens)) == 0.0
    assert levenshtein_distance(ref.tagged_tokens, hyp.tagged_tokens) == 0

    # Missing event must receive maximum mNTD penalty, not a free zero distance.
    missing = parse_tagged_transcript("a b", "en", context="self-test")
    aligned_missing = align_predicted_events_to_reference(ref, missing)
    assert math.isclose(
        utterance_mntd(ref.events, aligned_missing, len(ref.lexical_tokens)), 1.0
    )

    # A wrong tag must be maximum mNTD error and no F1 true positive.
    wrong_tag = parse_tagged_transcript("a [cough] b", "en", context="self-test")
    ref_one = parse_tagged_transcript("a [laugh] b", "en", context="self-test")
    aligned_wrong = align_predicted_events_to_reference(ref_one, wrong_tag)
    assert count_f1_matches(ref_one.events, aligned_wrong, collar=2) == 0
    assert math.isclose(
        utterance_mntd(ref_one.events, aligned_wrong, len(ref_one.lexical_tokens)), 1.0
    )

    # Corpus-level tagged error must be length-weighted, not an average of utterance rates.
    assert levenshtein_distance(["x"], ["y"]) == 1
    assert levenshtein_distance(["a"] * 100, ["a"] * 100) == 0

    # Score boundaries.
    assert math.isclose(track1_score(1.0, 0.0, 0.0), 100.0)
    assert 0.0 <= track1_score(0.0, 1.0, 1.0) <= 100.0

    print("Self-tests passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Official scorer for NVVSpeech Challenge @ ISCSLP 2026 Track 1."
    )
    parser.add_argument(
        "--reference",
        help="Hidden reference JSONL. Required unless --self-test is used.",
    )
    parser.add_argument(
        "--submission",
        help="Participant submission JSONL. Required unless --self-test is used.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path for full machine-readable metrics JSON.",
    )
    parser.add_argument(
        "--id-key",
        default=ID_KEY_DEFAULT,
        help=f"JSON key for utterance IDs (default: {ID_KEY_DEFAULT!r}).",
    )
    parser.add_argument(
        "--language-key",
        default=LANGUAGE_KEY_DEFAULT,
        help=f"Reference JSON key for language (default: {LANGUAGE_KEY_DEFAULT!r}).",
    )
    parser.add_argument(
        "--reference-text-key",
        default=TEXT_KEY_DEFAULT,
        help=f"Reference JSON key for tagged transcript (default: {TEXT_KEY_DEFAULT!r}).",
    )
    parser.add_argument(
        "--submission-text-key",
        default=TEXT_KEY_DEFAULT,
        help=f"Submission JSON key for tagged transcript (default: {TEXT_KEY_DEFAULT!r}).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in regression tests and exit.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        if args.reference or args.submission:
            parser.error("--self-test cannot be combined with --reference or --submission.")
        run_self_tests()
        return 0

    if not args.reference or not args.submission:
        parser.error("--reference and --submission are required unless --self-test is used.")

    reference = load_reference(
        args.reference,
        id_key=args.id_key,
        language_key=args.language_key,
        text_key=args.reference_text_key,
    )
    submission = load_submission(
        args.submission,
        id_key=args.id_key,
        text_key=args.submission_text_key,
    )
    validate_id_sets(reference, submission)

    results = evaluate(reference, submission)
    print_summary(results)
    if args.output_json:
        write_json(args.output_json, results)
        print(f"Detailed metrics JSON written to: {args.output_json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as exc:
        print(f"ERROR TYPE: {exc.code}", file=sys.stderr)
        print(f"DETAILS   : {exc}", file=sys.stderr)
        raise SystemExit(2)
