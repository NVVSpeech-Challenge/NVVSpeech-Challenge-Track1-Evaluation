#!/usr/bin/env python3
"""NVVSpeech Challenge Track 1 — CodaBench Scoring Program.

Reads the hidden reference JSONL and the participant submission JSONL,
computes the official Track 1 bilingual scores, and writes scores.json
for the CodaBench leaderboard.

Usage (from metadata.yaml):
  python3 $program/score.py $input/ref/ref.jsonl $input/res $output

The submission file can have any name — the first .jsonl file found in the
res/ directory is used.
"""

import glob
import json
import os
import sys
import zipfile
from collections import Counter

# Ensure the program directory is on the import path so the official
# evaluator can be imported from the same bundle without modification.
_program_dir = os.path.dirname(os.path.abspath(__file__))
if _program_dir not in sys.path:
    sys.path.insert(0, _program_dir)

from eval_track1_official import (
    EvaluationError,
    load_reference,
    load_submission,
    validate_id_sets,
    evaluate,
    write_json,
)


def _find_submission_file(res_dir: str) -> str:
    """Find the first .jsonl file in the submission directory."""
    candidates = sorted(glob.glob(os.path.join(res_dir, "*.jsonl")))
    if not candidates:
        raise FileNotFoundError(
            f"No .jsonl submission file found in {res_dir}"
        )
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Multiple .jsonl files found in submission: {len(candidates)}. "
            "Please submit exactly ONE .jsonl file."
        )


def _format_extension_counts(counts: Counter) -> str:
    """Format unsupported file types without exposing submitted filenames/IDs."""
    return ", ".join(
        f"{extension}: {count}"
        for extension, count in sorted(counts.items())
    )


def _write_validation_error(output_dir: str, code: str, message: str):
    """Write validation failure — no scores.json so CodaBench marks as failed."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "validation_error.json"), "w") as fh:
        json.dump({"code": code, "message": message}, fh, ensure_ascii=False, indent=2)
    print(f"\n  VALIDATION FAILED — {code}")
    print(f"  {message}")
    print("=" * 60)


def main() -> int:
    if len(sys.argv) < 4:
        print(
            f"Usage: {sys.argv[0]} <ref.jsonl> <res_dir> <output_dir>",
            file=sys.stderr,
        )
        return 1

    ref_path = sys.argv[1]
    res_dir = sys.argv[2]
    output_dir = sys.argv[3]

    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. Find submission file ----
    # CodaBench extracts uploaded zips, so the .jsonl usually lands
    # directly in res/.  We also handle the case where it stays zipped.
    candidates = sorted(glob.glob(os.path.join(res_dir, "*.jsonl")))
    bad_zip_count = 0
    valid_zip_count = 0
    archive_regular_file_count = 0
    nested_jsonl_count = 0
    unsupported_extension_counts = Counter()

    # Diagnostic-only scan. This does not change the original acceptance rule:
    # the submission JSONL must still be in res/ or at the root of a ZIP file.
    for entry in glob.glob(os.path.join(res_dir, "*")):
        if not os.path.isfile(entry):
            continue
        basename = os.path.basename(entry)
        extension = os.path.splitext(basename)[1].lower() or "<no extension>"
        if basename == ".DS_Store" or extension in {".jsonl", ".zip"}:
            continue
        unsupported_extension_counts[extension] += 1

    for root, _dirs, files in os.walk(res_dir):
        if os.path.abspath(root) == os.path.abspath(res_dir):
            continue
        for filename in files:
            if filename.lower().endswith(".jsonl"):
                nested_jsonl_count += 1

    if not candidates:
        # Fallback: check for zip files (some CodaBench versions keep them)
        for zpath in sorted(glob.glob(os.path.join(res_dir, "*.zip"))):
            try:
                with zipfile.ZipFile(zpath) as zf:
                    valid_zip_count += 1
                    for name in zf.namelist():
                        normalized_name = name.replace("\\", "/").strip("/")
                        if not normalized_name or name.endswith("/"):
                            continue
                        if normalized_name.startswith("__MACOSX/"):
                            continue

                        archive_regular_file_count += 1
                        if normalized_name.lower().endswith(".jsonl"):
                            if "/" in normalized_name:
                                nested_jsonl_count += 1
                                continue
                            import tempfile
                            tmpdir = tempfile.mkdtemp(prefix="sub_")
                            candidates.append(zf.extract(name, tmpdir))
                        else:
                            basename = os.path.basename(normalized_name)
                            if basename == ".DS_Store":
                                continue
                            extension = os.path.splitext(basename)[1].lower() or "<no extension>"
                            unsupported_extension_counts[extension] += 1
            except (zipfile.BadZipFile, OSError):
                bad_zip_count += 1
                continue

    if not candidates:
        if bad_zip_count:
            code = "INVALID_ZIP_ARCHIVE"
            message = (
                f"{bad_zip_count} ZIP archive(s) cannot be opened. "
                "Please upload a valid ZIP containing exactly one .jsonl file."
            )
        elif nested_jsonl_count:
            code = "INVALID_SUBMISSION_LAYOUT"
            message = (
                f"Found {nested_jsonl_count} .jsonl file(s) inside a subdirectory. "
                "The submission .jsonl file must be placed at the root of the upload."
            )
        elif unsupported_extension_counts:
            code = "UNSUPPORTED_SUBMISSION_FORMAT"
            message = (
                "No .jsonl submission file was found. Unsupported file type(s): "
                f"{_format_extension_counts(unsupported_extension_counts)}. "
                "Submit exactly one .jsonl file."
            )
        elif valid_zip_count and archive_regular_file_count == 0:
            code = "EMPTY_SUBMISSION_ARCHIVE"
            message = (
                f"Found {valid_zip_count} ZIP archive(s), but no submission file is present. "
                "The ZIP must contain exactly one .jsonl file at its root."
            )
        else:
            code = "SUBMISSION_NOT_FOUND"
            message = (
                "No .jsonl submission file was found. "
                "Submit a ZIP containing exactly one .jsonl file at its root."
            )

        _write_validation_error(output_dir, code, message)
        return 0
    if len(candidates) > 1:
        _write_validation_error(output_dir, "MULTIPLE_JSONL",
                                f"Found {len(candidates)} .jsonl files. "
                                "Please submit exactly ONE .jsonl file.")
        return 0
    sub_path = candidates[0]

    # ---- 2. Pre-validation: load, check format, check ID coverage ----
    print("=" * 60)
    print("  PRE-VALIDATION")
    print("=" * 60)

    try:
        reference = load_reference(
            ref_path,
            id_key="utt_id",
            language_key="language",
            text_key="text_with_nvvs",
        )
        submission_all = load_submission(
            sub_path,
            id_key="utt_id",
            text_key="text_with_nvvs",
        )
    except EvaluationError as exc:
        _write_validation_error(
            output_dir,
            getattr(exc, "code", "FORMAT_ERROR"),
            str(exc),
        )
        return 0

    # ---- Check coverage: missing = fail, extra = warn ----
    ref_ids = set(reference.keys())
    sub_ids = set(submission_all.keys())
    missing = sorted(ref_ids - sub_ids)
    extra = sorted(sub_ids - ref_ids)

    if missing:
        code = "ID_SET_MISMATCH" if extra else "MISSING_SAMPLES"
        if extra:
            message = (
                f"Submission ID coverage is invalid: {len(missing)} required utt_id(s) "
                f"are missing and {len(extra)} unknown utt_id(s) were submitted."
            )
        else:
            message = (
                f"Submission is missing {len(missing)} required utt_id(s)."
            )
        _write_validation_error(
            output_dir,
            code,
            message,
        )
        return 0

    if extra:
        print(f"  Warning: {len(extra)} extra utt_id(s) not in this phase "
              "(ignored)")

    # Keep only the required subset for scoring
    submission = {uid: submission_all[uid] for uid in ref_ids}

    # Check for empty predictions
    empty_ids = [uid for uid, rec in submission.items() if not rec.text.strip()]
    if empty_ids:
        _write_validation_error(
            output_dir, "EMPTY_PREDICTIONS",
            f"{len(empty_ids)} utterance(s) have empty text_with_nvvs."
        )
        return 0

    ref_zh = sum(1 for r in reference.values() if r.language == "zh")
    ref_en = sum(1 for r in reference.values() if r.language == "en")
    print(f"  Reference : {len(reference)} samples (ZH: {ref_zh}, EN: {ref_en})")
    print(f"  Submission: {len(submission_all)} samples → {len(submission)} used")
    print(f"  PRE-VALIDATION PASSED")
    print("=" * 60)

    # ---- 3. Evaluate ----
    try:
        results = evaluate(reference, submission)
    except EvaluationError as exc:
        _write_validation_error(
            output_dir,
            getattr(exc, "code", "EVALUATION_ERROR"),
            str(exc),
        )
        return 0

    # ---- 4. Write scores.json ----
    scores = {
        "Track1Score_ZH": results["scores"]["Track1Score_ZH"],
        "Track1Score_EN": results["scores"]["Track1Score_EN"],
        "FinalTrack1Score": results["scores"]["FinalTrack1Score"],
    }
    with open(os.path.join(output_dir, "scores.json"), "w") as fh:
        json.dump(scores, fh)

    # ---- 5. Write full detailed results ----
    write_json(os.path.join(output_dir, "detailed_results.json"), results)

    print(f"Track1Score_ZH      : {scores['Track1Score_ZH']:.6f}")
    print(f"Track1Score_EN      : {scores['Track1Score_EN']:.6f}")
    print(f"FinalTrack1Score    : {scores['FinalTrack1Score']:.6f}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("FATAL TYPE: INTERNAL_SCORING_ERROR", file=sys.stderr)
        print(f"DETAILS   : {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(2)
