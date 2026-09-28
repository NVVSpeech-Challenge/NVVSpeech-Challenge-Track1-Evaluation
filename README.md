# NVVSpeech Challenge Track 1 Evaluation

Track 1 maps an input speech recording to a transcript containing the spoken
content and canonical NVV tags.

Each submission row has this form:

```json
{"utt_id": "zh_0001", "text_with_nvvs": "[breath] example"}
```

Run the scorer with an authorized reference file and a directory containing
exactly one submission JSONL:

```bash
python3 program/score.py /path/to/ref.jsonl /path/to/res /path/to/output
```

The scorer writes `scores.json` on success. The hidden reference JSONL contains
the ground-truth tagged transcripts and is intentionally excluded from this
repository. The supplied test package contains 1,946 audio files (985 Chinese
and 961 English) and should be distributed through the challenge's approved
dataset channel rather than committed to Git.

## Test set
Huggingface: [Track 1 test set](https://huggingface.co/datasets/NVVSpeech-Challenge/NVVSpeech-Challenge-Track1-Test-Set)
