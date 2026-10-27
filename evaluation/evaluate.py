
import argparse
import re
from pathlib import Path

import stable_whisper


def normalize(text):
    """Lowercase text and remove punctuation for WER comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()


def word_error_rate(reference, hypothesis):
    """Calculate WER using word-level edit distance."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)

    if not ref:
        raise ValueError("The reference transcript contains no words.")

    # Dynamic programming edit-distance calculation.
    previous = list(range(len(hyp) + 1))

    for i, ref_word in enumerate(ref, start=1):
        current = [i]

        for j, hyp_word in enumerate(hyp, start=1):
            substitution_cost = 0 if ref_word == hyp_word else 1

            current.append(
                min(
                    previous[j] + 1,                  # deletion
                    current[j - 1] + 1,              # insertion
                    previous[j - 1] + substitution_cost,
                )
            )

        previous = current

    return previous[-1] / len(ref)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Stable-TS transcription against a reference."
    )
    parser.add_argument("audio", help="Path to the test audio file")
    parser.add_argument("reference", help="Path to the reference text file")
    parser.add_argument(
        "--model",
        default="turbo",
        help="Whisper model name (default: turbo)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    reference_path = Path(args.reference)

    if not audio_path.is_file():
        parser.error(f"Audio file not found: {audio_path}")

    if not reference_path.is_file():
        parser.error(f"Reference file not found: {reference_path}")

    reference = reference_path.read_text(encoding="utf-8").strip()

    print(f"Loading model: {args.model}")
    model = stable_whisper.load_model(args.model)

    print(f"Transcribing: {audio_path}")
    result = model.transcribe(
        str(audio_path),
        regroup=False,
        beam_size=5,
        best_of=5,
        patience=1.0,
        no_speech_threshold=0.1,
        compression_ratio_threshold=2.0,
        condition_on_previous_text=True,
        task="transcribe",
    )

    hypothesis = result.text.strip()
    wer = word_error_rate(reference, hypothesis)

    print("\n--- Reference transcript ---")
    print(reference)

    print("\n--- System transcription ---")
    print(hypothesis)

    print(f"\nWord Error Rate (WER): {wer:.4f} ({wer * 100:.2f}%)")


if __name__ == "__main__":
    main()
