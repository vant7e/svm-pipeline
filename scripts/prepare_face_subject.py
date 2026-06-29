import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from svm.core.loader import (
    save_face_subject,
    save_face_subject_epochs,
    find_face_subjects,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--average_repetitions", action="store_true")

    parser.add_argument(
        "--save_format",
        choices=["npy", "epochs"],
        default="epochs",
    )

    parser.add_argument("--sfreq", type=float, default=512.0)
    parser.add_argument("--tmin", type=float, default=0.0)

    args = parser.parse_args()

    root = Path(args.root)

    if args.scan:
        subjects = find_face_subjects(root)
        print(f"Found {len(subjects)} participant folders:")
        for s in subjects:
            print(s)
        return

    if args.save_format == "epochs":
        save_face_subject_epochs(
            subject_root=root,
            out_dir=args.out,
            subject_id=args.subject,
            average_repetitions=args.average_repetitions,
            sfreq=args.sfreq,
            tmin=args.tmin,
        )
    else:
        save_face_subject(
            subject_root=root,
            out_dir=args.out,
            average_repetitions=args.average_repetitions,
            subject_id=args.subject,
        )


if __name__ == "__main__":
    main()
