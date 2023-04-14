from argparse import ArgumentParser, Namespace
from pathlib import Path

from predict import predict


def main() -> None:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--test_directory_path", type=str, default="./test/images")
    parser.add_argument("--train_directory_path", type=str, default="./train/png")
    parser.add_argument("--sampling_levels", nargs="+", type=int, default=[3])
    args: Namespace = parser.parse_args()
    test_directory_path: Path = Path(args.test_directory_path).resolve(strict=True)
    train_directory_path: Path = Path(args.train_directory_path).resolve(strict=True)
    predict(test_directory_path, train_directory_path, args.sampling_levels)


if __name__ == "__main__":
    main()
