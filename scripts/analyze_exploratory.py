"""CPU-only analysis of one shared initial baseline and two fixed endpoints."""
import argparse
import json

from gasp.evaluation import analyze_paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Directory containing baseline metrics.json and predictions.jsonl")
    parser.add_argument("unified", help="Unified endpoint evaluation directory")
    parser.add_argument("split", help="Split endpoint evaluation directory")
    parser.add_argument("output")
    args = parser.parse_args()
    result = analyze_paired(args.baseline, args.unified, args.split, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
