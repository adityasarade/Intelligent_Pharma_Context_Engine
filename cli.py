"""
CLI for the Intelligent Pharma-Context Engine.

Usage:
    python cli.py process <image_path> [-o output.json]
"""

import argparse
import sys

from dotenv import load_dotenv

from src import PharmaContextPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Pharma-Context Engine CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: process
    process_parser = subparsers.add_parser(
        "process", help="Process a single image"
    )
    process_parser.add_argument("image_path", help="Path to the image file")
    process_parser.add_argument(
        "--output", "-o", help="Path to save output JSON", default=None
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    if args.command == "process":
        print(f"Initializing pipeline for {args.image_path}...")
        try:
            pipeline = PharmaContextPipeline()
            result = pipeline.process_image(args.image_path)

            if result:
                json_output = result.model_dump_json(indent=2)
                print(json_output)

                if args.output:
                    with open(args.output, "w") as f:
                        f.write(json_output)
                    print(f"\n[INFO] Output saved to {args.output}")
            else:
                print("\n[ERROR] Pipeline returned no result.")
                sys.exit(1)

        except Exception as e:
            print(f"\n[CRITICAL] Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
