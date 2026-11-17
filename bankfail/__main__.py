import argparse
import sys
import time

from bankfail.config import load_config

COMMANDS = ("download", "panel", "evaluate", "rolling", "case", "current", "stress", "export")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bankfail")
    parser.add_argument("command", choices=COMMANDS + ("all", "synth"))
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--banks", type=int, default=1500)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    started = time.perf_counter()
    if args.command == "download":
        from bankfail.download import download_all

        manifest = download_all(config)
        print(
            f"{manifest['financial_rows']:,} bank quarters, {manifest['failure_rows']:,} failures"
        )
    elif args.command == "synth":
        from bankfail.synth import write_synthetic

        write_synthetic(config, args.banks)
    elif args.command == "all":
        from bankfail.pipeline import run_all

        run_all(config)
    else:
        from bankfail.pipeline import run_stage

        run_stage(args.command, config)
    print(f"{args.command} finished in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
