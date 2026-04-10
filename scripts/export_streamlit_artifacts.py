"""Export deployable artifacts for the current Streamlit app."""

from __future__ import annotations

import argparse
import json

from thyroid_app.export_current_artifacts import export_all_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts", help="Artifact root directory")
    args = parser.parse_args()

    manifest = export_all_artifacts(args.output_dir)
    print(json.dumps(manifest["tasks"], indent=2))


if __name__ == "__main__":
    main()
