"""Verify exported artifacts against direct bundle inference."""

from __future__ import annotations

import argparse
import math

from thyroid_app.inference import load_manifest, predict_fixed_landmark, predict_relapse


def _assert_close(name: str, actual: float, expected: float, tol: float = 1e-8) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{name} mismatch: expected {expected}, got {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()
    manifest = load_manifest(args.artifacts_dir)

    relapse_case = manifest["sample_cases"]["relapse"]
    relapse_result = predict_relapse(relapse_case, args.artifacts_dir)
    _assert_close("relapse", relapse_result["predicted_probability"], relapse_case["expected_probability"])

    for landmark, case_payload in manifest["sample_cases"]["fixed"].items():
        result = predict_fixed_landmark(case_payload, landmark, args.artifacts_dir)
        if result["predicted_probability"] is None:
            raise AssertionError(f"Fixed landmark validation failed for {landmark}: {result['validation_messages']}")

    print("Artifact verification completed successfully.")


if __name__ == "__main__":
    main()

