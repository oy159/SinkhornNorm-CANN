#!/usr/bin/env python3
"""Tests for the independent offline utility, without any NPU execution."""

import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/reference/sinkhorn_reference.py"
SPEC = importlib.util.spec_from_file_location("sinkhorn_reference", SCRIPT)
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


class ReferenceSemanticsTest(unittest.TestCase):
    def assert_vector_close(self, actual, expected, tolerance=2e-7):
        self.assertEqual(len(actual), len(expected))
        for got, wanted in zip(actual, expected):
            self.assertLess(abs(got - wanted), tolerance)

    def test_direction_and_fixed_iteration_count(self):
        # Hand-derived rational examples. Softmax rows are [1/2,1/2] and
        # [1/4,3/4]; after columns only, rows are deliberately not normalized.
        token = [0.0] * 4 + [0.0, 0.0, 0.0, math.log(3.0)]
        pre, post, first = reference.reference_token(token, 2, 1, 0.0)
        self.assert_vector_close(pre, [0.5, 0.5])
        self.assert_vector_close(post, [1.0, 1.0])
        self.assert_vector_close(first, [2 / 3, 2 / 5, 1 / 3, 3 / 5])
        self.assertGreater(abs(sum(first[:2]) - 1.0), 0.05)
        _, _, second = reference.reference_token(token, 2, 2, 0.0)
        self.assert_vector_close(second, [7 / 11, 7 / 19, 4 / 11, 12 / 19])
        _, _, last = reference.reference_token(token, 2, 32, 0.0)
        self.assertGreater(max(abs(a - b) for a, b in zip(first, last)), 0.02)
        self.assertLess(abs(last[0] + last[2] - 1.0), 2e-7)
        self.assertLess(abs(last[1] + last[3] - 1.0), 2e-7)

    def test_eps_placement_and_gates(self):
        # eps=1/8 is exactly representable: softmax+eps=5/8 and the first
        # column denominator is 11/8, giving each comb element 5/11.
        pre, post, comb = reference.reference_token([0.0] * 8, 2, 1, 0.125)
        self.assertEqual(pre, [0.625, 0.625])
        self.assertEqual(post, [1.0, 1.0])
        self.assert_vector_close(comb, [5 / 11] * 4)

    def test_statement_example(self):
        pre, post, comb = reference.reference_token(
            [0.5, -0.5, 1.0, -1.0, 0.2, -0.2, 0.4, -0.4], 2, 2, 1e-6
        )
        self.assert_vector_close(pre, [0.6224603, 0.3775417], 1e-6)
        self.assert_vector_close(post, [1.4621172, 0.5378828], 1e-6)
        self.assert_vector_close(comb, [0.450309, 0.549976, 0.549691, 0.450023], 2e-6)

    def test_non_aligned_and_large_h_extreme_finite_logits(self):
        for h in (3, 7, 8, 9, 17, 65, 127, 128):
            values = [-3e38, -1000.0, -80.0, 0.0, 80.0, 1000.0, 3e38]
            token = [values[i % len(values)] for i in range(h * (h + 2))]
            pre, post, comb = reference.reference_token(token, h, 1, 1e-6)
            self.assertEqual((len(pre), len(post), len(comb)), (h, h, h * h))
            self.assertTrue(all(math.isfinite(value) for value in pre + post + comb))
            self.assertTrue(all(value > 0.0 for value in comb))
            self.assertTrue(all(0.0 <= value <= 2.0 for value in post))

    def test_input_is_unchanged_and_invalid_input_rejected(self):
        token = [0.5, -0.5, 1.0, -1.0, 0.2, -0.2, 0.4, -0.4]
        original = token.copy()
        reference.reference_token(token, 2, 20, 1e-6)
        self.assertEqual(token, original)
        with self.assertRaises(ValueError):
            reference.reference_token(token[:-1], 2)
        with self.assertRaises(ValueError):
            reference.reference_token([math.inf] + token[1:], 2)
        for h, iters in ((1, 20), (129, 20), (2, 0), (2, 33)):
            with self.assertRaises(ValueError):
                reference.reference_token(token, h, iters)


class RawFileTest(unittest.TestCase):
    def test_fixture_cli_and_full_output_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "generate", "--output-dir", str(fixture),
                 "--b", "2", "--s", "3", "--h", "3", "--iters", "32", "--eps", "0.001",
                 "--distribution", "wide", "--seed", "11"],
                check=True, capture_output=True, text=True,
            )
            self.assertEqual(Path(completed.stdout.strip()), fixture / "manifest.json")
            manifest = json.loads((fixture / "manifest.json").read_text())
            self.assertEqual(manifest["shape"], [2, 3, 15])
            self.assertEqual((fixture / "mixes.bin").stat().st_size, 2 * 3 * 15 * 4)
            command = [sys.executable, str(SCRIPT), "verify", "--fixture", str(fixture)]
            for name in ("pre", "post", "comb"):
                command.extend([f"--{name}", str(fixture / f"expected_{name}.bin")])
            identical = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertTrue(json.loads(identical.stdout)["passed"])

            # Corrupt only the last matrix element: checking rows/columns or a
            # prefix is insufficient; the comparator must inspect the full file.
            changed = fixture / "comb.bin"
            raw = bytearray((fixture / "expected_comb.bin").read_bytes())
            raw[-4:] = struct.pack("<f", 9.0)
            changed.write_bytes(raw)
            command[-1] = str(changed)
            failed = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 1)
            report = json.loads(failed.stdout)
            self.assertFalse(report["passed"])
            self.assertEqual(report["outputs"]["comb"]["first_mismatch_index"], 53)
            self.assertEqual(report["outputs"]["comb"]["mismatch_count"], 1)

    def test_reject_truncation_extra_bytes_nan_and_strict_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "expected.bin"
            actual = Path(directory) / "actual.bin"
            expected.write_bytes(struct.pack("<2f", 0.0, 0.0))
            for raw in (b"", b"\0" * 4, b"\0" * 12):
                actual.write_bytes(raw)
                result = reference.compare_raw(expected, actual, 2)
                self.assertFalse(result["passed"])
                self.assertIsNone(result["mismatch_count"])
            actual.write_bytes(struct.pack("<2f", 0.0, math.nan))
            result = reference.compare_raw(expected, actual, 2)
            self.assertFalse(result["passed"])
            self.assertEqual(result["nonfinite_count"], 1)
            actual.write_bytes(struct.pack("<2f", 0.0, 0.125))
            result = reference.compare_raw(expected, actual, 2, atol=0.125, rtol=0.0)
            self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
