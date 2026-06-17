from pathlib import Path
import tempfile
import unittest

from cai.constitution import ConstitutionError, load_constitution, validate_constitution


ROOT = Path(__file__).resolve().parents[1]


class ConstitutionTests(unittest.TestCase):
    def test_loads_core_constitution(self) -> None:
        constitution = load_constitution(ROOT / "constitutions" / "core.md")

        self.assertEqual(constitution.id, "core-harmlessness")
        self.assertEqual(len(constitution.principles), 8)
        self.assertEqual(constitution.principles[0].id, "safety-legality")
        self.assertIn("harmful or illegal guidance", constitution.principles[0].revision)

    def test_core_constitution_has_no_warnings(self) -> None:
        warnings = validate_constitution(ROOT / "constitutions" / "core.md")

        self.assertEqual(warnings, [])

    def test_rejects_missing_revision(self) -> None:
        bad_constitution = """+++
id = "bad"
title = "Bad"
version = "0.1.0"
description = "Missing revision."
+++

## test-rule: Test Rule

Tags: test

### Critique

This is a critique prompt with enough words to pass length checks.
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.md"
            path.write_text(bad_constitution, encoding="utf-8")

            with self.assertRaisesRegex(ConstitutionError, "Revision"):
                load_constitution(path)


if __name__ == "__main__":
    unittest.main()

