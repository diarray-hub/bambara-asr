"""Small checks for paired book resampling and review-case selection."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from estimate_uncertainty import run_bootstrap
from eval_common import cohort, score
from select_error_review import choose_cases


class EvaluationChecks(unittest.TestCase):
    def test_paired_book_bootstrap_uses_shared_draws(self):
        rows = [
            {"id": f"u{i}", "text": "a b", "book": "book-a" if i < 2 else "book-b", "speaker_age": 8 if i < 2 else 12}
            for i in range(4)
        ]
        predictions = {
            "soloni": [{"id": row["id"], "hypothesis": "a b"} for row in rows],
            "quartznet": [{"id": row["id"], "hypothesis": "a c"} for row in rows],
        }
        summary, replicates = run_bootstrap(rows, predictions, 16, 7)
        paired = [item for item in summary if item["model"] == "soloni_minus_quartznet"
                  and item["cohort"] == "overall" and item["metric"] == "wer"]
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["estimate"], -0.5)
        self.assertEqual(paired[0]["lower_95"], -0.5)
        self.assertEqual(paired[0]["upper_95"], -0.5)
        self.assertEqual(paired[0]["n_books"], 2)
        self.assertTrue(any(item["model"] == "soloni_minus_quartznet" for item in replicates))
        self.assertEqual(cohort(rows[0]), "under_10")

    def test_scoring_normalization_and_error_review_selection(self):
        self.assertEqual(score("Bɛ!", "bɛ")["word_errors"], 0)
        cases = [{"id": f"u{i:02d}", "character_errors": i + 1, "cer": (i + 1) / 50}
                 for i in range(40)]
        selected = choose_cases(cases)
        self.assertEqual(len(selected), 30)
        self.assertEqual(len({item["id"] for item in selected}), 30)
        self.assertEqual({band: sum(item["band"] == band for item in selected)
                          for band in ("highest", "middle", "lowest_positive")},
                         {"highest": 10, "middle": 10, "lowest_positive": 10})


if __name__ == "__main__":
    unittest.main()
