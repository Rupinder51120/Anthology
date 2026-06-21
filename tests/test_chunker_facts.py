import unittest
from src.ingestion.chunker import is_valuable_short_fact

class TestShortFactPreservation(unittest.TestCase):
    def test_preserve_facts(self):
        preserve_cases = [
            "Accuracy = 98.7%",
            "BLEU = 42.1",
            "F1 score: 95.4",
            "p < 0.05",
            "AUC = 0.91",
            "mAP = 78.2",
            "95.4%",
            "0.001",
            "p=0.01",
        ]
        for case in preserve_cases:
            with self.subTest(case=case):
                self.assertTrue(is_valuable_short_fact(case), f"Should preserve: {case}")

    def test_discard_noise(self):
        discard_cases = [
            "Page 7",
            "Table 2",
            "Figure 3",
            "2024",
            "Introduction",
            "Results",
            "100",
            "The",
            "  ",
            "Section 1",
        ]
        for case in discard_cases:
            with self.subTest(case=case):
                self.assertFalse(is_valuable_short_fact(case), f"Should discard: {case}")

if __name__ == "__main__":
    unittest.main()
