import unittest

import numpy as np
import pandas as pd

from scripts import relapse_two_stage_transition as transition


class CalibrationStrategyTests(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame(
            {
                "Interval_Name": ["1M->3M", "3M->6M", "12M->18M", "18M->24M"],
            }
        )
        self.test_df = pd.DataFrame(
            {
                "Interval_Name": ["1M->3M", "6M->12M", "18M->24M"],
            }
        )
        self.raw_train = np.array([-1.0, -0.2, 0.1, 0.7], dtype=float)
        self.raw_test = np.array([-0.3, 0.2, 0.9], dtype=float)

    def test_phase_strategy_uses_early_late_design(self):
        strategy = transition.CALIBRATION_STRATEGIES["phase"]
        tr, te = strategy.build_frames(self.train_df, self.test_df, self.raw_train, self.raw_test)
        self.assertIn("Score_Logit", tr.columns)
        self.assertIn("Phase_HighRisk", tr.columns)
        self.assertIn("Phase_Late", tr.columns)
        self.assertEqual(int(tr["Phase_HighRisk"].sum()), 2)
        self.assertEqual(int(te["Phase_Late"].sum()), 1)

    def test_interval_strategy_expands_interval_intercepts(self):
        strategy = transition.CALIBRATION_STRATEGIES["interval"]
        tr, te = strategy.build_frames(self.train_df, self.test_df, self.raw_train, self.raw_test)
        self.assertIn("Interval_1M->3M", tr.columns)
        self.assertIn("Interval_18M->24M", tr.columns)
        self.assertEqual(tr.shape[0], 4)
        self.assertEqual(te.shape[0], 3)


class StopRuleTests(unittest.TestCase):
    def test_stop_rule_counts_window_gain_and_ci_shift(self):
        summary_df = pd.DataFrame(
            [
                {
                    "Group": "current_champion_ref",
                    "AUC": 0.850,
                    "PR_AUC": 0.349,
                    "Brier": 0.063,
                    "Calibration_Intercept": 0.74,
                    "Calibration_Slope": 1.46,
                },
                {
                    "Group": "candidate_a",
                    "AUC": 0.8495,
                    "PR_AUC": 0.3515,
                    "Brier": 0.0635,
                    "Calibration_Intercept": 0.80,
                    "Calibration_Slope": 1.50,
                },
            ]
        )
        per_window_df = pd.DataFrame(
            [
                {"Group": "current_champion_ref", "Interval_Name": "1M->3M", "PR_AUC": 0.26},
                {"Group": "current_champion_ref", "Interval_Name": "3M->6M", "PR_AUC": 0.44},
                {"Group": "current_champion_ref", "Interval_Name": "6M->12M", "PR_AUC": 0.45},
                {"Group": "candidate_a", "Interval_Name": "1M->3M", "PR_AUC": 0.264},
                {"Group": "candidate_a", "Interval_Name": "3M->6M", "PR_AUC": 0.445},
                {"Group": "candidate_a", "Interval_Name": "6M->12M", "PR_AUC": 0.455},
            ]
        )
        delta_boot_df = pd.DataFrame(
            [
                {
                    "Metric": "prauc",
                    "Delta": 0.0025,
                    "CI_Low": -0.002,
                    "CI_High": 0.009,
                    "OneSided_P": 0.16,
                    "Group": "candidate_a",
                    "Baseline": "current_champion_ref",
                }
            ]
        )
        out = transition.build_anchor_champion_stop_rule(
            summary_df, per_window_df, delta_boot_df, "current_champion_ref"
        )
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(row["Improved_Window_Count"], 3)
        self.assertTrue(bool(row["Bootstrap_CI_RightShift"]))
        self.assertGreaterEqual(int(row["Pass_Count"]), 3)


if __name__ == "__main__":
    unittest.main()
