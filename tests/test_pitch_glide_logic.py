import ast
import pathlib
import types
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "pitch-glide-direction-threshold.py"


def load_logic_module() -> types.SimpleNamespace:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SCRIPT_PATH))

    keep_assigns = {
        "SR_FIXED",
        "N_TEST_TRIALS",
        "N_SMALL_REV_TARGET",
        "FINISH_REASON_LABELS",
        "SERIES_1",
        "SERIES_2",
        "FIXED_TRIAL_SERIES",
        "GLIDE_DIR_SERIES_1_CODES",
        "GLIDE_DIR_SERIES_2_CODES",
        "FIXED_GLIDE_DIR_CODES",
    }
    keep_defs = {
        "_trial_codes_to_types",
        "_dir_codes_to_labels",
        "_build_constrained_sequence",
        "build_test_plan",
        "_cosine_ramp_env",
        "rms_normalize",
        "glide_stimulus_linear_ramp_to_center",
        "flat_stimulus",
        "mono_to_stereo_bytes",
        "generate_trial_wav_single",
        "DurationStaircase",
        "finish_reason_label",
    }

    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(alias.name == "streamlit" for alias in node.names):
                continue
            selected_nodes.append(node)
        elif isinstance(node, ast.ImportFrom):
            selected_nodes.append(node)
        elif isinstance(node, ast.Assign):
            names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if names & keep_assigns:
                selected_nodes.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in keep_defs:
            selected_nodes.append(node)

    module = types.SimpleNamespace()
    namespace = {"__builtins__": __builtins__}
    code = compile(ast.Module(body=selected_nodes, type_ignores=[]), str(SCRIPT_PATH), "exec")
    exec(code, namespace)
    for key, value in namespace.items():
        setattr(module, key, value)
    return module


logic = load_logic_module()


class BuildPlanTests(unittest.TestCase):
    def test_pseudorandom_plan_has_expected_counts_and_constraints(self):
        plan = logic.build_test_plan(series_name="擬似ランダム", pseudo_seed=12345, max_consecutive=3)

        self.assertEqual(len(plan["schedule_codes"]), 100)
        self.assertEqual(plan["schedule_codes"].count(1), 40)
        self.assertEqual(plan["schedule_codes"].count(2), 60)
        self.assertEqual(plan["glide_dir_codes"].count(1), 30)
        self.assertEqual(plan["glide_dir_codes"].count(2), 30)

        run = 1
        for prev, cur in zip(plan["schedule_codes"], plan["schedule_codes"][1:]):
            run = run + 1 if prev == cur else 1
            self.assertLessEqual(run, 3)

    def test_fixed_series_uses_frozen_direction_schedule(self):
        plan = logic.build_test_plan(series_name="系列1")
        self.assertEqual(plan["glide_dir_codes"], logic.FIXED_GLIDE_DIR_CODES["系列1"])


class StaircaseTests(unittest.TestCase):
    def test_small_reversal_threshold_uses_last_six_small_phase_levels(self):
        sc = logic.DurationStaircase(
            start_ms=300,
            floor_ms=20,
            ceil_ms=600,
            step_big_ms=40,
            step_small_ms=20,
            switch_after_reversals=4,
        )

        sc.reversals = [
            {"update_index": 1, "level_ms": 340.0, "phase": "big", "step_ms": 40.0},
            {"update_index": 2, "level_ms": 300.0, "phase": "big", "step_ms": 40.0},
            {"update_index": 3, "level_ms": 340.0, "phase": "big", "step_ms": 40.0},
            {"update_index": 4, "level_ms": 300.0, "phase": "big", "step_ms": 40.0},
            {"update_index": 5, "level_ms": 300.0, "phase": "small", "step_ms": 20.0},
            {"update_index": 6, "level_ms": 320.0, "phase": "small", "step_ms": 20.0},
            {"update_index": 7, "level_ms": 300.0, "phase": "small", "step_ms": 20.0},
            {"update_index": 8, "level_ms": 320.0, "phase": "small", "step_ms": 20.0},
            {"update_index": 9, "level_ms": 300.0, "phase": "small", "step_ms": 20.0},
            {"update_index": 10, "level_ms": 320.0, "phase": "small", "step_ms": 20.0},
        ]

        self.assertEqual(sc.n_small_reversals(), 6)
        self.assertEqual(sc.usable_reversal_levels(), [300.0, 320.0, 300.0, 320.0, 300.0, 320.0])
        self.assertEqual(sc.threshold_last6_median(), 310.0)
        self.assertEqual(sc.threshold_last6_mean(), 310.0)


class AudioGenerationTests(unittest.TestCase):
    def test_flat_audio_duration_matches_ramp_plus_steady(self):
        wav_bytes, total_ms = logic.generate_trial_wav_single(
            sr=logic.SR_FIXED,
            f_center=500.0,
            delta=150.0,
            ramp_ms=300,
            steady_ms=0,
            ear="両耳",
            edge_ramp_ms=10,
            target_rms=0.1,
            trial_type="flat",
            direction="up",
        )

        self.assertGreater(len(wav_bytes), 44)
        self.assertEqual(total_ms, 300)
        self.assertEqual(logic.finish_reason_label("internal_error"), "内部エラー")


if __name__ == "__main__":
    unittest.main()
