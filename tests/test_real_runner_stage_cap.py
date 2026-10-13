"""An incomplete stage cannot authorize dependent robot motions."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from core.runners.real import RealEpisodeRunner
from core.v0_types import Subgoal, V0Config


def _stage(motion: str) -> Subgoal:
    return Subgoal(
        id=motion.lower(),
        target="rose",
        affordance="stem",
        motion=motion,
        description=f"{motion} the rose.",
        completion=f"The rose is visibly {motion.lower()}ed.",
    )


class RealRunnerStageCapTests(unittest.TestCase):
    def _runner(self, run_dir: str, stages: list[str], tokens: list[str], task_success_check=None):
        session = SimpleNamespace(
            get_observation=lambda: {
                "agentview": np.zeros((8, 8, 3), dtype=np.uint8),
                "ee_pose": [0.3, 0.0, 0.2],
                "gripper_width": 0.08,
            }
        )
        controller = SimpleNamespace(
            z_floor_m=0.0,
            gripper_closed=False,
            step=Mock(
                side_effect=lambda token, **kwargs: SimpleNamespace(
                    done=token == "DONE", gripper_closed=False
                )
            ),
        )
        planner = SimpleNamespace(
            plan=lambda *args, **kwargs: ([_stage(motion) for motion in stages], "plan")
        )
        decisions = Mock(
            side_effect=[SimpleNamespace(token=token, payload={}) for token in tokens]
        )
        logger = SimpleNamespace(
            run_dir=Path(run_dir),
            write_plan=Mock(),
            log_step=Mock(),
            close=Mock(return_value=Path(run_dir) / "rollout.mp4"),
            write_summary=Mock(),
        )
        return RealEpisodeRunner(
            session=session,
            controller=controller,
            planner=planner,
            controls=SimpleNamespace(controller=SimpleNamespace(decide=decisions)),
            logger=logger,
            config=V0Config(max_subgoal_steps=2, max_replans=0, video_fps=1.0),
            task="put a rose into the vase",
            gripper_color="black",
            max_steps=10,
            loop_period_s=0.0,
            use_wrist_image=False,
            debug=False,
            task_success_check=task_success_check,
        )

    def test_grasp_timeout_stops_before_any_lift_decision(self) -> None:
        predicate = Mock(return_value=True)
        with tempfile.TemporaryDirectory() as run_dir:
            runner = self._runner(
                run_dir, ["GRASP", "LIFT"], ["MV_DOWN", "MV_DOWN"], predicate
            )
            result = runner.run()

        predicate.assert_not_called()

        self.assertFalse(result.success)
        self.assertEqual(result.end_reason, "subgoal_step_cap_exceeded")
        self.assertEqual(result.steps, 2)
        self.assertEqual(
            [call.args[0] for call in runner.controller.step.call_args_list],
            ["RELEASE", "MV_DOWN", "MV_DOWN"],
        )
        contexts = [call.kwargs["ctx"] for call in runner.controls.controller.decide.call_args_list]
        self.assertEqual([ctx.subgoal.motion for ctx in contexts], ["GRASP", "GRASP"])
        summary = runner.logger.write_summary.call_args.args[0]
        self.assertEqual(summary["steps"], runner.logger.log_step.call_count)
        self.assertEqual(summary["end_reason"], "subgoal_step_cap_exceeded")
        self.assertFalse(runner.logger.close.call_args.kwargs["success"])
        self.assertFalse(summary["plan_complete"])
        self.assertIsNone(summary["environment_success"])

    def test_final_stage_timeout_does_not_count_an_unexecuted_step(self) -> None:
        with tempfile.TemporaryDirectory() as run_dir:
            runner = self._runner(run_dir, ["GRASP"], ["MV_DOWN", "MV_DOWN"])
            result = runner.run()

        self.assertFalse(result.success)
        self.assertEqual(result.end_reason, "subgoal_step_cap_exceeded")
        self.assertEqual(result.steps, 2)
        self.assertEqual(runner.logger.log_step.call_count, 2)

    def test_done_at_step_cap_advances_and_resets_budget(self) -> None:
        with tempfile.TemporaryDirectory() as run_dir:
            runner = self._runner(
                run_dir, ["GRASP", "LIFT"], ["GRASP", "DONE", "MV_UP", "DONE"]
            )
            runner.session.success = Mock(return_value=False)
            result = runner.run()

        # Session methods are never auto-discovered; checking is opt-in.
        runner.session.success.assert_not_called()
        self.assertTrue(result.success)
        self.assertEqual(result.end_reason, "plan_complete")
        self.assertEqual(result.steps, 4)
        contexts = [call.kwargs["ctx"] for call in runner.controls.controller.decide.call_args_list]
        self.assertEqual([ctx.subgoal.motion for ctx in contexts], ["GRASP", "GRASP", "LIFT", "LIFT"])
        self.assertEqual([ctx.subgoal_step_idx for ctx in contexts], [0, 1, 0, 1])
        summary = runner.logger.write_summary.call_args.args[0]
        self.assertTrue(summary["plan_complete"])
        self.assertIsNone(summary["environment_success"])

    def test_independent_predicate_gates_success_for_completed_plan(self) -> None:
        for environment_success in (True, False):
            with self.subTest(environment_success=environment_success):
                predicate = Mock(return_value=environment_success)
                with tempfile.TemporaryDirectory() as run_dir:
                    runner = self._runner(
                        run_dir, ["GRASP", "LIFT"], ["DONE", "DONE"], predicate
                    )
                    result = runner.run()

                predicate.assert_called_once_with()
                self.assertEqual(result.success, environment_success)
                self.assertEqual(
                    result.end_reason,
                    "plan_complete" if environment_success else "task_success_check_failed",
                )
                self.assertEqual(result.steps, 2)
                summary = runner.logger.write_summary.call_args.args[0]
                self.assertTrue(summary["plan_complete"])
                self.assertIs(summary["environment_success"], environment_success)
                self.assertIs(summary["success"], environment_success)
                self.assertIs(runner.logger.close.call_args.kwargs["success"], environment_success)
                records = [call.kwargs["record"] for call in runner.logger.log_step.call_args_list]
                self.assertNotIn("ok", records[0])
                self.assertTrue(records[-1]["done"])
                self.assertEqual(bool(records[-1].get("ok")), environment_success)



class DualRunnerStageCapTests(unittest.TestCase):
    """The dual-arm runner follows the same rule: a capped stage never advances."""

    def test_grasp_timeout_ends_the_episode_instead_of_advancing_to_lift(self) -> None:
        from core.runners.dual import DualEpisodeRunner

        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        session = SimpleNamespace(
            get_observation=lambda: {
                "agentview": frame, "wrist_left": frame, "wrist_right": frame,
            }
        )
        controllers = {
            side: SimpleNamespace(z_floor_m=0.0, step=Mock(return_value=None))
            for side in ("left", "right")
        }
        planner = SimpleNamespace(
            plan=lambda *args, **kwargs: (
                {"left": [_stage("GRASP"), _stage("LIFT")], "right": []},
                "plan",
            )
        )
        agent = SimpleNamespace(decide=Mock(), verify_task=Mock())
        with tempfile.TemporaryDirectory() as run_dir:
            logger = SimpleNamespace(
                run_dir=Path(run_dir),
                write_plan=Mock(),
                log_step=Mock(),
                close=Mock(return_value=Path(run_dir) / "rollout.mp4"),
                write_summary=Mock(),
            )
            runner = DualEpisodeRunner(
                session=session,
                controllers=controllers,
                planner=planner,
                controller_agent=agent,
                logger=logger,
                # A zero budget caps the first stage before any decision is made.
                config=V0Config(max_subgoal_steps=0, max_replans=0, video_fps=1.0),
                task="put a rose into the vase",
                max_steps=5,
                loop_period_s=0.0,
                debug=False,
            )
            result = runner.run()

        self.assertFalse(result.success)
        self.assertEqual(result.end_reason, "left: subgoal_step_cap_exceeded")
        agent.decide.assert_not_called()
        agent.verify_task.assert_not_called()
        self.assertEqual(runner.arms["left"].index, 0, "the capped GRASP must not advance")
        # Only the startup RELEASE ran; no LIFT/PLACE motion was ever commanded.
        self.assertEqual(
            [call.args[0] for call in controllers["left"].step.call_args_list], ["RELEASE"]
        )


if __name__ == "__main__":
    unittest.main()
