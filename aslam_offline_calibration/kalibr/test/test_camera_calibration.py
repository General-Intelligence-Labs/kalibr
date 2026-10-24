"""Run under the sourced Kalibr ROS environment with python3 -m unittest."""
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch


calibration = importlib.import_module('kalibr_camera_calibration.CameraCalibrator')


class RejectedViewLifecycle(unittest.TestCase):
    def make_calibrator(self, batches=2, accepted=False, iterations=50,
                        batches_after=None):
        calibrator = object.__new__(calibration.CameraCalibration)
        calibrator.cameras = []
        calibrator.target = object()
        calibrator.baselines = []
        calibrator.useBlakeZissermanMest = False
        calibrator.optimizerOptions = SimpleNamespace(maxIterations=50)
        calibrator.views = [object() for _ in range(batches)]
        result = SimpleNamespace(numIterations=iterations, batchAccepted=accepted)
        counts = iter([batches, batches if batches_after is None else batches_after])
        calibrator.estimator = SimpleNamespace(
            getNumBatches=lambda: next(counts),
            addBatch=lambda problem, force: result)
        return calibrator

    def add_view(self, calibrator, force=False):
        problem = object()
        with patch.object(calibration.CalibrationTargetOptimizationProblem,
                          'fromTargetViewObservations', return_value=problem):
            return calibrator.addTargetView(123.0, [], None, force), problem

    def test_rejected_failed_view_preserves_fit_and_next_good_view_can_be_added(self):
        calibrator = self.make_calibrator()
        previous = list(calibrator.views)
        success, _ = self.add_view(calibrator)
        self.assertFalse(success)
        self.assertEqual(calibrator.views, previous)
        calibrator.estimator = SimpleNamespace(
            getNumBatches=lambda: len(previous),
            addBatch=lambda problem, force: SimpleNamespace(
                numIterations=3, batchAccepted=True))
        success, problem = self.add_view(calibrator)
        self.assertTrue(success)
        self.assertEqual(calibrator.views, previous + [problem])

    def test_unsafe_nonconvergence_still_restarts(self):
        scenarios = [dict(batches=0), dict(accepted=True),
                     dict(batches_after=3), dict(force=True)]
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                options = dict(scenario)
                force = options.pop('force', False)
                calibrator = self.make_calibrator(**options)
                previous = list(calibrator.views)
                with self.assertRaises(calibration.OptimizationDiverged):
                    self.add_view(calibrator, force)
                self.assertEqual(calibrator.views, previous)

    def test_convergent_information_rejection_does_not_add_a_view(self):
        calibrator = self.make_calibrator(iterations=3)
        previous = list(calibrator.views)
        success, _ = self.add_view(calibrator)
        self.assertFalse(success)
        self.assertEqual(calibrator.views, previous)


if __name__ == '__main__':
    unittest.main()
