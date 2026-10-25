"""Native binding checks, run in the sourced ROS/Kalibr environment."""
import importlib
import pathlib
import pickle
import tempfile
import unittest
from types import SimpleNamespace

import aslam_cameras_april as april
import kalibr_common as kc
from kalibr_imu_camera_calibration.reference_track import select_interval

calibration = importlib.import_module('kalibr_camera_calibration.CameraCalibrator')


class StationCalibration(unittest.TestCase):
    def test_window_default_and_serialization(self):
        options = april.AprilgridOptions()
        self.assertEqual(options.subpixWindowHalfSize, 2)
        options.subpixWindowHalfSize = 5
        self.assertEqual(pickle.loads(pickle.dumps(options)).subpixWindowHalfSize, 5)

    def test_target_window_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'target.yaml'
            body = 'target_type: aprilgrid\ntagRows: 6\ntagCols: 6\ntagSize: 0.0352\ntagSpacing: 0.3\n'
            path.write_text(body)
            self.assertEqual(kc.CalibrationTargetParameters(str(path)).getTargetParams()['subpixWindowHalfSize'], 2)
            path.write_text(body + 'subpixWindowHalfSize: 5\n')
            self.assertEqual(kc.CalibrationTargetParameters(str(path)).getTargetParams()['subpixWindowHalfSize'], 5)
            for invalid in ['0', '-1', '11', '2.5', 'true']:
                path.write_text(body + 'subpixWindowHalfSize: ' + invalid + '\n')
                with self.assertRaises(RuntimeError):
                    kc.CalibrationTargetParameters(str(path)).getTargetParams()

    def test_fixed_lenses_cannot_be_reactivated(self):
        camera = object.__new__(calibration.CameraGeometry)
        camera.fixedIntrinsics = True
        states = []
        variable = SimpleNamespace(setActive=states.append)
        camera.dv = SimpleNamespace(projectionDesignVariable=lambda: variable,
                                    distortionDesignVariable=lambda: variable,
                                    shutterDesignVariable=lambda: variable)
        camera.setDvActiveStatus(True, True, True)
        self.assertEqual(states, [False, False, False])
        self.assertTrue(camera.initGeometryFromObservations([object()]))

    def test_longest_interval_preserves_absolute_clock(self):
        times = [1000 + i*.5 for i in range(50)] + [1040 + i*.5 for i in range(100)]
        self.assertEqual(select_interval(times, 2, 20), (1040, 1089.5))

    def test_short_or_invalid_tracks_fail(self):
        for times in [[], [1, 2, 3], [1, 1, 50], [1, float('nan'), 50]]:
            with self.assertRaises(ValueError):
                select_interval(times, 2, 20)


class SharedClockInitialization(unittest.TestCase):
    def test_reference_prior_does_not_lock_individual_camera_delays(self):
        from kalibr_imu_camera_calibration.IccCalibrator import IccCalibrator
        from kalibr_imu_camera_calibration.IccSensors import IccCamera
        import sm
        import aslam_backend as backend
        calibrator = IccCalibrator()
        calibrator.ImuList = [object()]
        cameras, correlated = [], []
        for index in range(6):
            camera = object.__new__(IccCamera)
            camera.T_extrinsic = sm.Transformation()
            def prior(imu, verbose, camera=camera, index=index):
                correlated.append(index)
                camera.timeshiftCamToImuPrior = .002
            camera.findTimeshiftCameraImuPrior = prior
            cameras.append(camera)
        class Initialized(Exception): pass
        def stop(imu): raise Initialized()
        calibrator.CameraChain = SimpleNamespace(camList=cameras, findOrientationPriorCameraChainToImu=stop)
        with self.assertRaises(Initialized):
            calibrator.buildProblem(sharedCameraClock=True)
        self.assertEqual(correlated, [0])
        self.assertEqual([c.timeshiftCamToImuPrior for c in cameras], [.002]*6)
        import incremental_calibration
        problem = incremental_calibration.CalibrationOptimizationProblem()
        for camera in cameras:
            camera.addDesignVariables(problem, noTimeCalibration=False)
            self.assertTrue(camera.cameraTimeToImuTimeDv.isActive())

    def test_interval_prefers_usable_all_camera_coverage(self):
        first = list(range(0, 51))
        second = list(range(60, 101))
        self.assertEqual(select_interval(first + second, 2, 20,
                         required_tracks=[[65, 75, 90]], imu_range=(0, 100)), (60, 100))


class ExtractionObservation:
    def __init__(self, index): self.index, self.cleared = index, False
    def clearImage(self): self.cleared = True


class ExtractionDetector:
    def __init__(self, completed, fail=False): self.completed, self.fail = completed, fail
    def findTarget(self, stamp, image):
        import time
        time.sleep(.01)
        if self.fail:
            raise ValueError('detector failure')
        with self.completed.get_lock(): self.completed.value += 1
        return True, ExtractionObservation(int(stamp))


class BoundedExtraction(unittest.TestCase):
    def test_bounded_images_order_and_worker_failure(self):
        import multiprocessing
        import numpy as np
        from kalibr_common.TargetExtractor import extractCornersFromDataset
        completed = multiprocessing.Value('i', 0)
        def images():
            for index in range(20):
                if index - completed.value >= 4:
                    raise AssertionError('decoded-image backlog exceeded twice the worker count')
                yield index, np.zeros((2, 2), dtype=np.uint8)
        dataset = SimpleNamespace(numImages=lambda: 20, readDataset=images, topic='/test')
        result = extractCornersFromDataset(dataset, ExtractionDetector(completed), multithreading=True, numProcesses=2)
        self.assertEqual([obs.index for obs in result], list(range(20)))
        self.assertTrue(all(obs.cleared for obs in result))
        completed.value = 0
        with self.assertRaisesRegex(ValueError, 'detector failure'):
            extractCornersFromDataset(dataset, ExtractionDetector(completed, fail=True), multithreading=True, numProcesses=2)


class OutlierReadiness(unittest.TestCase):
    def test_short_multicam_sweep_counts_actual_views_per_camera(self):
        views = [SimpleNamespace(rerrs={camera: [object()] for camera in range(6)}) for _ in range(20)]
        self.assertEqual(calibration.camerasReadyForOutlierFiltering(views, 20), set(range(6)))
        views[-1].rerrs[5] = [None, None]
        self.assertEqual(calibration.camerasReadyForOutlierFiltering(views, 20), set(range(5)))


class ReprojectionDiagnostics(unittest.TestCase):
    def test_euclidean_rms_weights_corners_and_preserves_missing_data(self):
        import json
        import numpy as np
        from kalibr_camera_calibration.CameraUtils import exportReprojectionDiagnostics
        def error(x, y):
            return SimpleNamespace(getMeasurement=lambda: np.array([x, y]),
                                   getPredictedMeasurement=lambda: np.zeros(2))
        calibrator = SimpleNamespace(cameras=[None, None], views=[
            SimpleNamespace(timestamp=1.25, rerrs={0: [error(3, 4), None]}),
            SimpleNamespace(timestamp=2.5, rerrs={0: [error(0, 0), error(0, 12)]})])
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'diagnostics.json'
            exportReprojectionDiagnostics(calibrator, str(path))
            result = json.loads(path.read_text())
        self.assertAlmostEqual(result['cam0']['rms_px'], np.sqrt(169 / 3))
        self.assertEqual(result['cam0']['corners'], 3)
        self.assertEqual(result['cam0']['views'][0],
                         dict(timestamp_s=1.25, corners=1, rms_px=5.0, max_px=5.0))
        self.assertEqual(result['cam1'], dict(views=[], corners=0, rms_px=None))


if __name__ == '__main__':
    unittest.main()
