import sm

import numpy as np
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import copy
import cv2

def _initialize_worker(detector, clear_images, no_transformation):
    global _worker_detector, _clear_images, _no_transformation
    _worker_detector, _clear_images, _no_transformation = detector, clear_images, no_transformation


def _extract_task(task):
    index, (stamp, image) = task
    method = _worker_detector.findTargetNoTransformation if _no_transformation else _worker_detector.findTarget
    success, observation = method(stamp, np.asarray(image))
    if _clear_images:
        observation.clearImage()
    return index, observation if success else None


def extractCornersFromDataset(dataset, detector, multithreading=False, numProcesses=None, clearImages=True, noTransformation=False):
    print("Extracting calibration target corners")    
    targetObservations = []
    numImages = dataset.numImages()
    
    # prepare progess bar
    iProgress = sm.Progress2(numImages)
    iProgress.sample()
            
    if multithreading:   
        if not numProcesses:
            numProcesses = max(1,multiprocessing.cpu_count()-1)
        tasks = iter(enumerate(dataset.readDataset()))
        # Limit outstanding decoded images as well as workers. A full queue of
        # 1080p images can exhaust a station before any extraction starts.
        with ProcessPoolExecutor(max_workers=numProcesses,
                                 mp_context=multiprocessing.get_context('fork'),
                                 initializer=_initialize_worker,
                                 initargs=(copy.copy(detector), clearImages, noTransformation)) as pool:
            pending = set()
            for _ in range(2 * numProcesses):
                task = next(tasks, None)
                if task is not None:
                    pending.add(pool.submit(_extract_task, task))
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    index, observation = future.result()
                    if observation is not None:
                        targetObservations.append((index, observation))
                    iProgress.sample()
                    task = next(tasks, None)
                    if task is not None:
                        pending.add(pool.submit(_extract_task, task))
        targetObservations = [obs for _, obs in sorted(targetObservations, key=lambda pair: pair[0])]

    #single threaded implementation
    else:
        for timestamp, image in dataset.readDataset():
            if noTransformation:
                success, observation = detector.findTargetNoTransformation(timestamp, np.array(image))
            else:
                success, observation = detector.findTarget(timestamp, np.array(image))
            if clearImages:
                observation.clearImage()
            if success == 1:
                targetObservations.append(observation)
            iProgress.sample()

    if len(targetObservations) == 0:
        print("\r")
        sm.logFatal("No corners could be extracted for camera {0}! Check the calibration target configuration and dataset.".format(dataset.topic))
    else:    
        print("\r  Extracted corners for %d images (of %d images)                              " % (len(targetObservations), numImages))

    #close all opencv windows that might be open
    cv2.destroyAllWindows()
    
    return targetObservations
