"""Select a continuous visual initialization interval on the original clock."""
import math


def select_interval(timestamps, max_gap, min_duration, required_tracks=(), imu_range=None):
    if not math.isfinite(max_gap) or max_gap <= 0 or not math.isfinite(min_duration) or min_duration <= 0:
        raise ValueError('Reference gap and minimum duration must be positive and finite')
    times = list(timestamps)
    if len(times) < 3 or any(not math.isfinite(t) for t in times):
        raise ValueError('Reference camera has too few finite observations')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Reference camera timestamps must be strictly increasing')
    intervals, start = [], times[0]
    for previous, current in zip(times, times[1:]):
        if current - previous > max_gap:
            intervals.append((start, previous))
            start = current
    intervals.append((start, times[-1]))
    if imu_range is not None:
        intervals = [(max(start, imu_range[0]), min(end, imu_range[1])) for start, end in intervals]
    intervals = [(start, end) for start, end in intervals
                 if all(sum(start <= t <= end for t in track) >= 3 for track in required_tracks)]
    if not intervals:
        raise ValueError('No reference-camera interval contains enough observations from every camera')
    start, end = max(intervals, key=lambda pair: pair[1] - pair[0])
    if end - start < min_duration:
        raise ValueError('No sufficiently long continuous reference-camera track; keep the target visible during transitions')
    return start, end
