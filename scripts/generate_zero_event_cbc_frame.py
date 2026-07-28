import logging
from pathlib import Path

import bilby
import numpy as np


bilby.core.utils.setup_logger(log_level=logging.WARNING)

# Frame config
job_number = 1
duration = 2048
sampling_frequency = 32
minimum_frequency = 10
detectors_to_show = ["H1", "L1"]
num_values_to_show = 100


def build_interferometers(frame_start: float):
    interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
    for ifo in interferometers:
        ifo.minimum_frequency = minimum_frequency
        ifo.sampling_frequency = sampling_frequency
        ifo.duration = duration
    interferometers.set_strain_data_from_zero_noise(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=frame_start,
    )
    return interferometers


def main():
    frame_start = (job_number - 1) * duration
    frame_end = frame_start + duration

    print("Generating zero-event CBC frame")
    print(f"Frame window: [{frame_start}, {frame_end}]")
    print(f"Sampling frequency: {sampling_frequency} Hz")
    print(f"Duration: {duration} s")
    print(f"Detectors: {detectors_to_show}")
    print()

    interferometers = build_interferometers(frame_start)

    for ifo in interferometers:
        if ifo.name not in detectors_to_show:
            continue

        strain = np.asarray(ifo.time_domain_strain, dtype=np.float64)
        head = strain[:num_values_to_show]

        print(f"{ifo.name}:TEST_INJ")
        print(f"  total_length = {len(strain)}")
        print(f"  mean = {strain.mean():.10e}")
        print(f"  std = {strain.std():.10e}")
        print(f"  first {len(head)} values:")
        print(head)
        print()


if __name__ == "__main__":
    main()
