"""Smoke test for Recorder: starts, stops, saves a stub recording.

Run with UnicornUDP streaming on 127.0.0.1:1000.
Records 5 seconds, saves to data/raw, prints stats.
"""
import time
from pathlib import Path

from src.config import load_config
from src.recorder import Recorder


def main():
    cfg = load_config()
    rec = Recorder(cfg)

    print("Starting recorder...")
    rec.start()

    # Record for 5 seconds, printing stats every second
    for i in range(5):
        time.sleep(1.0)
        stats = rec.stats()
        print(f"  t={i + 1}s  n={stats['samples_received']:4d}  "
              f"rate={stats['actual_rate_hz']:.1f}Hz  "
              f"dropped={stats['samples_dropped']}")

    print("Stopping...")
    rec.stop()

    print("Saving...")
    meta = rec.save("data/raw", subject_id="smoketest", condition="recorder")

    print(f"\nFiles written to data/raw/")
    print(f"Duration: {meta['duration_s']:.3f}s")
    print(f"Samples: {meta['n_samples']}")
    print(f"Dropped: {meta['samples_dropped']}")

    # Verify we can load the files back
    import numpy as np
    eeg = np.load("data/raw/sub-smoketest_cond-recorder_eeg.npy")
    ts = np.load("data/raw/sub-smoketest_cond-recorder_timestamps.npy")
    print(f"\nReloaded shape: eeg={eeg.shape}, timestamps={ts.shape}")
    print(f"Timestamp range: {ts[-1] - ts[0]:.3f}s")
    print(f"Median inter-sample interval: {np.median(np.diff(ts)) * 1000:.2f}ms "
          f"(expected ~4.00ms at 250Hz)")

    # Validate counter-derived acquisition time
    eeg2 = np.load("data/raw/sub-smoketest_cond-recorder_eeg.npy")
    acq = np.load("data/raw/sub-smoketest_cond-recorder_acqtime.npy")
    recv = np.load("data/raw/sub-smoketest_cond-recorder_timestamps.npy")

    acq_diffs_ms = np.diff(acq) * 1000.0
    recv_diffs_ms = np.diff(recv) * 1000.0
    print(f"\n=== TIME AXIS COMPARISON ===")
    print(f"  RECEIVE timestamps (network, jittery):")
    print(f"    mean={recv_diffs_ms.mean():.3f}ms  std={recv_diffs_ms.std():.3f}ms  "
          f"max={recv_diffs_ms.max():.3f}ms")
    print(f"  COUNTER-derived acquisition time (device, clean):")
    print(f"    mean={acq_diffs_ms.mean():.4f}ms  std={acq_diffs_ms.std():.4f}ms  "
          f"max={acq_diffs_ms.max():.4f}ms")
    print(f"  Expected: counter mean=4.0000ms, std=0.0000ms (perfectly uniform)")
    print(f"  Total recording length: receive={recv[-1] - recv[0]:.3f}s  "
          f"acq={acq[-1]:.3f}s  (should match closely)")


if __name__ == "__main__":
    main()