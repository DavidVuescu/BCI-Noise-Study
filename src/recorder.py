"""
UDP recorder for the Unicorn Hybrid Black via the Unicorn UDP Interface.

Receives packets in a background thread, stores samples in a pre-allocated
numpy buffer with wall-clock timestamps, and saves to disk on stop.

Usage:
    from src.config import load_config
    from src.recorder import Recorder

    cfg = load_config()
    rec = Recorder(cfg)
    rec.start()
    # ... do other things (run stimulus, sleep, etc.) ...
    rec.stop()
    rec.save(output_dir="data/raw", subject_id="pilot01", condition="control")
"""
from __future__ import annotations

import json
import socket
import struct
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---- Constants from Unicorn UDP protocol docs ---------------------------
# Order of channels in each packet (matches g.tec docs section 18.3):
CHANNEL_NAMES = [
    "Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8",  # EEG 1-8
    "AccX", "AccY", "AccZ",  # Accelerometer
    "GyroX", "GyroY", "GyroZ",  # Gyroscope
    "Battery", "Counter", "Validation",
]
N_CHANNELS = 17
BYTES_PER_SAMPLE = 68  # 17 channels * 4 bytes (float32)
COUNTER_INDEX = 15  # Position of counter channel in each sample
STRUCT_FORMAT = "<17f"  # Little-endian, 17 float32s

# Socket recv timeout. Sets max latency for stop() to take effect.
SOCKET_TIMEOUT_S = 0.1


class Recorder:
    """Threaded UDP recorder for the Unicorn UDP Interface."""

    def __init__(self, config: dict, max_duration_s: float = 900.0):
        """
        Args:
            config: Loaded config dict (see config.yaml).
            max_duration_s: Maximum recording duration. Used to pre-allocate
                the sample buffer. Default 900s (15 min) - safe for any
                single condition. Going over this raises BufferError.
        """
        self.cfg = config
        self.host = config["device"]["udp_host"]
        self.port = config["device"]["udp_port"]
        self.sample_rate = config["device"]["sampling_rate"]

        # Pre-allocate buffers. Pre-allocation matters because:
        # 1. No reallocation during hot loop = predictable latency
        # 2. We know exact memory footprint upfront
        # 3. np.zeros is page-mapped lazily on Linux; on Windows it actually
        #    touches the memory, but for 15min*250Hz*17ch*4B ~ 6 MB this is trivial
        max_samples = int(max_duration_s * self.sample_rate * 1.1)  # 10% headroom
        self._samples = np.zeros((max_samples, N_CHANNELS), dtype=np.float32)
        self._timestamps = np.zeros(max_samples, dtype=np.float64)  # wall-clock, needs float64 precision
        self._max_samples = max_samples

        # Write position. Only the recv thread writes it. Main thread reads it
        # under the lock to get a consistent snapshot.
        self._write_idx = 0

        # Threading primitives
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

        # Metadata captured at start/stop
        self._start_time_wall: float | None = None
        self._start_time_iso: str | None = None
        self._stop_time_wall: float | None = None
        self._first_counter: float | None = None
        self._last_counter: float | None = None

    # ---- Public API ----------------------------------------------------

    def start(self) -> None:
        """Start the background receive thread. Non-blocking."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Recorder already running")

        self._stop_event.clear()
        self._write_idx = 0
        self._first_counter = None
        self._last_counter = None
        self._start_time_wall = time.time()
        self._start_time_iso = datetime.now(timezone.utc).isoformat()

        self._thread = threading.Thread(
            target=self._receive_loop,
            name="UnicornUDPReceiver",
            daemon=True,  # Dies with main thread if main crashes
        )
        self._thread.start()

    def stop(self, join_timeout_s: float = 2.0) -> None:
        """Signal the thread to stop and wait for it to finish."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=join_timeout_s)

        if self._thread.is_alive():
            # Thread didn't exit. This shouldn't happen unless UnicornUDP
            # is misbehaving or the socket timeout is misconfigured.
            print("WARNING: Recorder thread did not exit cleanly within "
                  f"{join_timeout_s}s. Data may be incomplete.")

        self._stop_time_wall = time.time()
        self._thread = None

    def stats(self) -> dict:
        """Snapshot of current recording state. Safe to call while running."""
        with self._lock:
            n = self._write_idx
            first = self._first_counter
            last = self._last_counter

        elapsed = (time.time() - self._start_time_wall) if self._start_time_wall else 0.0

        # Expected vs actual samples based on counter
        if first is not None and last is not None and n > 0:
            expected = int(last - first) + 1
            dropped = expected - n
            dropout_rate = dropped / expected if expected > 0 else 0.0
        else:
            expected = dropped = 0
            dropout_rate = 0.0

        return {
            "samples_received": n,
            "elapsed_s": elapsed,
            "actual_rate_hz": n / elapsed if elapsed > 0 else 0.0,
            "first_counter": first,
            "last_counter": last,
            "samples_expected_by_counter": expected,
            "samples_dropped": dropped,
            "dropout_rate": dropout_rate,
        }

    def get_arrays(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return analysis-ready arrays.

        Returns:
            eeg:        (n, 17) float32, raw values as received.
            acq_time:   (n,) float64, seconds since first sample, derived
                        from the DEVICE counter (jitter-free, 4.000ms grid).
            wall_anchor: float, time.time() of the first received sample.
                        Use this to map stimulus marker timestamps onto
                        acq_time: marker_acq = marker_walltime - wall_anchor.

        Why counter-derived time: the UDP interface delivers samples in
        bursts, so per-packet receive timestamps jitter by tens of ms.
        The device counter increments at the true 250 Hz acquisition rate
        before any network involvement, so (counter - counter0)/rate is a
        perfectly uniform, jitter-free time base.
        """
        with self._lock:
            n = self._write_idx
            eeg = self._samples[:n].copy()
            counters = self._samples[:n, COUNTER_INDEX].astype(np.float64).copy()
            wall_anchor = float(self._timestamps[0]) if n > 0 else 0.0

        if n == 0:
            return eeg, np.zeros(0, dtype=np.float64), wall_anchor

        # Counter-derived uniform time axis, in seconds, starting at 0.
        acq_time = (counters - counters[0]) / float(self.sample_rate)

        # Integrity check: counter must be strictly monotonic +1.
        # We surface violations rather than silently trusting the grid.
        deltas = np.diff(counters)
        bad = np.where(deltas != 1.0)[0]
        if bad.size > 0:
            print(f"WARNING: device counter has {bad.size} non-unit step(s); "
                  f"first at sample {int(bad[0])} "
                  f"(delta={deltas[bad[0]]:.0f}). Acquisition-time grid "
                  f"assumes uniform sampling and may be locally wrong here.")

        return eeg, acq_time, wall_anchor

    def save(self, output_dir: str | Path, subject_id: str, condition: str) -> dict:
        """Persist EEG, timestamps, and metadata to disk.

        Three files are written:
            sub-<id>_cond-<cond>_eeg.npy        - (n_samples, 17) float32
            sub-<id>_cond-<cond>_timestamps.npy - (n_samples,) float64 wall-clock
            sub-<id>_cond-<cond>_meta.json      - all metadata

        Returns:
            The metadata dict that was written.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Truncate to actual data. The buffer is allocated generously; we only
        # save what was actually received.
        with self._lock:
            n = self._write_idx
            eeg = self._samples[:n].copy()
            ts = self._timestamps[:n].copy()

        stem = f"sub-{subject_id}_cond-{condition}"
        eeg_path = output_dir / f"{stem}_eeg.npy"
        ts_path = output_dir / f"{stem}_timestamps.npy"
        meta_path = output_dir / f"{stem}_meta.json"

        # Counter-derived (jitter-free) acquisition time + wall anchor.
        eeg2, acq_time, wall_anchor = self.get_arrays()
        acq_path = output_dir / f"{stem}_acqtime.npy"

        np.save(eeg_path, eeg)
        np.save(ts_path, ts)  # raw receive timestamps (forensic)
        np.save(acq_path, acq_time)  # canonical analysis time axis

        meta = {
            "subject_id": subject_id,
            "condition": condition,
            "start_time_iso_utc": self._start_time_iso,
            "start_time_unix": self._start_time_wall,
            "stop_time_unix": self._stop_time_wall,
            "duration_s": (self._stop_time_wall - self._start_time_wall)
            if (self._stop_time_wall and self._start_time_wall) else None,
            "wall_clock_anchor_unix": wall_anchor,
            "n_samples": int(n),
            "sample_rate_hz": self.sample_rate,
            "n_channels": N_CHANNELS,
            "channel_names": CHANNEL_NAMES,
            "first_counter": self._first_counter,
            "last_counter": self._last_counter,
            "samples_expected_by_counter": int(self._last_counter - self._first_counter + 1)
            if (self._last_counter is not None
                and self._first_counter is not None) else None,
            "samples_dropped": (int(self._last_counter - self._first_counter + 1) - int(n))
            if (self._last_counter is not None
                and self._first_counter is not None) else None,
            "config_snapshot": self.cfg,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return meta

    # ---- Internal: the receive loop ------------------------------------

    def _receive_loop(self) -> None:
        """Runs in the background thread. Receives UDP packets until stopped."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        sock.settimeout(SOCKET_TIMEOUT_S)

        try:
            while not self._stop_event.is_set():
                try:
                    data, _addr = sock.recvfrom(2048)
                except socket.timeout:
                    # No packet within timeout window. Loop back to check
                    # stop_event. This is normal and expected.
                    continue

                # Record wall-clock timestamp ASAP after receive.
                # Important: time.time() here, not perf_counter(), because
                # we need to correlate with stimulus markers across the
                # process. time.time() is wall-clock; perf_counter is
                # monotonic-but-arbitrary-origin.
                ts = time.time()

                if len(data) != BYTES_PER_SAMPLE:
                    # Malformed packet. Skip silently for now; a future
                    # version could log these.
                    continue

                # Parse 17 float32s. struct.unpack returns a tuple; we
                # convert to ndarray for assignment.
                values = struct.unpack(STRUCT_FORMAT, data)

                with self._lock:
                    idx = self._write_idx
                    if idx >= self._max_samples:
                        # Out of buffer. Drop the packet rather than overwrite
                        # earlier samples. This shouldn't happen if max_duration_s
                        # is set correctly.
                        continue

                    self._samples[idx, :] = values
                    self._timestamps[idx] = ts

                    counter = values[COUNTER_INDEX]
                    if self._first_counter is None:
                        self._first_counter = counter
                    self._last_counter = counter

                    self._write_idx = idx + 1
        finally:
            sock.close()