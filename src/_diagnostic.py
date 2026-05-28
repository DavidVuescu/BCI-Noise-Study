"""
Throwaway diagnostic: prove UnicornUDP is sending what we expect.

Run this with UnicornUDP.exe streaming to 127.0.0.1:1000.
It receives 250 packets (~1 second of data) and prints diagnostics.

delete.me
"""
import socket
import struct
import time

# Configuration (hardcoded here - this is throwaway)
UDP_HOST = "127.0.0.1"
UDP_PORT = 1000
N_CHANNELS = 17
BYTES_PER_SAMPLE = 68  # 17 channels * 4 bytes (float32)
N_PACKETS_TO_RECEIVE = 2500  # ~1 second at 250 Hz


# Channel layout per docs (0-indexed):
# 0-7:   EEG 1-8
# 8-10:  Accelerometer X, Y, Z
# 11-13: Gyroscope X, Y, Z
# 14:    Battery level
# 15:    Counter
# 16:    Validation indicator

def main():
    # Open UDP socket and bind to the port we expect data on
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_HOST, UDP_PORT))
    sock.settimeout(5.0)  # If no data in 5 seconds, something's wrong

    print(f"Listening on {UDP_HOST}:{UDP_PORT}")
    print(f"Expecting {N_PACKETS_TO_RECEIVE} packets of {BYTES_PER_SAMPLE} bytes each\n")

    samples = []
    start_time = time.perf_counter()

    for i in range(N_PACKETS_TO_RECEIVE):
        try:
            data, addr = sock.recvfrom(2048)  # Buffer larger than needed
        except socket.timeout:
            print(f"\nTIMEOUT after {i} packets. Is UnicornUDP actually streaming?")
            return

        # Sanity: packet size
        if len(data) != BYTES_PER_SAMPLE:
            print(f"  Packet {i}: WRONG SIZE - got {len(data)} bytes, expected {BYTES_PER_SAMPLE}")
            continue

        # Parse 17 little-endian float32 values
        # '<' = little-endian, '17f' = 17 floats
        values = struct.unpack('<17f', data)
        samples.append(values)

        # Print every 50th packet so output isn't a wall of text
        if i % 50 == 0:
            eeg = values[0:8]
            counter = values[15]
            validation = values[16]
            battery = values[14]
            print(f"Packet {i:3d}: counter={counter:8.0f} valid={validation:.0f} "
                  f"battery={battery:.1f}% EEG1={eeg[0]:+8.2f} EEG8={eeg[7]:+8.2f}")

    elapsed = time.perf_counter() - start_time
    sock.close()



    # Post-hoc diagnostics
    print(f"\n=== DIAGNOSTICS ===")
    print(f"Received {len(samples)} packets in {elapsed:.3f}s")
    print(f"Actual rate: {len(samples) / elapsed:.1f} packets/sec (expected ~250)")



    # Counter continuity check
    counters = [s[15] for s in samples]
    gaps = []
    for i in range(1, len(counters)):
        delta = counters[i] - counters[i - 1]
        if delta != 1:
            gaps.append((i, counters[i - 1], counters[i], delta))

    if gaps:
        print(f"\nCOUNTER GAPS DETECTED ({len(gaps)} total):")
        for idx, prev, curr, delta in gaps[:10]:
            print(f"  At packet {idx}: counter went {prev:.0f} -> {curr:.0f} (delta={delta:.0f})")
    else:
        print(f"\nCounter increments cleanly by 1 across all packets ✓")

    # EEG range and noise per channel
    import statistics
    eeg_per_channel = [[s[ch] for s in samples] for ch in range(8)]

    print(f"\nEEG noise characterization per channel ({len(samples)} samples, ~{len(samples) / 250:.1f}s):")
    print(f"  {'Ch':<5}{'DC offset':>12}{'AC stdev':>10}{'AC p2p':>10}  {'note'}")
    for ch, vals in enumerate(eeg_per_channel):
        mean = statistics.mean(vals)
        stdev = statistics.stdev(vals)
        centered = [v - mean for v in vals]
        p2p = max(centered) - min(centered)

        # Heuristic notes
        if stdev < 15:
            note = "clean"
        elif stdev < 30:
            note = "ok"
        elif stdev < 60:
            note = "noisy"
        else:
            note = "BAD - check electrode"

        print(f"  EEG{ch + 1:<2}{mean:+12.1f}{stdev:>8.2f}µV{p2p:>8.1f}µV  {note}")

    print(f"\n  Stdev = noise level estimate. Lower = better.")
    print(f"  Sane resting stdev: 5-20 µV. >30 suggests poor contact or movement.")



    # Validation indicator
    validations = [s[16] for s in samples]
    invalid_count = sum(1 for v in validations if v != 1)
    print(f"\nInvalid samples: {invalid_count}/{len(samples)}")


if __name__ == "__main__":
    main()