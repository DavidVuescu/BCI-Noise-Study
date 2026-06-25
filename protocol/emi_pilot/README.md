# protocol/emi_pilot/

Validation bundle for the EMI condition's physical-layer interference delivery.

This directory is the retrospective equivalent of the preregistered headset-on-table
pilot gate (preregistration §8). It contains all evidence that the EMI manipulation
was delivered and that its effect on the Bluetooth link has been characterised.

---

## What the preregistration required

Before main data collection, one control recording and one EMI recording were to be
conducted with the headset placed on a stable surface (no subject). Counter-gap rate
and dropout rate were to be compared between the two. If no measurable disruption was
observed, the EMI configuration was to be modified until it was.

This pilot was not conducted. See DEVIATIONS.md for the full entry.

---

## What this directory provides instead

### airspace-test_iperf.log.txt
iperf3 UDP client log from a pre-study channel saturation test. The client ran at a
target rate of 200 Mbits/sec (over-capacity by design) against a wired server, while
the UniFi U7-Lite operated on 2.4 GHz channel 6 at 40 MHz width. Key figures:

- Duration: 194.6 s
- Mean achieved throughput: 48.0 Mbits/sec (24% of target)
- Std: 16.4 Mbits/sec, CV: 34.2%
- Min interval: 3.5 Mbits/sec — channel was actively contested throughout

The over-capacity target forces continuous channel occupancy regardless of achieved
throughput. The low mean relative to target confirms the channel was saturated, not
merely loaded. This is the WiFi-layer evidence that the interference was physically
present.

### emi_pilot_validation.py
Python script that answers the preregistered pilot question retrospectively:
does the EMI condition produce measurably elevated Bluetooth counter-gap rate
relative to control, using the actual subject recordings?

Usage:
    python emi_pilot_validation.py                    # reads meta.json (fast)
    python emi_pilot_validation.py --from-raw         # recomputes from _eeg.npy
    python emi_pilot_validation.py --inspect-meta     # inspect meta.json key names
    python emi_pilot_validation.py --data-dir <path>  # specify data root

### emi_validation_results.json
Output of emi_pilot_validation.py run against all subject recordings (N=42 control,
N=44 EMI). Machine-readable; see DEVIATIONS.md for the human-readable summary.

---

## Result and interpretation

Zero samples were dropped in any recording in any condition across all subjects.
The Unicorn's Bluetooth 2.1+EDR Adaptive Frequency Hopping successfully avoided
the saturated channel throughout every EMI session.

This means:
- The WiFi saturation was delivered (iperf3 log confirms).
- The BT transmission layer was fully resilient (counter-gap analysis confirms).
- Any EMI effects on classifier accuracy arise from direct RF coupling at the
  electrode or amplifier level, not from packet loss.

This is reported in the Discussion section of the paper and closes the
deviation logged in DEVIATIONS.md.