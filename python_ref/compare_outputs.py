"""
compare_outputs.py — Metrica de similitud y comparacion C++ vs Python.
"""
import json
import sys
import numpy as np


def similarity_score(times_a, times_b, tolerance_sec=0.04):
    """Porcentaje de beats que coinciden dentro de tolerance_sec (metrica simetrica)."""
    times_a = list(times_a); times_b = list(times_b)
    if not times_a or not times_b:
        return 0.0
    matched_a = sum(1 for ta in times_a if any(abs(ta - tb) <= tolerance_sec for tb in times_b))
    matched_b = sum(1 for tb in times_b if any(abs(tb - ta) <= tolerance_sec for ta in times_a))
    score_a = matched_a / len(times_a)
    score_b = matched_b / len(times_b)
    return 100.0 * (score_a + score_b) / 2.0


def compare(reference_json, cpp_output_json, tolerance_sec=0.04):
    with open(reference_json) as f:
        ref = json.load(f)
    with open(cpp_output_json) as f:
        cpp = json.load(f)

    ref_times = ref["beat_times"]
    cpp_times = cpp["beat_times"]

    overall = similarity_score(ref_times, cpp_times, tolerance_sec)

    print(f"Python beats: {len(ref_times)}")
    print(f"C++ beats:    {len(cpp_times)}")
    print(f"Python tempo: {ref['global_tempo']:.2f} BPM")
    print(f"C++ tempo:    {cpp['global_tempo']:.2f} BPM")
    print(f"Tempo error:  {abs(ref['global_tempo'] - cpp['global_tempo']):.4f} BPM")
    print(f"Overall:      {overall:.1f}%")
    print(f"PASS: {overall >= 93.0}")

    if "onset_envelope" in cpp and "onset_envelope" in ref:
        onset_ref = np.array(ref["onset_envelope"])
        onset_cpp = np.array(cpp["onset_envelope"])
        mae = np.mean(np.abs(onset_ref - onset_cpp))
        corr = np.corrcoef(onset_ref, onset_cpp)[0, 1]
        print(f"Onset env MAE: {mae:.6f}")
        print(f"Onset env correlation: {corr:.6f}")

    return overall >= 93.0


if __name__ == "__main__":
    ok = compare(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
