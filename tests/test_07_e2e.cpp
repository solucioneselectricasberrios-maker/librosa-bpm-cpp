// test_07_e2e.cpp — End-to-end: BPMDetector sobre el WAV golden, similitud vs Python >= 93%
#include <cstdio>
#include <string>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include "golden.h"
#include "librosa_bpm/bpm_detector.h"

using namespace librosa_bpm;
using librosa_bpm::test::Golden;

static int failures = 0;
static void chk(const char* name, bool ok, const std::string& detail = "") {
    printf("  [%s] %-32s %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
    if (!ok) ++failures;
}

// Metrica de similitud (espejo de compare_outputs.similarity_score)
static double similarity(const std::vector<double>& a, const std::vector<double>& b, double tol) {
    if (a.empty() || b.empty()) return 0.0;
    int ma = 0, mb = 0;
    for (double ta : a) for (double tb : b) if (std::fabs(ta - tb) <= tol) { ++ma; break; }
    for (double tb : b) for (double ta : a) if (std::fabs(tb - ta) <= tol) { ++mb; break; }
    return 100.0 * (double(ma) / a.size() + double(mb) / b.size()) / 2.0;
}

int main() {
    Golden g("tests/golden/bin");
    double golden_tempo = g.read_double("tempo")[0];
    auto golden_beats = g.read_double("beat_times");

    // Ejecutar el detector sobre el WAV golden (bit-exacto vs soundfile)
    BPMDetector detector;
    BPMResult res = detector.analyze_wav("tests/golden/song_f32.wav");

    chk("tempo == golden", std::fabs(res.global_tempo - golden_tempo) < 0.01,
        "cpp=" + std::to_string(res.global_tempo) + " golden=" + std::to_string(golden_tempo));
    chk("n_beats == golden", res.n_beats == int(golden_beats.size()),
        "cpp=" + std::to_string(res.n_beats) + " golden=" + std::to_string(golden_beats.size()));

    double sim = similarity(res.beat_times, golden_beats, 0.04);
    chk("similarity >= 93% (tol 0.04s)", sim >= 93.0, "similarity=" + std::to_string(sim) + "%");

    double sim_strict = similarity(res.beat_times, golden_beats, 0.02);
    printf("    [info] similarity estricta (tol 0.02s) = %.2f%%\n", sim_strict);

    printf("\n=== test_07: %s (%d fallos) ===\n", failures == 0 ? "OK" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
