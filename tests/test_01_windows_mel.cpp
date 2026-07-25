// test_01_windows_mel.cpp — Fase 1: ventanas Hann, hanning5 y mel filterbank vs golden
#include <cstdio>
#include <string>
#include "golden.h"
#include "librosa_bpm/constants.h"
#include "librosa_bpm/math_util.h"
#include "librosa_bpm/mel_filterbank.h"

using namespace librosa_bpm;
using librosa_bpm::test::Golden;

static int failures = 0;
static void chk(const char* name, bool ok, const std::string& detail = "") {
    printf("  [%s] %-32s %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
    if (!ok) ++failures;
}

int main() {
    Golden g("tests/golden/bin");

    // --- hann_stft (C1) ---
    auto hann_stft = hann_periodic(N_FFT);
    auto golden_hann = g.read_double("hann_stft");
    chk("hann_stft maxdiff < 1e-12",
        test::max_abs_diff(hann_stft, golden_hann) < 1e-12,
        "maxdiff=" + std::to_string(test::max_abs_diff(hann_stft, golden_hann)));

    // --- hann_tg (C1) ---
    auto hann_tg = hann_periodic(WIN_LENGTH);
    auto golden_hann_tg = g.read_double("hann_tg");
    chk("hann_tg maxdiff < 1e-12",
        test::max_abs_diff(hann_tg, golden_hann_tg) < 1e-12,
        "maxdiff=" + std::to_string(test::max_abs_diff(hann_tg, golden_hann_tg)));

    // --- hanning5 (C3) ---
    auto h5 = hanning5();
    std::vector<double> h5v(h5.begin(), h5.end());
    std::vector<double> expected5 = {0.0, 0.5, 1.0, 0.5, 0.0};
    chk("hanning5 == [0,0.5,1,0.5,0]",
        test::max_abs_diff(h5v, expected5) == 0.0);

    // --- mel filterbank (C4) ---
    Matrix<double> H = build_mel_filterbank();
    auto golden_melfb = g.read_double("melfb");
    // aplanar H a vector<double> row-major
    std::vector<double> Hflat(H.storage().begin(), H.storage().end());
    chk("melfb maxdiff < 1e-8",
        test::max_abs_diff(Hflat, golden_melfb) < 1e-8,
        "maxdiff=" + std::to_string(test::max_abs_diff(Hflat, golden_melfb)));

    printf("\n=== test_01: %s (%d fallos) ===\n", failures == 0 ? "OK" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
