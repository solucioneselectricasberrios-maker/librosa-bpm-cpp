// test_04_onset.cpp — Fase 4: onset envelope vs golden (METRICA BISAGRA)
#include <cstdio>
#include <string>
#include "golden.h"
#include "librosa_bpm/audio_loader.h"
#include "librosa_bpm/stft.h"
#include "librosa_bpm/mel_filterbank.h"
#include "librosa_bpm/power_to_db.h"
#include "librosa_bpm/onset_detector.h"

using namespace librosa_bpm;
using librosa_bpm::test::Golden;

static int failures = 0;
static void chk(const char* name, bool ok, const std::string& detail = "") {
    printf("  [%s] %-32s %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
    if (!ok) ++failures;
}

int main() {
    Golden g("tests/golden/bin");
    auto golden_y = g.read_double("y");
    AudioData audio = load_pcm_f32("tests/golden/song_f32.pcm", long long(golden_y.size()));

    Matrix<double> S_power = stft_power(audio.samples);
    Matrix<double> melfb = build_mel_filterbank();
    Matrix<double> S_mel = matmul(melfb, S_power);
    Matrix<double> S_dB = power_to_db(S_mel);
    std::vector<double> oenv = onset_strength(S_dB);

    auto golden_oenv = g.read_double("oenv");
    chk("oenv size == golden", oenv.size() == golden_oenv.size(),
        "got " + std::to_string(oenv.size()) + " expected " + std::to_string(golden_oenv.size()));
    chk("oenv mean close to golden",
        std::fabs(mean(oenv) - mean(golden_oenv)) < 1e-3,
        "mean=" + std::to_string(mean(oenv)) + " golden_mean=" + std::to_string(mean(golden_oenv)));
    double c = test::correlation(oenv, golden_oenv);
    chk("oenv corr > 0.9999", c > 0.9999, "corr=" + std::to_string(c));
    chk("oenv MAE < 1e-4",
        test::mean_abs_diff(oenv, golden_oenv) < 1e-4,
        "mae=" + std::to_string(test::mean_abs_diff(oenv, golden_oenv)));

    printf("\n=== test_04: %s (%d fallos) ===\n", failures == 0 ? "OK" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
