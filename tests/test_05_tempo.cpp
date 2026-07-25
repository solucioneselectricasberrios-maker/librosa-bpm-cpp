// test_05_tempo.cpp — Fase 5: tempogram + tempo vs golden
#include <cstdio>
#include <string>
#include "golden.h"
#include "librosa_bpm/audio_loader.h"
#include "librosa_bpm/stft.h"
#include "librosa_bpm/mel_filterbank.h"
#include "librosa_bpm/power_to_db.h"
#include "librosa_bpm/onset_detector.h"
#include "librosa_bpm/tempogram.h"

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
    Matrix<double> S_dB = power_to_db(matmul(melfb, S_power));
    std::vector<double> oenv = onset_strength(S_dB);
    TempoResult tr = tempo_estimate(oenv);

    // tg_avg
    auto golden_tg_avg = g.read_double("tg_avg");
    double c = test::correlation(tr.tg_avg, golden_tg_avg);
    chk("tg_avg corr > 0.999", c > 0.999, "corr=" + std::to_string(c));

    // tempo
    double golden_tempo = g.read_double("tempo")[0];
    double dt = std::fabs(tr.tempo - golden_tempo);
    chk("tempo +-0.01 BPM", dt < 0.01,
        "cpp=" + std::to_string(tr.tempo) + " golden=" + std::to_string(golden_tempo)
        + " dt=" + std::to_string(dt));

    printf("\n=== test_05: %s (%d fallos) ===\n", failures == 0 ? "OK" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
