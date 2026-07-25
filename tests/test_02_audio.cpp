// test_02_audio.cpp — Fase 2: audio_loader (WAV + PCM) vs golden y
#include <cstdio>
#include <string>
#include "golden.h"
#include "librosa_bpm/audio_loader.h"
#include "librosa_bpm/constants.h"

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

    // PCM crudo (bit-exacto, bypassea decodificador)
    AudioData pcm = load_pcm_f32("tests/golden/song_f32.pcm", long long(golden_y.size()));
    chk("PCM n_samples == golden",
        pcm.n_frames == (long long)golden_y.size(),
        "pcm=" + std::to_string(pcm.n_frames) + " golden=" + std::to_string(golden_y.size()));
    chk("PCM MAE vs golden < 1e-7",
        test::mean_abs_diff(pcm.samples, golden_y) < 1e-7,
        "mae=" + std::to_string(test::mean_abs_diff(pcm.samples, golden_y)));

    // WAV (dr_wav; el WAV golden se genero con soundfile FLOAT)
    AudioData wav = load_wav("tests/golden/song_f32.wav");
    chk("WAV n_samples == golden",
        wav.n_frames == (long long)golden_y.size(),
        "wav=" + std::to_string(wav.n_frames) + " golden=" + std::to_string(golden_y.size()));
    chk("WAV MAE vs golden < 1e-7",
        test::mean_abs_diff(wav.samples, golden_y) < 1e-7,
        "mae=" + std::to_string(test::mean_abs_diff(wav.samples, golden_y)));

    printf("\n=== test_02: %s (%d fallos) ===\n", failures == 0 ? "OK" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
