// test_03_stft_mel.cpp — Fase 3: STFT potencia + Mel + dB vs golden
#include <cstdio>
#include <string>
#include "golden.h"
#include "librosa_bpm/audio_loader.h"
#include "librosa_bpm/stft.h"
#include "librosa_bpm/mel_filterbank.h"
#include "librosa_bpm/power_to_db.h"
#include "librosa_bpm/matrix.h"

using namespace librosa_bpm;
using librosa_bpm::test::Golden;

static int failures = 0;
static void chk(const char* name, bool ok, const std::string& detail = "") {
    printf("  [%s] %-32s %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
    if (!ok) ++failures;
}

// Submuestrea una matriz para comparar solo el bloque [0:r, 0:c] (rapido para debug)
static std::vector<double> top_left(const Matrix<double>& M, int r, int c) {
    std::vector<double> out;
    out.reserve(r * c);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            out.push_back(M(i, j));
    return out;
}
static std::vector<double> top_left_vec(const std::vector<double>& flat, int rows, int cols, int r, int c) {
    std::vector<double> out;
    out.reserve(r * c);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            out.push_back(flat[i * cols + j]);
    return out;
}

int main() {
    Golden g("tests/golden/bin");
    auto golden_y = g.read_double("y");

    AudioData audio = load_pcm_f32("tests/golden/song_f32.pcm", long long(golden_y.size()));

    int g_rows = 1025;
    int g_cols = (int)golden_y.size() / 512 + 1;  // = 1 + N/hop, igual que librosa

    // --- STFT potencia ---
    Matrix<double> S_power = stft_power(audio.samples);
    auto gp = g.read_double("S_power");
    {
        auto mine = top_left(S_power, 10, 10);
        auto gold = top_left_vec(gp, g_rows, g_cols, 10, 10);
        chk("S_power shape",
            S_power.rows() == g_rows && S_power.cols() == g_cols,
            "got " + std::to_string(S_power.rows()) + "x" + std::to_string(S_power.cols()) +
            " expected " + std::to_string(g_rows) + "x" + std::to_string(g_cols));
        chk("S_power[0:10,0:10] MAE < 1e-6",
            test::mean_abs_diff(mine, gold) < 1e-6,
            "mae=" + std::to_string(test::mean_abs_diff(mine, gold)));
    }

    // --- Mel spectrogram = melfb @ S_power ---
    Matrix<double> melfb = build_mel_filterbank();
    Matrix<double> S_mel = matmul(melfb, S_power);
    auto gm = g.read_double("S_mel");
    {
        auto mine = top_left(S_mel, 10, 10);
        auto gold = top_left_vec(gm, 128, g_cols, 10, 10);
        chk("S_mel[0:10,0:10] MAE < 1e-5",
            test::mean_abs_diff(mine, gold) < 1e-5,
            "mae=" + std::to_string(test::mean_abs_diff(mine, gold)));
    }

    // --- dB ---
    Matrix<double> S_dB = power_to_db(S_mel);
    auto gd = g.read_double("S_dB");
    {
        auto mine = top_left(S_dB, 10, 10);
        auto gold = top_left_vec(gd, 128, g_cols, 10, 10);
        double m = test::max_abs_diff(mine, gold);
        chk("S_dB[0:10,0:10] maxdiff < 0.01 dB",
            m < 0.01, "maxdiff=" + std::to_string(m));
    }

    printf("\n=== test_03: %s (%d fallos) ===\n", failures == 0 ? "OK" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
