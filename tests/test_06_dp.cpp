// test_06_dp.cpp — Fase 6: DP beat tracker vs golden (localscore, backlink, tail, beats)
#include <cstdio>
#include <string>
#include "golden.h"
#include "librosa_bpm/audio_loader.h"
#include "librosa_bpm/stft.h"
#include "librosa_bpm/mel_filterbank.h"
#include "librosa_bpm/power_to_db.h"
#include "librosa_bpm/onset_detector.h"
#include "librosa_bpm/tempogram.h"
#include "librosa_bpm/beat_tracker.h"

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
    Matrix<double> S_dB = power_to_db(matmul(build_mel_filterbank(), S_power));
    std::vector<double> oenv = onset_strength(S_dB);
    TempoResult tr = tempo_estimate(oenv);
    BeatTrackerResult bt = beat_track(oenv, tr.tempo);

    // localscore
    auto golden_ls = g.read_double("localscore");
    double cl = test::correlation(bt.localscore, golden_ls);
    chk("localscore corr > 0.999", cl > 0.999, "corr=" + std::to_string(cl));

    // cumscore
    auto golden_cs = g.read_double("cumscore");
    double cc = test::correlation(bt.cumscore, golden_cs);
    chk("cumscore corr > 0.999", cc > 0.999, "corr=" + std::to_string(cc));

    // FPB
    int golden_fpb = int(g.read_double("frames_per_beat")[0]);
    chk("frames_per_beat == golden",
        bt.frames_per_beat == golden_fpb,
        "cpp=" + std::to_string(bt.frames_per_beat) + " golden=" + std::to_string(golden_fpb));

    // backlink: % identico frame a frame
    auto golden_bl = g.read_double("backlink");
    int same = 0;
    for (size_t i = 0; i < golden_bl.size(); ++i)
        if (bt.backlink[i] == int(golden_bl[i])) ++same;
    double pct = 100.0 * double(same) / double(golden_bl.size());
    chk("backlink 99% identico", pct > 99.0, std::to_string(pct) + "%");

    // tail
    int golden_tail = int(g.read_double("tail")[0]);
    chk("tail == golden", bt.tail == golden_tail,
        "cpp=" + std::to_string(bt.tail) + " golden=" + std::to_string(golden_tail));

    // beats (array booleano identico)
    auto golden_beats = g.read_double("beats_bool");
    bool beats_equal = (golden_beats.size() == bt.beats.size());
    int my_count = 0, gold_count = 0;
    if (beats_equal) {
        for (size_t i = 0; i < golden_beats.size(); ++i) {
            if (bt.beats[i]) my_count++;
            if (golden_beats[i] > 0.5) gold_count++;
            if ((bt.beats[i] ? 1.0 : 0.0) != (golden_beats[i] > 0.5 ? 1.0 : 0.0)) beats_equal = false;
        }
    }
    chk("beats bool array identico", beats_equal,
        "cpp=" + std::to_string(my_count) + " golden=" + std::to_string(gold_count));

    printf("\n=== test_06: %s (%d fallos) ===\n", failures == 0 ? "OK" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
