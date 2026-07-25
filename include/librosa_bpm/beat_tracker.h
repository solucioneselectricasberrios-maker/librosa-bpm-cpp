// beat_tracker.h — DP beat tracking (Ellis 2007). Reproduce librosa.beat._beat_tracker.
// Incluye todos los detalles omitidos por el doc: A2 (localmax asimetrico),
// A4 (rango DP exclusivo), C3 (hanning5), ddof=1, trim con localscore[beats].
#pragma once
#include "librosa_bpm/constants.h"
#include "librosa_bpm/math_util.h"
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace librosa_bpm {

// A2: maximo local asimetrico. x[0] NUNCA es maximo. x[-1] lo es si x[-1]>x[-2].
inline std::vector<char> localmax_asymmetric(const std::vector<double>& x) {
    int n = int(x.size());
    std::vector<char> m(n, 0);
    for (int i = 1; i < n - 1; ++i) {
        m[i] = (x[i] > x[i - 1]) && (x[i] >= x[i + 1]) ? 1 : 0;
    }
    if (n >= 2) m[n - 1] = (x[n - 1] > x[n - 2]) ? 1 : 0;
    return m;
}

struct BeatTrackerResult {
    std::vector<double> localscore;
    std::vector<double> cumscore;
    std::vector<int>    backlink;
    std::vector<char>   beats;        // bool array longitud M
    int frames_per_beat;
    int tail;
};

// oenv: onset envelope (M muestras). tempo: BPM estimado.
inline BeatTrackerResult beat_track(const std::vector<double>& oenv,
                                    double tempo,
                                    double tightness = TIGHTNESS,
                                    bool trim = TRIM) {
    int M = int(oenv.size());
    BeatTrackerResult r;
    r.localscore.assign(M, 0.0);
    r.cumscore.assign(M, 0.0);
    r.backlink.assign(M, -1);
    r.beats.assign(M, 0);
    r.tail = 0;

    // FPB = round(FRAME_RATE*60/tempo)
    r.frames_per_beat = int(std::llround(FRAME_RATE * 60.0 / tempo));

    // normalizar onsets: oenv / (std(ddof=1) + tiny)
    double sd = std_ddof1(oenv);
    std::vector<double> on(M);
    for (int t = 0; t < M; ++t) on[t] = oenv[t] / (sd + TINY64);

    // local score: gauss blur con kernel win[k]=exp(-0.5*(k*32/FPB)^2), k=-FPB..FPB
    int fpb = r.frames_per_beat;
    int K = 2 * fpb + 1;
    std::vector<double> win(K);
    for (int k = -fpb; k <= fpb; ++k) win[k + fpb] = std::exp(-0.5 * std::pow(double(k) * 32.0 / fpb, 2));
    r.localscore = convolve_same(on, win);

    // DP
    double score_thresh = 0.01 * (*std::max_element(r.localscore.begin(), r.localscore.end()));
    bool first_beat = true;
    r.backlink[0] = -1;
    r.cumscore[0] = r.localscore[0];
    for (int i = 1; i < M; ++i) {
        double best_score = NEG_INF;
        int beat_location = -1;
        int lo = i - int(std::llround(fpb / 2.0));
        int hi = i - 2 * fpb - 1;   // A4: extremo inferior EXCLUSIVO
        for (int loc = lo; loc > hi; --loc) {
            if (loc < 0) break;
            double score = r.cumscore[loc] - tightness * std::pow(std::log(double(i - loc)) - std::log(double(fpb)), 2);
            if (score > best_score) { best_score = score; beat_location = loc; }
        }
        r.cumscore[i] = (beat_location >= 0) ? r.localscore[i] + best_score : r.localscore[i];
        if (first_beat && r.localscore[i] < score_thresh) {
            r.backlink[i] = -1;
        } else {
            r.backlink[i] = beat_location;
            first_beat = false;
        }
    }

    // tail: mediana de cumscore donde es max local; thr=0.5*mediana
    std::vector<char> lmax = localmax_asymmetric(r.cumscore);
    std::vector<double> peaks;
    for (int t = 0; t < M; ++t) if (lmax[t]) peaks.push_back(r.cumscore[t]);
    // mediana de peaks (librosa masked median). n par -> media de los 2 centrales.
    double med;
    if (peaks.empty()) {
        med = 0.0;
    } else {
        size_t n = peaks.size();
        std::nth_element(peaks.begin(), peaks.begin() + n / 2, peaks.end());
        if (n % 2 == 1) med = peaks[n / 2];
        else {
            double a = peaks[n / 2];
            std::nth_element(peaks.begin(), peaks.begin() + n / 2 - 1, peaks.end());
            med = 0.5 * (a + peaks[n / 2 - 1]);
        }
    }
    double thr = 0.5 * med;
    r.tail = M - 1;
    while (r.tail >= 0) {
        if (lmax[r.tail] && r.cumscore[r.tail] >= thr) break;
        r.tail--;
    }

    // backtracking: seguir backlinks hasta -1 (terminador). backlink[0] = -1.
    {
        int t = r.tail;
        while (t >= 0) {
            r.beats[t] = 1;
            int next = r.backlink[t];
            if (next < 0) break;   // terminador
            t = next;
        }
    }

    // trim: smooth_boe = convolve(localscore[beats], hanning5) recortado;
    // threshold = 0.5 * sqrt(mean(smooth_boe^2)). La poda usa localscore[n] por frame.
    auto w5 = hanning5();
    std::vector<double> w5v(w5.begin(), w5.end());
    std::vector<double> boe;
    for (int t = 0; t < M; ++t) if (r.beats[t]) boe.push_back(r.localscore[t]);
    std::vector<double> smooth_full = convolve_full(boe, w5v);
    // slicing [len(w)//2 : len(localscore)+len(w)//2] (recorta solo; long real = n_beats+4)
    int half_w = int(w5v.size()) / 2;   // 2
    int smooth_end = std::min(int(smooth_full.size()), M + half_w);
    double s2 = 0.0;
    int smooth_count = 0;
    for (int i = half_w; i < smooth_end; ++i) { s2 += smooth_full[i] * smooth_full[i]; smooth_count++; }
    double thrT = trim ? 0.5 * std::sqrt(s2 / double(smooth_count)) : 0.0;

    int n = 0;
    while (n < M && r.localscore[n] <= thrT) { r.beats[n] = 0; n++; }
    n = M - 1;
    while (n >= 0 && r.localscore[n] <= thrT) { r.beats[n] = 0; n--; }

    return r;
}

} // namespace librosa_bpm
