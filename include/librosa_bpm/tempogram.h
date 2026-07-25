// tempogram.h — Tempogram (autocorrelacion enventanada) + estimation de tempo.
// Reproduce librosa.feature.tempogram + librosa.feature.tempo.
//
// 1. pad oenv con linear_ramp de WIN_HALF=172 a cada lado
// 2. frame hop=1, len=WIN_LENGTH=344; x[t] = opad[t:t+344] * hann344
// 3. tg[:,t] = AC_bounded(x)[:344]   (n_pad=next_fast_len(2*344-1)=720)
// 4. norm L-inf por COLUMNA
// 5. tg_avg[tau] = mean_t tg[tau,t]
// 6. prior lognormal + score = log1p(1e6*tg_avg) + logprior ; argmax -> tempo
#pragma once
#include "librosa_bpm/constants.h"
#include "librosa_bpm/math_util.h"
#include "librosa_bpm/matrix.h"
#include "kiss_fft.h"
#include "kiss_fftr.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace librosa_bpm {

// Autocorrelacion acotada por FFT: ac[l] = irfft(|rfft(x, n_pad)|^2)[:max_size]
inline std::vector<double> autocorrelate_bounded(const std::vector<double>& x, int max_size) {
    int n = int(x.size());
    // next_fast_len(2*n-1, real): menor numero >= 2n-1 con factores primos 2,3,5,7
    // Para 2*344-1=687 -> 720. Implementamos una busqueda simple.
    auto is_smooth = [](int v) {
        for (int p : {2, 3, 5, 7}) while (v % p == 0) v /= p;
        return v == 1;
    };
    int n_pad = 2 * n - 1;
    while (!is_smooth(n_pad)) ++n_pad;

    std::vector<double> xp(n_pad, 0.0);
    for (int i = 0; i < n; ++i) xp[i] = x[i];

    kiss_fftr_cfg cfg = kiss_fftr_alloc(n_pad, 0, nullptr, nullptr);
    std::vector<kiss_fft_cpx> spec(n_pad / 2 + 1);
    kiss_fftr(cfg, xp.data(), spec.data());
    // potencia espectral = |spec|^2 (real). Se coloca en la parte real de los bins
    // y parte imaginaria 0; la IFFT Hermitiana da la autocorrelacion.
    std::vector<kiss_fft_cpx> powspec(n_pad / 2 + 1);
    for (int k = 0; k < n_pad / 2 + 1; ++k) {
        powspec[k].r = spec[k].r * spec[k].r + spec[k].i * spec[k].i;
        powspec[k].i = 0.0;
    }
    free(cfg);

    kiss_fftr_cfg cfgi = kiss_fftr_alloc(n_pad, 1, nullptr, nullptr);
    std::vector<double> ac(n_pad, 0.0);
    kiss_fftri(cfgi, powspec.data(), ac.data());
    double scale = 1.0 / double(n_pad);   // kissfft no escala en inversa
    for (int i = 0; i < max_size && i < n_pad; ++i) ac[i] *= scale;
    free(cfgi);

    ac.resize(max_size);
    return ac;
}

// Padding linear_ramp de 'pad' muestras a cada lado.
// head: rampa 0 -> oenv[0]  (librosa: linspace(0, oenv[0], pad+2)[1:-1])
// tail: rampa oenv[-1] -> 0
inline std::vector<double> linear_ramp_pad(const std::vector<double>& oenv, int pad) {
    int n = int(oenv.size());
    std::vector<double> out;
    out.reserve(n + 2 * pad);
    if (pad > 0) {
        // head: valores oenv[0]*(i+1)/(pad+1)  para i=0..pad-1
        for (int i = 0; i < pad; ++i) out.push_back(oenv[0] * double(i + 1) / double(pad + 1));
    }
    out.insert(out.end(), oenv.begin(), oenv.end());
    if (pad > 0) {
        for (int i = 0; i < pad; ++i) out.push_back(oenv[n - 1] * double(pad - i) / double(pad + 1));
    }
    return out;
}

struct TempoResult {
    double tempo;
    std::vector<double> tg_avg;   // (WIN_LENGTH,)
    std::vector<double> bpms;     // (WIN_LENGTH,)
    Matrix<double> tg;            // (WIN_LENGTH, M)
};

inline TempoResult tempo_estimate(const std::vector<double>& oenv) {
    int n = int(oenv.size());
    std::vector<double> opad = linear_ramp_pad(oenv, WIN_HALF);
    auto w = hann_periodic(WIN_LENGTH);

    Matrix<double> tg(WIN_LENGTH, n, 0.0);
    std::vector<double> seg(WIN_LENGTH);
    for (int t = 0; t < n; ++t) {
        for (int tau = 0; tau < WIN_LENGTH; ++tau) {
            seg[tau] = opad[t + tau] * w[tau];
        }
        auto ac = autocorrelate_bounded(seg, WIN_LENGTH);
        for (int tau = 0; tau < WIN_LENGTH; ++tau) tg(tau, t) = ac[tau];
    }

    // norm L-inf por COLUMNA (axis=-2): dividir cada columna por su max-abs
    for (int t = 0; t < n; ++t) {
        double mx = 0.0;
        for (int tau = 0; tau < WIN_LENGTH; ++tau) {
            double a = std::fabs(tg(tau, t));
            if (a > mx) mx = a;
        }
        if (mx > 0.0) {
            for (int tau = 0; tau < WIN_LENGTH; ++tau) tg(tau, t) /= mx;
        }
    }

    // tg_avg = mean sobre t
    std::vector<double> tg_avg(WIN_LENGTH, 0.0);
    for (int tau = 0; tau < WIN_LENGTH; ++tau) {
        double s = 0.0;
        for (int t = 0; t < n; ++t) s += tg(tau, t);
        tg_avg[tau] = s / double(n);
    }

    // eje BPM: bpms[0]=inf, bpms[L]=60*SR/(HOP*L)
    std::vector<double> bpms(WIN_LENGTH, 0.0);
    bpms[0] = std::numeric_limits<double>::infinity();
    for (int L = 1; L < WIN_LENGTH; ++L)
        bpms[L] = 60.0 * SR_ANALYSIS / (HOP_LENGTH * double(L));

    // prior lognormal + score
    std::vector<double> logprior(WIN_LENGTH);
    double log2_start = std::log2(START_BPM);
    for (int L = 0; L < WIN_LENGTH; ++L) {
        logprior[L] = -0.5 * std::pow((std::log2(bpms[L]) - log2_start) / STD_BPM, 2);
    }
    // A3: max_idx = argmax(bpms < MAX_TEMPO); logprior[:max_idx] = -inf
    int max_idx = 0;
    for (int L = 0; L < WIN_LENGTH; ++L) {
        if (bpms[L] < MAX_TEMPO) { max_idx = L; break; }
    }
    for (int L = 0; L < max_idx; ++L) logprior[L] = NEG_INF;

    // score = log1p(1e6 * tg_avg) + logprior ; argmax
    double best_score = NEG_INF;
    int best = 0;
    for (int L = 0; L < WIN_LENGTH; ++L) {
        double score = std::log1p(1e6 * tg_avg[L]) + logprior[L];
        if (score > best_score) { best_score = score; best = L; }
    }

    TempoResult r;
    r.tempo = bpms[best];
    r.tg_avg = std::move(tg_avg);
    r.bpms = std::move(bpms);
    r.tg = std::move(tg);
    return r;
}

} // namespace librosa_bpm
