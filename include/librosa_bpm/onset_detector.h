// onset_detector.h — Onset strength envelope (median, lag=1, pad 3).
// Reproduce librosa.onset.onset_strength(aggregate=np.median).
//
//   flux[i,m]   = max(0, S_dB[i,m] - S_dB[i,m-1])       para m=1..M-1
//   oenv_raw[m-1] = median_128(flux[:,m])
//   oenv = [0,0,0] + oenv_raw ; oenv = oenv[:M]
#pragma once
#include "librosa_bpm/constants.h"
#include "librosa_bpm/matrix.h"
#include "librosa_bpm/math_util.h"
#include <vector>

namespace librosa_bpm {

inline std::vector<double> onset_strength(const Matrix<double>& S_dB) {
    int M = int(S_dB.cols());
    std::vector<double> oenv(M, 0.0);

    // oenv_raw tiene M-1 valores (flux se calcula para m=1..M-1)
    std::vector<double> oenv_raw(M - 1, 0.0);
    std::vector<double> col(N_MELS);
    for (int m = 1; m < M; ++m) {
        for (int i = 0; i < N_MELS; ++i) {
            double diff = S_dB(i, m) - S_dB(i, m - 1);
            col[i] = diff > 0.0 ? diff : 0.0;     // half-wave rectify
        }
        oenv_raw[m - 1] = median128(col);          // copia por valor (nth_element muta)
    }

    // padding: 3 ceros (PAD_ONSET = lag + n_fft/(2*hop) = 1 + 2) al inicio
    // oenv_final[t] = oenv_raw[t - PAD_ONSET] si t-PAD_ONSET en [0, M-1), sino 0
    for (int t = 0; t < M; ++t) {
        int src = t - PAD_ONSET;
        if (src >= 0 && src < (M - 1)) {
            oenv[t] = oenv_raw[src];
        }   // else queda 0 (padding de ceros)
    }
    return oenv;
}

} // namespace librosa_bpm
