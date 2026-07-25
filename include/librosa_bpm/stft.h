// stft.h — STFT con centering + RFFT + espectro de potencia.
// C2: M = 1 + N//HOP.  Reproduce librosa.stft(center=True, pad_mode='constant').
#pragma once
#include "librosa_bpm/constants.h"
#include "librosa_bpm/math_util.h"
#include "librosa_bpm/matrix.h"
#include "kiss_fftr.h"
#include <vector>
#include <cstring>

namespace librosa_bpm {

// Numero de frames M (corregido). N = muestras de audio.
inline int num_frames(long long N) {
    return 1 + int(N / HOP_LENGTH);
}

// STFT -> espectro de potencia (1025 x M), row-major.
// y: N muestras mono float64.
inline Matrix<double> stft_power(const std::vector<double>& y) {
    long long N = static_cast<long long>(y.size());
    int M = num_frames(N);
    int pad = N_FFT / 2;   // 1024 ceros a cada lado

    // y_padded = [0]*1024 + y + [0]*1024
    std::vector<double> ypad(N + 2 * pad, 0.0);
    std::memcpy(ypad.data() + pad, y.data(), N * sizeof(double));

    auto w = hann_periodic(N_FFT);
    Matrix<double> S(1 + N_FFT / 2, M, 0.0);

    kiss_fftr_cfg cfg = kiss_fftr_alloc(N_FFT, 0, nullptr, nullptr);
    std::vector<double> frame(N_FFT);

    for (int m = 0; m < M; ++m) {
        // frame[n] = ypad[m*HOP + n] * w[n]
        const double* src = ypad.data() + long long(m) * HOP_LENGTH;
        for (int n = 0; n < N_FFT; ++n) {
            frame[n] = src[n] * w[n];
        }
        // RFFT real de tamaño N_FFT -> N_FFT/2+1 bins
        kiss_fft_cpx out[N_FFT / 2 + 1];
        kiss_fftr(cfg, frame.data(), out);
        for (int k = 0; k <= N_FFT / 2; ++k) {
            double re = out[k].r, im = out[k].i;
            S(k, m) = re * re + im * im;   // |X|^2 (potencia)
        }
    }
    free(cfg);
    return S;
}

} // namespace librosa_bpm
