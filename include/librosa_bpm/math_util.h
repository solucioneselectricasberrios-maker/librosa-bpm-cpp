// math_util.h — Funciones matematicas del pipeline (formulas CORREGIDAS, ver SECCION8_ANALISIS_CRITICO.md)
#pragma once
#include "constants.h"
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>

namespace librosa_bpm {

// C1: Hann periodica = 0.5 - 0.5*cos(2*pi*n/N)  (denominador N, no N+1)
inline std::vector<double> hann_periodic(int N) {
    std::vector<double> w(N);
    for (int n = 0; n < N; ++n)
        w[n] = 0.5 - 0.5 * std::cos(2.0 * M_PI * n / double(N));
    return w;
}

// C3: np.hanning(5) = [0, 0.5, 1.0, 0.5, 0]
inline std::array<double, 5> hanning5() {
    return {0.0, 0.5, 1.0, 0.5, 0.0};
}

// Mel Slaney (verificado)
inline double hz_to_mel(double f) {
    if (f < 1000.0) return f / 66.66666666666667;
    return 15.0 + std::log(f / 1000.0) / 0.06875177742094912;
}
inline double mel_to_hz(double m) {
    if (m < 15.0) return m * 66.66666666666667;
    return 1000.0 * std::exp(0.06875177742094912 * (m - 15.0));
}

// Mediana de 128 valores (par): media de idx 63 y 64 tras ordenar.
// Cuidado: nth_element MUTA el array -> se recibe por valor.
inline double median128(std::vector<double> v) {
    if (v.size() != 128) throw std::runtime_error("median128: requiere 128 elementos");
    std::nth_element(v.begin(), v.begin() + 63, v.end());
    // segundo nth_element sobre el rango [64, end] para el idx 64
    std::nth_element(v.begin() + 64, v.begin() + 64, v.end());
    return 0.5 * (v[63] + v[64]);
}

// std con ddof=1 (corregido: numpy default es ddof=0, librosa usa ddof=1)
inline double std_ddof1(const std::vector<double>& v) {
    double mean = 0.0;
    for (double x : v) mean += x;
    mean /= double(v.size());
    double s = 0.0;
    for (double x : v) { double d = x - mean; s += d * d; }
    return std::sqrt(s / double(v.size() - 1));
}

inline double mean(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s / double(v.size());
}

// convolucion 'same' 1D con kernel simetrico
// (usado en local_score y trim; espejo de np.convolve(mode='same'))
inline std::vector<double> convolve_same(const std::vector<double>& signal,
                                         const std::vector<double>& kernel) {
    int N = int(signal.size());
    int K = int(kernel.size());
    std::vector<double> out(N, 0.0);
    int half = K / 2;
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int k = 0; k < K; ++k) {
            int idx = i + k - half;
            if (idx >= 0 && idx < N) s += kernel[k] * signal[idx];
        }
        out[i] = s;
    }
    return out;
}

// convolucion 'full' (usada en trim: np.convolve(boe, w5) sin mode)
inline std::vector<double> convolve_full(const std::vector<double>& signal,
                                         const std::vector<double>& kernel) {
    std::vector<double> out(signal.size() + kernel.size() - 1, 0.0);
    for (std::size_t i = 0; i < signal.size(); ++i) {
        for (std::size_t k = 0; k < kernel.size(); ++k) {
            out[i + k] += signal[i] * kernel[k];
        }
    }
    return out;
}

} // namespace librosa_bpm
