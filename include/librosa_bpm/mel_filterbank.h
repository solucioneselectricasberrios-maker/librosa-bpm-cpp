// mel_filterbank.h — Filtros Mel triangulares (Slaney).
// C4: formula del codigo REAL de librosa, no la del doc (que daba suma 0).
#pragma once
#include "matrix.h"
#include "constants.h"
#include "math_util.h"

namespace librosa_bpm {

inline Matrix<double> build_mel_filterbank() {
    // 1025 frecuencias FFT
    std::vector<double> fftfreqs(1 + N_FFT / 2);
    for (int k = 0; k <= N_FFT / 2; ++k)
        fftfreqs[k] = double(k) * double(SR_ANALYSIS) / double(N_FFT);

    // 130 frecuencias centrales equiespaciadas en Mel
    double mel_min = hz_to_mel(FMIN);
    double mel_max = hz_to_mel(FMAX);
    std::vector<double> mel_f(N_MELS + 2);
    for (int i = 0; i < N_MELS + 2; ++i) {
        double frac = double(i) / double(N_MELS + 1);   // linspace(mel_min, mel_max, 130)
        mel_f[i] = mel_to_hz(mel_min + frac * (mel_max - mel_min));
    }

    Matrix<double> H(N_MELS, 1 + N_FFT / 2, 0.0);
    for (int i = 0; i < N_MELS; ++i) {
        double fd0 = mel_f[i + 1] - mel_f[i];           // fdiff[i]
        double fd1 = mel_f[i + 2] - mel_f[i + 1];       // fdiff[i+1]
        double enorm = 2.0 / (mel_f[i + 2] - mel_f[i]); // normalizacion Slaney
        for (int k = 0; k <= N_FFT / 2; ++k) {
            double lower = (fftfreqs[k] - mel_f[i]) / fd0;     // asciende 0->1
            double upper = (mel_f[i + 2] - fftfreqs[k]) / fd1; // desciende 1->0
            double v = std::max(0.0, std::min(lower, upper));
            H(i, k) = v * enorm;
        }
    }
    return H;
}

} // namespace librosa_bpm
