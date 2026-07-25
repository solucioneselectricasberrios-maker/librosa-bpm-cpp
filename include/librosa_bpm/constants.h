// constants.h — Constantes del pipeline (defaults de librosa 0.11.0, verificado)
#pragma once
#include <limits>
#include <cstddef>
#include <cmath>

// MSVC no define M_PI por defecto
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace librosa_bpm {

// Pipeline de analisis
constexpr int    SR_ANALYSIS  = 22050;     // sample rate de analisis
constexpr int    HOP_LENGTH   = 512;       // hop length STFT
constexpr int    N_FFT        = 2048;      // tamaño FFT
constexpr int    N_MELS       = 128;       // bandas mel
constexpr double FMAX         = 11025.0;   // = SR/2 (kwargs.setdefault fmax)
constexpr double FMIN         = 0.0;

// power_to_db
constexpr double AMIN         = 1e-10;
constexpr double TOP_DB       = 80.0;      // A1: top_db ACTIVO (no documentado en doc original)

// tempo / prior
constexpr double START_BPM    = 120.0;
constexpr double STD_BPM      = 1.0;
constexpr double AC_SIZE      = 8.0;
constexpr double MAX_TEMPO    = 320.0;
constexpr int    WIN_LENGTH   = 344;       // floor(8*22050/512)

// beat tracker DP
constexpr double TIGHTNESS    = 100.0;
constexpr bool   TRIM         = true;

// derivados
constexpr double FRAME_RATE   = double(SR_ANALYSIS) / HOP_LENGTH;   // 43.0664
constexpr int    PAD_ONSET    = 1 + N_FFT / (2 * HOP_LENGTH);       // 3 (lag=1 + 2)
constexpr int    WIN_HALF     = WIN_LENGTH / 2;                     // 172

// epsilon (tiny de float64)
constexpr double TINY64       = std::numeric_limits<double>::min(); // 2.225e-308
constexpr double NEG_INF      = -std::numeric_limits<double>::infinity();

// conversion frame -> segundo
constexpr double FRAME_TO_SEC = double(HOP_LENGTH) / SR_ANALYSIS;   // 0.0232199546...

} // namespace librosa_bpm
