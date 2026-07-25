// power_to_db.h — Conversion a dB con top_db=80 (A1).
// log_spec = 10*log10(max(amin,S)) - 10*log10(max(amin,ref)); clamp peak - top_db.
#pragma once
#include "librosa_bpm/constants.h"
#include "librosa_bpm/matrix.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace librosa_bpm {

inline Matrix<double> power_to_db(const Matrix<double>& S,
                                  double amin = AMIN,
                                  double top_db = TOP_DB,
                                  double ref = 1.0) {
    Matrix<double> out(S.rows(), S.cols(), 0.0);
    double log_ref = 10.0 * std::log10(std::max(amin, std::fabs(ref)));

    double peak = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < S.size(); ++i) {
        double v = 10.0 * std::log10(std::max(amin, S[i])) - log_ref;
        out[i] = v;
        if (v > peak) peak = v;
    }
    // clamp top_db
    if (top_db > 0) {
        double floor_val = peak - top_db;
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (out[i] < floor_val) out[i] = floor_val;
        }
    }
    return out;
}

} // namespace librosa_bpm
