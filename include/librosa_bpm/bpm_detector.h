// bpm_detector.h — API publica que orquesta todo el pipeline (etapas 1-4).
// Reproduce el contrato de tempo_analysis.py (NOISE): 3 llamadas equivalentes a librosa.
#pragma once
#include "librosa_bpm/constants.h"
#include "librosa_bpm/audio_loader.h"
#include "librosa_bpm/stft.h"
#include "librosa_bpm/mel_filterbank.h"
#include "librosa_bpm/power_to_db.h"
#include "librosa_bpm/onset_detector.h"
#include "librosa_bpm/tempogram.h"
#include "librosa_bpm/beat_tracker.h"
#include "librosa_bpm/matrix.h"
#include <vector>
#include <string>
#include <stdexcept>

namespace librosa_bpm {

struct BPMResult {
    double global_tempo = 0.0;
    std::vector<double> beat_times;
    std::vector<double> local_bpms;
    int n_beats = 0;
    double duration_seconds = 0.0;
};

struct BPMConfig {
    int sr_analysis = SR_ANALYSIS;
    int hop_length  = HOP_LENGTH;
    int n_fft       = N_FFT;
    int n_mels      = N_MELS;
    double start_bpm = START_BPM;
    double tightness = TIGHTNESS;
    bool trim = TRIM;
};

class BPMDetector {
public:
    explicit BPMDetector(const BPMConfig& config = BPMConfig{}) : cfg_(config) {}

    // Analiza un buffer de audio (mono, ya remuestreado a sr_analysis).
    BPMResult analyze_buffer(const std::vector<double>& y) {
        BPMResult res;
        res.duration_seconds = double(y.size()) / double(cfg_.sr_analysis);

        // Etapa 1: onset envelope
        Matrix<double> S_power = stft_power(y);
        Matrix<double> melfb = build_mel_filterbank();
        Matrix<double> S_dB = power_to_db(matmul(melfb, S_power));
        std::vector<double> oenv = onset_strength(S_dB);

        // Caso degenerado: sin onsets -> 0 BPM (como librosa)
        bool any = false;
        for (double v : oenv) if (v != 0.0) { any = true; break; }
        if (!any) {
            res.global_tempo = 0.0;
            return res;
        }

        // Etapa 2: tempo
        TempoResult tr = tempo_estimate(oenv);
        res.global_tempo = tr.tempo;
        if (res.global_tempo <= 0.0) res.global_tempo = 120.0;   // salvaguarda NOISE

        // Etapa 3: DP beat tracker
        BeatTrackerResult bt = beat_track(oenv, res.global_tempo, cfg_.tightness, cfg_.trim);

        // Etapa 4: frames -> tiempos
        for (int t = 0; t < int(bt.beats.size()); ++t) {
            if (bt.beats[t]) res.beat_times.push_back(double(t) * FRAME_TO_SEC);
        }
        res.n_beats = int(res.beat_times.size());

        // Post-procesado NOISE: local_bpms (repite el ultimo si >1 beat)
        compute_local_bpms(res);
        return res;
    }

    // Analiza un archivo WAV (dr_wav).
    BPMResult analyze_wav(const std::string& path) {
        AudioData audio = load_wav(path);
        return analyze_buffer(audio.samples);
    }

private:
    void compute_local_bpms(BPMResult& res) {
        // Replica de tempo_analysis.py:
        // if len(beat_times) > 1: bpms entre pares + repite el ultimo
        // elif len == 1: [tempo]
        // else: fallback grid (aqui no aplica, devolvemos vacio)
        auto& bt = res.beat_times;
        auto& lb = res.local_bpms;
        if (bt.size() > 1) {
            for (size_t i = 0; i + 1 < bt.size(); ++i) {
                double dt = bt[i + 1] - bt[i];
                lb.push_back(dt > 0.0 ? 60.0 / dt : res.global_tempo);
            }
            lb.push_back(lb.back());
        } else if (bt.size() == 1) {
            lb.push_back(res.global_tempo);
        }
    }

    BPMConfig cfg_;
};

} // namespace librosa_bpm
