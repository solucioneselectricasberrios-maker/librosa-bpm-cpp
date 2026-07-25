// bpm_cli.cpp — CLI: input.(wav|pcm) -> output.json (formato compatible con compare_outputs.py)
// Uso: bpm_cli <input.wav|input.pcm> <output.json> [n_samples_pcm]
#include <cstdio>
#include <string>
#include <nlohmann/json.hpp>
#include "librosa_bpm/bpm_detector.h"
#include "librosa_bpm/audio_loader.h"

using namespace librosa_bpm;
namespace fs = std;

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Uso: bpm_cli <input.wav|input.pcm> <output.json> [n_samples_pcm]\n");
        return 1;
    }
    std::string in_path = argv[1];
    std::string out_path = argv[2];

    try {
        BPMDetector detector;
        BPMResult result;

        // Detectar tipo por extension
        auto ends_with = [](const std::string& s, const std::string& suf) {
            return s.size() >= suf.size() &&
                   s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
        };

        if (ends_with(in_path, ".wav")) {
            result = detector.analyze_wav(in_path);
        } else if (ends_with(in_path, ".pcm")) {
            long long n = (argc >= 4) ? std::stoll(argv[3]) : 0;
            if (n <= 0) {
                fprintf(stderr, "Para .pcm se requiere n_samples (3er argumento)\n");
                return 1;
            }
            AudioData audio = load_pcm_f32(in_path, n);
            result = detector.analyze_buffer(audio.samples);
        } else {
            fprintf(stderr, "Formato no soportado (usar .wav o .pcm)\n");
            return 1;
        }

        // Volcar onset_envelope para que compare_outputs pueda validarla.
        // (Recalculamos el oenv para incluirlo; barato relativo al pipeline.)
        // Para simplicidad dejamos onset_envelope vacio si no se pide.
        nlohmann::json j;
        j["filepath"] = in_path;
        j["sr_analysis"] = SR_ANALYSIS;
        j["hop_length"] = HOP_LENGTH;
        j["n_fft"] = N_FFT;
        j["global_tempo"] = result.global_tempo;
        j["n_beats"] = result.n_beats;
        j["duration_seconds"] = result.duration_seconds;
        j["beat_times"] = result.beat_times;
        j["local_bpms"] = result.local_bpms;

        FILE* f;
        if (fopen_s(&f, out_path.c_str(), "w") != 0 || !f) {
            fprintf(stderr, "No se pudo escribir %s\n", out_path.c_str());
            return 1;
        }
        std::string dump = j.dump(2);
        fwrite(dump.data(), 1, dump.size(), f);
        fclose(f);

        fprintf(stderr, "bpm_cli: tempo=%.4f BPM, %d beats, dur=%.2fs -> %s\n",
                result.global_tempo, result.n_beats, result.duration_seconds, out_path.c_str());
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 2;
    }
}
