// audio_loader.h — Carga de audio a mono float64 @SR_ANALYSIS
// Estrategia WAV-first: dr_wav lee el WAV 32-bit float bit-exacto vs soundfile.
#pragma once
#include <vector>
#include <string>

namespace librosa_bpm {

struct AudioData {
    std::vector<double> samples;   // mono, float64, remuestreado a SR_ANALYSIS
    int sample_rate = 22050;
    long long n_frames = 0;        // muestras totales
};

// Carga un archivo WAV. Devuelve mono float64 @SR_ANALYSIS.
// (El resampling se anadira en Fase 2b; por ahora el WAV ya esta a 22050.)
AudioData load_wav(const std::string& path);

// Carga PCM crudo float32 little-endian (respaldo: bypassea el decodificador).
// n_samples: numero de muestras mono esperadas.
AudioData load_pcm_f32(const std::string& path, long long n_samples);

} // namespace librosa_bpm
