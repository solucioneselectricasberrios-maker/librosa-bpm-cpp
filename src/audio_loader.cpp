// audio_loader.cpp — dr_wav + PCM crudo
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "librosa_bpm/audio_loader.h"
#include "librosa_bpm/constants.h"
#include <cstdio>
#include <stdexcept>

namespace librosa_bpm {

AudioData load_wav(const std::string& path) {
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        throw std::runtime_error("load_wav: no se pudo abrir " + path);
    }

    // Leer todos los frames (dr_wav devuelve interleaved float)
    std::vector<float> interleaved(wav.totalPCMFrameCount * wav.channels);
    drwav_uint64 read = drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount,
                                                  interleaved.data());
    drwav_uninit(&wav);

    AudioData out;
    out.sample_rate = SR_ANALYSIS;   // asumimos ya remuestreado (el WAV golden lo esta)

    // Mezclar a mono (media de canales) y promover a float64
    out.samples.reserve(read);
    for (drwav_uint64 i = 0; i < read; ++i) {
        double s = 0.0;
        for (unsigned c = 0; c < wav.channels; ++c) {
            s += interleaved[i * wav.channels + c];
        }
        out.samples.push_back(s / double(wav.channels));
    }
    out.n_frames = static_cast<long long>(out.samples.size());
    return out;
}

AudioData load_pcm_f32(const std::string& path, long long n_samples) {
    FILE* f;
    if (fopen_s(&f, path.c_str(), "rb") != 0 || !f) {
        throw std::runtime_error("load_pcm_f32: no se pudo abrir " + path);
    }
    std::vector<float> buf(n_samples);
    size_t got = fread(buf.data(), sizeof(float), n_samples, f);
    fclose(f);

    AudioData out;
    out.sample_rate = SR_ANALYSIS;
    out.samples.reserve(got);
    for (size_t i = 0; i < got; ++i) out.samples.push_back(double(buf[i]));
    out.n_frames = static_cast<long long>(out.samples.size());
    return out;
}

} // namespace librosa_bpm
