// golden.h — Lector de arrays del golden master (tests/golden/bin/*.bin + manifest.json)
// Lee little-endian C-contiguous. No depende de cnpy/zlib.
#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <nlohmann/json.hpp>

namespace librosa_bpm {
namespace test {

struct GoldenEntry {
    std::string file;
    std::vector<int> shape;
    std::string dtype;   // "<f8", "<i8", "?"
};

class Golden {
public:
    explicit Golden(const std::string& bin_dir) : bin_dir_(bin_dir) {
        std::ifstream ifs(bin_dir + "/golden_manifest.json");
        if (!ifs) throw std::runtime_error("no se encontro golden_manifest.json en " + bin_dir);
        nlohmann::json j;
        ifs >> j;
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (it.key().rfind("_", 0) == 0) continue;  // saltar _comment
            const auto& v = it.value();
            GoldenEntry e;
            e.file = v["file"].get<std::string>();
            for (const auto& s : v["shape"]) e.shape.push_back(s.get<int>());
            e.dtype = v["dtype"].get<std::string>();
            entries_[it.key()] = std::move(e);
        }
    }

    // Lee un array escalar o 1D/2D como double. Los bool se convierten a 0/1.
    std::vector<double> read_double(const std::string& key) const {
        auto it = entries_.find(key);
        if (it == entries_.end()) throw std::runtime_error("golden key no existe: " + key);
        const GoldenEntry& e = it->second;
        std::size_t n = 1;
        for (int s : e.shape) n *= std::size_t(s > 0 ? s : 1);
        std::vector<double> out(n);
        std::ifstream f(bin_dir_ + "/" + e.file, std::ios::binary);
        if (!f) throw std::runtime_error("no se pudo abrir " + e.file);
        if (e.dtype == "<f8") {
            f.read(reinterpret_cast<char*>(out.data()), n * 8);
        } else if (e.dtype == "<i8") {
            std::vector<int64_t> tmp(n);
            f.read(reinterpret_cast<char*>(tmp.data()), n * 8);
            for (std::size_t i = 0; i < n; ++i) out[i] = double(tmp[i]);
        } else if (e.dtype == "?") {
            std::vector<uint8_t> tmp(n);
            f.read(reinterpret_cast<char*>(tmp.data()), n);
            for (std::size_t i = 0; i < n; ++i) out[i] = tmp[i] ? 1.0 : 0.0;
        } else {
            throw std::runtime_error("dtype no soportado: " + e.dtype);
        }
        return out;
    }

    int64_t read_int_scalar(const std::string& key) const {
        std::vector<double> v = read_double(key);
        return int64_t(v[0]);
    }

    const GoldenEntry& info(const std::string& key) const {
        auto it = entries_.find(key);
        if (it == entries_.end()) throw std::runtime_error("golden key no existe: " + key);
        return it->second;
    }

private:
    std::string bin_dir_;
    std::unordered_map<std::string, GoldenEntry> entries_;
};

// --- Metricas de comparacion (espejo de validate_reference_impl.py) ---
inline double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

inline double mean_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += std::fabs(a[i] - b[i]);
    return s / double(a.size());
}

inline double correlation(const std::vector<double>& a, const std::vector<double>& b) {
    double ma = 0, mb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) { ma += a[i]; mb += b[i]; }
    ma /= a.size(); mb /= a.size();
    double cov = 0, va = 0, vb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double da = a[i] - ma, db = b[i] - mb;
        cov += da * db; va += da * da; vb += db * db;
    }
    if (va == 0 || vb == 0) return 0.0;
    return cov / std::sqrt(va * vb);
}

} // namespace test
} // namespace librosa_bpm
