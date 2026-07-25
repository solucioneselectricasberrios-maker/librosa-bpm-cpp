// matrix.h — Contenedor 2D row-major minimal (sin Eigen).
// Matrix<double> almacena (rows x cols) en un std::vector<double> contiguo.
#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <algorithm>

namespace librosa_bpm {

template <typename T>
class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}
    Matrix(std::size_t rows, std::size_t cols, T fill = T{})
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t size()  const { return data_.size(); }

    // acceso row-major: M(fila, col)
    T&       operator()(std::size_t r, std::size_t c)       { return data_[r * cols_ + c]; }
    const T& operator()(std::size_t r, std::size_t c) const { return data_[r * cols_ + c]; }

    // acceso lineal
    T&       operator[](std::size_t i)       { return data_[i]; }
    const T& operator[](std::size_t i) const { return data_[i]; }

    T*       data()       { return data_.data(); }
    const T* data() const { return data_.data(); }

    std::vector<T>&       storage()       { return data_; }
    const std::vector<T>& storage() const { return data_; }

    void fill(T v) { std::fill(data_.begin(), data_.end(), v); }

private:
    std::size_t rows_, cols_;
    std::vector<T> data_;
};

// Multiplicacion matriz-matriz C = A (m x k) * B (k x n) -> (m x n)
// Usado para S_mel = melfb (128x1025) * S_power (1025xM)
inline Matrix<double> matmul(const Matrix<double>& A, const Matrix<double>& B) {
    std::size_t m = A.rows(), k = A.cols(), n = B.cols();
    if (B.rows() != k) throw std::runtime_error("matmul: dimension mismatch");
    Matrix<double> C(m, n, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        const double* arow = &A(i, 0);
        for (std::size_t p = 0; p < k; ++p) {
            double a = arow[p];
            if (a == 0.0) continue;
            const double* brow = &B(p, 0);
            double* crow = &C(i, 0);
            for (std::size_t j = 0; j < n; ++j) {
                crow[j] += a * brow[j];
            }
        }
    }
    return C;
}

} // namespace librosa_bpm
