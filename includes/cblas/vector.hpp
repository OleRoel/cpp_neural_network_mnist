#ifndef _CBLAS_VECTOR
#define _CBLAS_VECTOR

#if defined(TARGET_MKL)
#include <mkl.h>
#else
#include <cblas.h>
#endif

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <algorithm>
#include <boost/algorithm/string.hpp>                      /* split */

#include "../normal_distribution.hpp"

template<int M> class Vector;

template<int M, int N>
class Matrix : public Vector<M*N> {
    public:

        inline Vector<M>& multiply(const Vector<N>& vec, Vector<M>& result) const {
            cblas_sgemv(CblasColMajor, CblasNoTrans, M, N, 1.0, *this, M, vec, 1, 0.0, result, 1);
            return result;
        }

        inline Vector<N>& multiply_transposed(const Vector<M>& vec, Vector<N>& result) const {
            cblas_sgemv(CblasColMajor, CblasTrans, M, N, 1.0, *this, M, vec, 1, 0.0, result, 1);
            return result;
        }

        inline Matrix& multiply(const Vector<M>& vec1, const Vector<N>& vec2, float factor) {
            cblas_sger(CblasColMajor, M, N, factor, vec1, 1, vec2, 1, *this, M);
            return *this;
        }

        void print(const std::string& name) const {
            std::cout << "Matrix " << name << " has " << M << " rows and " << N << " columns:\n";
            std::cout <<  *this << std::endl;
        }
};

auto _sigmoid = [](auto d) { return (1.0 / (1.0 + std::exp(-d))); };
auto _derive = [](auto a, auto b) { return a * b * (1.0 - b); };

template<std::size_t M> float* allocate_buffer() {
#if defined(TARGET_MKL)
    return (float*)(mkl_malloc(M*sizeof(float), 64));
#else
    return new float[M];
#endif
}
void free_buffer(float* buff) {
#if defined(TARGET_MKL)
    mkl_free(buff);
#else
    delete[] buff;
#endif
}

template<int M> 
class Vector {
    private:
        float* v;
 
    public:
        Vector() : 
            v{allocate_buffer<M>()} {
        }
        Vector(const std::vector<float>& vec) :
            Vector() {
            set(vec);
        }
        Vector(const float vec[10]) :
            Vector() {
            set(vec);
        }
        Vector(const Vector& vec) :
            Vector() {
            set(vec.v);
        }
        ~Vector() {
            free_buffer(v);
        }

        operator const float* () const { return v; }
        operator float* () { return v; }

        const float* begin() const { return v; }
        float* begin()  { return v; }
        const float* end() const { return &v[M]; }
        float* end()  { return &v[M]; }

        float& operator[](std::size_t idx)       { return v[idx]; }
        const float& operator[](std::size_t idx) const { return v[idx]; }

        Vector& minus(const Vector& vec) {
            cblas_saxpy(M, -1.0, vec, 1, v, 1);
            return *this;
        }

        Vector& plus(const Vector& vec) {
            cblas_saxpy(M, 1.0, vec, 1, v, 1);        
            return *this;
        }

        inline Vector& operator = (const std::vector<float>& vec) {
            return set(vec);
        }

        inline Vector& operator = (const Vector& vec) {
            return set(vec.v);
        }

        inline Vector& operator -= (const Vector& vec) {
            return minus(vec);
        }

        inline Vector& operator += (const Vector& vec) {
            return plus(vec);
        }

        template<int N>
        inline Vector& multiply(const Matrix<M, N>& matrix, const Vector<N>& vec) {
            return matrix.multiply(vec, *this);
        }

        template<int N>
        inline Vector& multiply_transposed(const Matrix<M, N>& matrix, const Vector<M>& vec) {
            return matrix.multiply_transposed(vec, *this);
        }

        inline void get(std::vector<float>& vec) const {
            std::memcpy(&vec[0], v, M*sizeof(float));
            // cblas_scopy(N, v, 0, &vec[0], 0);
        }

        inline void get(Vector& vec) const {
            std::memcpy(vec.v, v, M*sizeof(float));
            // cblas_scopy(N, v, 0, &vec[0], 0);
        }

        inline Vector& set(const std::vector<float>& vec) {
            std::memcpy(v, &vec[0], M*sizeof(float));
            // cblas_scopy(N, &vec[0], 0, v, 0);

            return *this;
        }

        inline Vector& set(const Vector& vec) {
            std::memcpy(v, vec.v, M*sizeof(float));
            // cblas_scopy(N, &vec[0], 0, v, 0);

            return *this;
        }

        inline Vector& set(const float vec[10]) {
            std::memcpy(v, vec, M*sizeof(float));

            return *this;
        }

        inline Vector& set_at(std::size_t idx, float val) {
            v[idx] = val;
            return *this;
        }

        inline void sigmoid() {
            std::transform(begin(), end(), begin(), _sigmoid);
        }

        inline void derive(const Vector<M>& v1, const Vector<M>& v2) {
            std::transform(v1.begin(), v1.end(), v2.begin(), v, _derive);
        }

        inline void derive(const Vector<M>& vec) {
            std::transform(begin(), end(), vec.begin(), begin(), _derive);
        }

        void random() {
            NormalDistribution<M> nd;

            std::generate(begin(), end(), nd);
        }

        void print(const std::string& name) const {
            std::cout << name << ":\n" << *this << std::endl;
        }

        void read(const std::string& filename) {
            std::ifstream infile(filename);
            infile >> *this;
        }

        inline int size() const {
            return M;
        }
};

template<int M>
std::ostream& operator << (std::ostream& os, const Vector<M>& obj)
{
    for (std::size_t i = 0; i < M; i++) {
        os << obj[i] <<  ", ";
    }

    return os;
}

template<int M, int N>
std::ostream& operator << (std::ostream& os, const Matrix<M, N>& obj)
{
    for (std::size_t i = 0; i < N; i++){
        for (std::size_t j = 0; j < N; j++) {
            // std::cout << '[' << i << ',' << j << ']'<< std::setw(6) << matrix[i*column + j] << " ";
            std::cout << std::setw(6) << obj[i*M + j] << " ";
        }
        std::cout << '\n';
    }

    return os;
}


template<int N> 
std::istream& operator >> (std::istream& is, Vector<N>& obj)
{
    std::string line;
    std::string delims = ",";
    float* it = obj.begin();
    std::vector<std::string> vec(N);

    while (std::getline(is, line)) {
        boost::split(vec, line, boost::is_any_of(delims));
        it = std::transform(vec.begin(), vec.end(), it, [](const std::string& p) -> float { return std::stod(p); });
    }

    return is;
}

#endif