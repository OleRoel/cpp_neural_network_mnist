#ifndef _MKL_VECTOR
#define _MKL_VECTOR

#include <mkl.h>

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
            cblas_dgemv(CblasColMajor, CblasNoTrans, M, N, 1.0, *this, M, vec, 1, 0.0, result, 1);
            return result;
        }

        inline Vector<N>& multiply_transposed(const Vector<M>& vec, Vector<N>& result) const {
            cblas_dgemv(CblasColMajor, CblasTrans, M, N, 1.0, *this, M, vec, 1, 0.0, result, 1);
            return result;
        }

        inline Matrix& multiply(const Vector<M>& vec1, const Vector<N>& vec2, double factor) {
            cblas_dger(CblasColMajor, M, N, factor, vec1, 1, vec2, 1, *this, M);
            return *this;
        }
};

auto _sigmoid = [](auto d) { return (1.0 / (1.0 + std::exp(-d))); };
auto _derive = [](auto a, auto b) { return a * b * (1.0 - b); };

template<int M> 
class Vector {
    private:
        double* v;
 
    public:
        Vector() : 
            v{static_cast<double*>(mkl_malloc(M*sizeof(double), 64))} {
        }
        Vector(const std::vector<double>& vec) :
            Vector() {
            set(vec);
        }
        ~Vector() {
            mkl_free(v);
        }

        operator const double* () const { return v; }
        operator double* () { return v; }

        const double* begin() const { return v; }
        double* begin()  { return v; }
        const double* end() const { return &v[M]; }
        double* end()  { return &v[M]; }

        Vector& minus(const Vector& vec) {
            cblas_daxpy(M, -1.0, vec, 1, v, 1);
            return *this;
        }

        Vector& plus(const Vector& vec) {
            cblas_daxpy(M, 1.0, vec, 1, v, 1);        
            return *this;
        }

        inline Vector& operator = (const std::vector<double>& vec) {
            return set(vec);
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

        inline void get(std::vector<double>& vec) const {
            std::memcpy(&vec[0], v, M*sizeof(double));
            // cblas_dcopy(N, v, 0, &vec[0], 0);
        }

        inline Vector& set(const std::vector<double>& vec) {
            std::memcpy(v, &vec[0], M*sizeof(double));
            // cblas_dcopy(N, &vec[0], 0, v, 0);

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

        void print(const std::string& name, int row, int column) const {
            std::vector<double> matrix(M);
            get(matrix);

            std::cout << "Matrix " << name << " has " << row << " rows and " << column << " columns:\n";
            for (int i = 0; i < row; i++){
                for (int j = 0; j < column; j++) {
                    // std::cout << '[' << i << ',' << j << ']'<< std::setw(6) << matrix[i*column + j] << " ";
                    std::cout << std::setw(6) << matrix[i*column + j] << " ";
                }
                std::cout << '\n';
            }
            std::cout << std::endl;
        }

        void read(const std::string& filename) {
            std::ifstream infile(filename);
            infile >> *this;
        }

        inline int size() const {
            return M;
        }
};

template<int N> 
std::ostream& operator << (std::ostream& os, const Vector<N>& obj)
{
    for (std::size_t i = 0; i < N; i++) {
        os << obj[i] <<  ", ";
    }

    return os;
}

template<int N> 
std::istream& operator >> (std::istream& is, Vector<N>& obj)
{
    std::string line;
    std::string delims = ",";
    double* it = obj.begin();
    std::vector<std::string> vec(N);

    while (std::getline(is, line)) {
        boost::split(vec, line, boost::is_any_of(delims));
        it = std::transform(vec.begin(), vec.end(), it, [](const std::string& p) -> double { return std::stod(p); });
    }

    return is;
}

#endif