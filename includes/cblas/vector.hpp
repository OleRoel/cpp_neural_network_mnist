#ifndef _MKL_VECTOR
#define _MKL_VECTOR

#include <cblas.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <algorithm>
#include <boost/algorithm/string.hpp>                      /* split */

#include "../normal_distribution.hpp"

auto _sigmoid = [](auto d) { return (1.0 / (1.0 + std::exp(-d))); };
auto _derive = [](auto a, auto b) { return a * b * (1.0 - b); };

template<int N> 
class Vector {
    private:
        double* v;
 
    public:
        Vector() : 
            v{new double[N]} {
        }
        Vector(const std::vector<double>& vec) :
            Vector() {
            set(vec);
        }
        ~Vector() {
            delete[] v;
        }

        operator const double* () const { return v; }
        operator double* () { return v; }

        const double* begin() const { return v; }
        double* begin()  { return v; }
        const double* end() const { return &v[N]; }
        double* end()  { return &v[N]; }

        Vector& minus(const Vector& vec) {
            cblas_daxpy(N, -1.0, vec, 1, v, 1);
            return *this;
        }

        Vector& plus(const Vector& vec) {
            cblas_daxpy(N, 1.0, vec, 1, v, 1);        
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

        inline void get(std::vector<double>& vec) const {
            std::memcpy(&vec[0], v, N*sizeof(double));
            // cblas_dcopy(N, v, 0, &vec[0], 0);
        }

        inline Vector& set(const std::vector<double>& vec) {
            std::memcpy(v, &vec[0], N*sizeof(double));
            // cblas_dcopy(N, &vec[0], 0, v, 0);

            return *this;
        }

        inline void sigmoid() {
            std::transform(begin(), end(), begin(), _sigmoid);
        }

        inline void derive(const Vector<N>& v1, const Vector<N>& v2) {
            std::transform(v1.begin(), v1.end(), v2.begin(), v, _derive);
        }

        inline void derive(const Vector<N>& vec) {
            std::transform(begin(), end(), vec.begin(), begin(), _derive);
        }

        void random() {
            NormalDistribution<N> nd;

            std::generate(begin(), end(), nd);
        }

        void print(const std::string& name) const {
            std::cout << name << ":\n" << *this << std::endl;
        }

        void print(const std::string& name, int row, int column) const {
            std::vector<double> matrix(N);
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
            return N;
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