#ifndef _MKL_LAYER
#define _MKL_LAYER

// #include "mkl.h"

// #include "mkl_vector.hpp"

template<int M, int N>
class Layer {
    private:
        Vector<M*N> weights;
        Vector<M> neurons;

        Vector<M> errors;

        double learningrate;
        
    public:
        Layer(double learningrate) : learningrate{learningrate} {
            weights.random();
        }
        Layer(const std::string& filename, double learningrate) : learningrate{learningrate} {
            weights.read(filename);
        }

        void feed_forward(const Vector<N>& inputs) {
            cblas_dgemv(CblasColMajor, CblasNoTrans, M, N, 1.0, weights, M, inputs, 1, 0.0, neurons, 1);
            neurons.sigmoid();
        }

        void backpropagate_errors(Vector<N>& out_errors) {
            cblas_dgemv(CblasColMajor, CblasTrans, M, N, 1.0, weights, M, errors, 1, 0.0, out_errors, 1);
        }

        void backpropagate(const Vector<N>& inputs) {
            errors.derive(neurons);
            cblas_dger(CblasColMajor, M, N, learningrate, errors, 1, inputs, 1, weights, M);
        }

        inline const  Vector<M*N>& get_weights() const {
            return weights;
        }

        inline const Vector<M>& get_neurons() const {
            return neurons;
        }

        inline const Vector<M>& get_errors() const {
            return errors;
        }

        inline Vector<M>& get_errors() {
            return errors;
        }

        inline void calc_errors(const std::vector<double>& vec) {
            errors = vec;
            errors -= neurons;
        }
};

#endif