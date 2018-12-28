#ifndef _MKL_LAYER
#define _MKL_LAYER

#include <boost/preprocessor/stringize.hpp>

template<int M> class Vector;
template<int M, int N> class Matrix;

template<int M, int N>
class Layer {
    private:
        Matrix<M, N> weights;
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
            weights.multiply(inputs, neurons).sigmoid();
        }

        void backpropagate_errors(Vector<N>& out_errors) {
            weights.multiply_transposed(errors, out_errors);
        }

        void backpropagate(const Vector<N>& inputs) {
            errors.derive(neurons);
            weights.multiply(errors, inputs, learningrate);
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

        inline void calc_errors(const Vector<M>& vec) {
            errors = vec;
            errors -= neurons;
        }
};

#endif