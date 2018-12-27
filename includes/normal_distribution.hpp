#ifndef _NORMAL_DISTRIBUTION
#define _NORMAL_DISTRIBUTION

#include <random>

template<int N> 
class NormalDistribution {
    private:
        std::random_device rd;
        std::mt19937 gen{rd()};

        double sigma;
        std::normal_distribution<> d;
    public:
        NormalDistribution() :
            rd{},
            gen{rd()},
            sigma{std::pow(N, -0.5)},
            d{0.0, sigma} {

            }
        NormalDistribution(const NormalDistribution& nd) :
            rd{},
            gen{rd()},
            sigma{nd.sigma},
            d{0.0, sigma} {

            }
        ~NormalDistribution() {

        }

    double operator() () {
        return d(gen);
    }
};

#endif