#ifndef _NORMAL_DISTRIBUTION
#define _NORMAL_DISTRIBUTION

#include <random>

template<int N> 
class NormalDistribution {
    private:
        std::random_device rd;
        std::mt19937 gen{rd()};

        float sigma;
        std::normal_distribution<> d;
    public:
        NormalDistribution() :
            rd{},
            gen{rd()},
            sigma{std::powf(N, -0.5F)},
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

    float operator() () {
        return d(gen);
    }
};

#endif