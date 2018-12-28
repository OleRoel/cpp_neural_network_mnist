#ifndef _IMAGES_BUFFER
#define _IMAGES_BUFFER

#include <vector>
#include <iostream>
#include <boost/algorithm/string.hpp>  /* split */
#include <fstream>

template<int M> class Vector;
template<int M, int N> class Matrix;

template<int M>
class ImagesBuffer {
    private:
        typedef Vector<M> image_array;
        std::vector<image_array> buffers;
        std::vector<int> numbers;

        void read(const std::string& filename) {
            std::ifstream infile(filename);
            std::string line;
            std::string delims = ",";
            std::vector<std::string> vec;
            
            for (auto&& p : buffers)
            {
                if (std::getline(infile, line)) {
                    boost::split(vec, line, boost::is_any_of(delims));

                    std::transform(vec.begin()+1, vec.end(), p.begin(), [](const std::string& p) -> float { return (float(std::stoi(p)) / 255.0 * 0.99) + 0.001; });

                    int correct_label = std::stoi(vec[0]);

                    numbers.push_back(correct_label);
                }
            }
        }

    public:
        ImagesBuffer(const std::string& filename, std::size_t buf_size) :
            buffers(buf_size)
        {
            read(filename);
        }

        const Vector<M>& get_image_array_at(std::size_t idx) const {
            return buffers[idx];
        }
        
        const int get_number_at(std::size_t idx) const {
            return numbers[idx];
        }

        std::size_t size() const {
            return numbers.size();
        }
};

#endif