#ifndef _IMAGES_BUFFER
#define _IMAGES_BUFFER

#include <vector>
#include <iostream>
#include <boost/algorithm/string.hpp>                      /* split */
#include <fstream>

template<int N>
class ImagesBuffer {
    private:
        typedef std::vector<double> image_array;
        std::vector<image_array> buffers;
        std::vector<int> numbers;

        void read(const std::string& filename) {
            std::ifstream infile(filename);
            std::string line;
            std::string delims = ",";
            std::vector<std::string> vec;
            image_array inputs(N);
            while (std::getline(infile, line)) {
                boost::split(vec, line, boost::is_any_of(delims));

                std::transform(vec.begin()+1, vec.end(), inputs.begin(), [](const std::string& p) -> double { return (double(std::stoi(p)) / 255.0 * 0.99) + 0.001; });

                int correct_label = std::stoi(vec[0]);

                buffers.push_back(inputs);
                numbers.push_back(correct_label);
            }
        }

    public:
        ImagesBuffer(const std::string& filename) {
            read(filename);
        }

        const std::vector<double>& get_image_array_at(std::size_t idx) const {
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