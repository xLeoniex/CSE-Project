#include "interpolation.hpp"

#include <utility>

namespace tensor_network {
    static void interpolation(cv::Mat data);

    static int mean_width = 26;
    static int mean_height = 26;

    static void interpolation(cv::Mat data) {
        cv::resize(data, data, {mean_width, mean_height}, cv::INTER_LINEAR);
    }

    auto run(cv::Mat cone_data) -> std::pair<Cones, double> {
        interpolation(std::move(cone_data));
        // Call Tensor Network
        // auto result = run_tensor_network(cone_data);
        // result[0] = Farbe
        // result[1] = probability
        label = model.predict()


        return std::pair<Cones, double>{Cones::BLUE, 0.9999999};
    }

} // namespace tensor_network