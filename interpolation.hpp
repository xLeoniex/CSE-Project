#ifndef AUTONOMOUS_SYSTEM_INTERPOLATION_HPP
#define AUTONOMOUS_SYSTEM_INTERPOLATION_HPP

#include <opencv2/opencv.hpp>

namespace tensor_network {
    enum class Cones {
        ORANGE = 0,
        BLUE = 1,
        YELLOW = 2,
        NO_CONE = 3,
        OTHER = 4
    };

    auto run(cv::Mat cone_data) -> std::pair<Cones, double>;

}

#endif // AUTONOMOUS_SYSTEM_INTERPOLATION_HPP
