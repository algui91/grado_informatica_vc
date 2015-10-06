#include "utils.h"

#include <opencv2/imgcodecs.hpp>

Mat leerImagen(std::string name, int flag) {
    return imread(name, flag ? IMREAD_GRAYSCALE : IMREAD_COLOR);
}
