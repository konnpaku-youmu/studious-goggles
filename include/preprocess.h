#ifndef PREPROC_H
#define PREPROC_H

#include <opencv4/opencv2/opencv.hpp>

using LabeledImage = std::map<std::string, cv::Mat>;

class ImageLoader
{
private:
    void make_segment_mask(cv::Mat &src, cv::Mat &mask);

public:
    ImageLoader();

    ~ImageLoader();

    std::vector<cv::Mat> raw_images;

    std::vector<cv::Mat> segment_masks;

    LabeledImage labeled_images;
};

#endif
