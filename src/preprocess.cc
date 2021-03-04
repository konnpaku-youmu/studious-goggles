#include "preprocess.h"

ImageLoader::ImageLoader()
{
    std::vector<std::string> file_names;
    cv::glob("../images/*.jpeg", file_names, true);

    for (auto name : file_names)
    {
        cv::Mat resized_image, raw_image;
        raw_image = cv::imread(name);
        cv::resize(raw_image, resized_image, cv::Size(640, 360));

        cv::Mat segment_mask;
        make_segment_mask(resized_image, segment_mask);

        this->segment_masks.push_back(segment_mask);
        this->raw_images.push_back(resized_image);
    }

    return;
}

ImageLoader::~ImageLoader()
{
}

void ImageLoader::make_segment_mask(cv::Mat &src, cv::Mat &mask)
{
    cv::Mat src_mono;
    cv::cvtColor(src, src_mono, cv::COLOR_BGR2GRAY);

    cv::Mat src_threshold;
    cv::threshold(src_mono, src_threshold, 64, 255, cv::THRESH_BINARY_INV);

    cv::imshow("window", src_threshold);

    // connected components analysis
    cv::medianBlur(src_threshold, src_threshold, 7);
    cv::Mat cca_container;
    int connected_areas_cnt = cv::connectedComponents(src_threshold, cca_container);

    cca_container.convertTo(cca_container, CV_8UC1);
    int32_t total_pixel_num = cca_container.rows * cca_container.cols;

    cv::Mat cca_histogram;
    int histSize = 256;
    float range[] = {0, 256}; //the upper boundary is exclusive
    const float *histRange = {range};
    cv::calcHist(&cca_container, 1, 0, cv::Mat(), cca_histogram, 1, &histSize, &histRange);

    float avg_area = (total_pixel_num - cca_histogram.at<float>(0)) / connected_areas_cnt;
    
    std::vector<int> valid_area_idx;
    // std::map<int, cv::Rect> RoI;

    for (int i = 1; i < 255; ++i)
    {
        if (cca_histogram.at<float>(i) == 0)
        {
            break;
        }
        else if (cca_histogram.at<float>(i) >= 1.2 * avg_area)
        {
            valid_area_idx.push_back(i);
            // RoI.emplace(std::make_pair(i, cv::Rect(0, 0, 0, 0)));
        }
    }

    cv::Mat dst_mono = src_mono.clone();
    mask = cca_container.clone();

    for(int row = 0; row < cca_container.rows; ++row)
    {
        for(int col = 0; col < cca_container.cols; ++col)
        {
            int val = cca_container.at<uint8_t>(row, col);
            bool isValid = false;
            int area_idx = -1;
            for(auto idx = valid_area_idx.begin(); idx != valid_area_idx.end(); ++idx)
            {
                if(val == *idx)
                {
                    isValid = true;
                    area_idx = idx - valid_area_idx.begin();
                    break;
                }
            }

            if(!isValid)
            {
                mask.at<uint8_t>(row, col) = 0;
                dst_mono.at<uint8_t>(row, col) = 0;
            }
        }
    }

    cv::normalize(mask, mask, 0, 255, cv::NORM_MINMAX);

    std::vector<cv::Mat> display_images;
    display_images.push_back(src_mono);
    display_images.push_back(mask);
    display_images.push_back(dst_mono);
    cv::Mat concat;
    cv::hconcat(&display_images.front(), 3, concat);
    cv::imshow("window", concat);
    cv::waitKey(0);

    return;
}
