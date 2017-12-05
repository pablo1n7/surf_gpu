#include <boost/python.hpp>
#include <stdexcept>
 
#include <string>
#include <iostream>
#include <fstream>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"


using namespace std;
namespace bp = boost::python;


 std::vector<float> keypoints_to_vector(vector<cv::KeyPoint> mat){
     std::vector<float> vector;
     for(std::size_t i = 0; i != (mat.size()); i++) {
         vector.push_back(mat[i].pt.x);
         vector.push_back(mat[i].pt.y);
     }
     return vector;
 }

template <class T>
bp::list vector_to_pythonlist(std::vector<T> vector) {
    typename std::vector<T>::iterator iter;
    bp::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        list.append(*iter);
    }
    return list;
}

bp::list compute(std::string img_root){
    cv::cuda::GpuMat img;
    std::string folder_result;
    img.upload(cv::imread(img_root, cv::IMREAD_GRAYSCALE));
    CV_Assert(!img.empty());
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    cv::cuda::SURF_CUDA surf;
    cv::cuda::GpuMat keypointsGPU;
    cv::cuda::GpuMat descriptorsGPU;
    surf(img, cv::cuda::GpuMat(), keypointsGPU, descriptorsGPU);
    vector<cv::KeyPoint> keypoints;
    vector<float> descriptors; 
    surf.downloadKeypoints(keypointsGPU, keypoints);
    surf.downloadDescriptors(descriptorsGPU, descriptors);
    return vector_to_pythonlist(keypoints_to_vector(keypoints));
}   


BOOST_PYTHON_MODULE(libsurfgpu){
    Py_Initialize(); 
    bp::def("compute", compute);
  //  bp::def("test", test);
}
