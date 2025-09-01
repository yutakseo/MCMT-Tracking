#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class StreamCCTV {
public:
    StreamCCTV(const std::string& url, int maxWidth = 0, int reconnectDelay = 2)
        : url_(url), maxWidth_(maxWidth), reconnectDelay_(reconnectDelay), stopFlag_(false) {}

    void start() {
        stopFlag_ = false;
        worker_ = std::thread(&StreamCCTV::update, this);
    }

    void stop() {
        stopFlag_ = true;
        if (worker_.joinable()) worker_.join();
        releaseCap();
    }

    py::array_t<unsigned char> capture(bool copy = false) {
        std::lock_guard<std::mutex> lock(frameMutex_);
        if (latestFrame_.empty()) return py::array_t<unsigned char>();

        cv::Mat frame = copy ? latestFrame_.clone() : latestFrame_;

        // 구버전 pybind11 호환: shape, strides를 vector로 넘겨야 함
        std::vector<ssize_t> shape   = { frame.rows, frame.cols, frame.channels() };
        std::vector<ssize_t> strides = { (ssize_t)frame.step[0], (ssize_t)frame.step[1], 1 };

        return py::array_t<unsigned char>(
            py::buffer_info(
                frame.data,
                sizeof(unsigned char),
                py::format_descriptor<unsigned char>::format(),
                3,
                shape,
                strides
            )
        );
}
private:
    std::string url_;
    int maxWidth_;
    int reconnectDelay_;
    cv::VideoCapture cap_;
    std::thread worker_;
    std::mutex frameMutex_;
    cv::Mat latestFrame_;
    std::atomic<bool> stopFlag_;

    void releaseCap() {
        if (cap_.isOpened()) cap_.release();
    }

    cv::Mat resizeIfNeeded(const cv::Mat& frame) {
        if (maxWidth_ > 0 && frame.cols > maxWidth_) {
            double scale = (double)maxWidth_ / frame.cols;
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(), scale, scale);
            return resized;
        }
        return frame;
    }

    void update() {
        while (!stopFlag_) {
            if (!cap_.isOpened()) {
                cap_.open(url_);
                if (!cap_.isOpened()) {
                    std::this_thread::sleep_for(std::chrono::seconds(reconnectDelay_));
                    continue;
                }
                cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
            }

            cv::Mat frame;
            if (!cap_.read(frame)) {
                releaseCap();
                std::this_thread::sleep_for(std::chrono::seconds(reconnectDelay_));
                continue;
            }

            frame = resizeIfNeeded(frame);

            {
                std::lock_guard<std::mutex> lock(frameMutex_);
                latestFrame_ = frame;
            }
        }
        releaseCap();
    }
};

// pybind11 모듈 정의
PYBIND11_MODULE(stream_cctv_cpp, m) {
    py::class_<StreamCCTV>(m, "StreamCCTV")
        .def(py::init<const std::string&, int, int>())
        .def("start", &StreamCCTV::start)
        .def("stop", &StreamCCTV::stop)
        .def("capture", &StreamCCTV::capture, py::arg("copy")=false);
}
