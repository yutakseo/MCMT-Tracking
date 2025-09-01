#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <chrono>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class StreamCCTV {
public:
    StreamCCTV(const std::string& url, int maxWidth = 0, int reconnectDelay = 2)
        : url_(url),
          maxWidth_(maxWidth),
          reconnectDelay_(reconnectDelay),
          stopFlag_(false),
          started_(false) {}

    ~StreamCCTV() {
        try {
            stop();
        } catch (...) {
            // destructor에서는 예외를 외부로 던지지 않음
        }
    }

    void start() {
        // 중복 시작 방지
        if (started_.exchange(true)) return;
        stopFlag_ = false;
        worker_ = std::thread(&StreamCCTV::update, this);
    }

    void stop() {
        if (!started_.exchange(false)) return;
        stopFlag_ = true;
        if (worker_.joinable()) worker_.join();
        releaseCap();
    }

    // 최신 프레임 1장을 넘파이(BGR, HxWx3)로 반환
    // copy=True면 Mat clone → 독립 버퍼 / False면 zero-copy지만 capsule로 수명 보장
    py::array_t<unsigned char> capture(bool copy = false) {
        std::lock_guard<std::mutex> lock(frameMutex_);
        if (latestFrame_.empty()) {
            return py::array_t<unsigned char>(); // 빈 배열
        }

        cv::Mat mat = copy ? latestFrame_.clone() : latestFrame_;

        // numpy가 참조하는 동안 메모리 보장을 위해 capsule에 cv::Mat 소유권을 넣는다.
        // (pybind11 3.0.1: buffer_info + base(handle) 생성자 사용)
        cv::Mat* mat_ptr = new cv::Mat(std::move(mat));

        std::vector<ssize_t> shape   = { mat_ptr->rows, mat_ptr->cols, mat_ptr->channels() };
        std::vector<ssize_t> strides = { (ssize_t)mat_ptr->step[0], (ssize_t)mat_ptr->step[1], 1 };

        py::capsule base(mat_ptr, [](void* p){
            delete reinterpret_cast<cv::Mat*>(p);
        });

        // 구버전 호환 buffer_info 생성자 사용
        py::buffer_info info(
            (void*)mat_ptr->data,
            sizeof(unsigned char),
            py::format_descriptor<unsigned char>::format(),
            3,
            shape,
            strides,
            /*readonly=*/false
        );

        // base(handle)를 함께 넘겨 Mat 수명을 numpy 배열과 동기화
        return py::array_t<unsigned char>(info, base);
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
    std::atomic<bool> started_;

    void releaseCap() {
        if (cap_.isOpened()) cap_.release();
    }

    cv::Mat resizeIfNeeded(const cv::Mat& frame) {
        if (maxWidth_ > 0 && frame.cols > maxWidth_) {
            double scale = static_cast<double>(maxWidth_) / frame.cols;
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);
            return resized;
        }
        return frame;
    }

    void update() {
        while (!stopFlag_) {
            if (!cap_.isOpened()) {
                // 필요한 경우 FFMPEG 명시: cap_.open(url_, cv::CAP_FFMPEG);
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
                latestFrame_ = std::move(frame); // move로 교체
            }
        }
        releaseCap();
    }
};


// pybind11 모듈 정의
PYBIND11_MODULE(stream_cctv_cpp, m) {
    py::class_<StreamCCTV>(m, "StreamCCTV")
        // Python에서 기본값/키워드 인자 사용 가능하게
        .def(py::init<const std::string&, int, int>(),
             py::arg("url"),
             py::arg("maxWidth") = 0,
             py::arg("reconnectDelay") = 2)
        .def("start",   &StreamCCTV::start)
        .def("stop",    &StreamCCTV::stop)
        .def("capture", &StreamCCTV::capture, py::arg("copy") = false);
}
