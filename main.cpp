#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;


void addImageToComposition(Mat& composedImage, Mat& image, const bool isGrayscale, const int quarter) {
    const auto width = composedImage.size().width;
    const auto height = composedImage.size().height;
    const auto halfWidth = width / 2;
    const auto halfHeight = height / 2;

    resize(image, image, Size(halfWidth, halfHeight));
    if (isGrayscale) {
        cvtColor(image, image, COLOR_GRAY2RGB);
    }

    Range rowsRange, colsRange;
    if (quarter == 1) {
        rowsRange = Range(0, halfHeight);
        colsRange = Range(halfWidth, width);
    } else if (quarter == 2) {
        rowsRange = Range(0, halfHeight);
        colsRange = Range(0, halfWidth);
    } else if (quarter == 3) {
        rowsRange = Range(halfHeight, height);
        colsRange = Range(0, halfWidth);
    } else if (quarter == 4) {
        rowsRange = Range(halfHeight, height);
        colsRange = Range(halfWidth, width);
    }

    image.copyTo(Mat(composedImage, rowsRange, colsRange));
}

void uniteRectangles(vector<Rect>& rectangles) {
    vector<char> useRectangles(rectangles.size(), 1);
    while (true) {
        auto wasChange = false;
        for (int i = 0; i < rectangles.size(); ++i) {
            if (!useRectangles[i]) {
                continue;
            }
            for (int j = i + 1; j < rectangles.size(); ++j) {
                if (useRectangles[j] && ((rectangles[i] & rectangles[j]).area() > 0)) {
                    rectangles[i] |= rectangles[j];
                    useRectangles[j] = false;
                    wasChange = true;
                }
            }
        }
        if (!wasChange) {
            break;
        }
    }
    vector<Rect> resultRectangles;
    for (int i = 0; i < rectangles.size(); ++i) {
        if (useRectangles[i]) {
            resultRectangles.push_back(rectangles[i]);
        }
    }
    rectangles = resultRectangles;
}

void processVideo() {
    VideoCapture capture(0);
    Mat cameraImage, blurredImage, blocksImage, contoursImage;
    Ptr<BackgroundSubtractorKNN> backgroundSubstructor = createBackgroundSubtractorKNN(10, 400, true);
    Mat dilateElement = getStructuringElement(MORPH_RECT, Size(32, 32));
    while(true) {
        capture.read(cameraImage);

        cvtColor(cameraImage, blurredImage, CV_BGR2GRAY);
        GaussianBlur(blurredImage, blurredImage, Size(21, 21), 0, 0);

        backgroundSubstructor->apply(blurredImage, blurredImage);
        threshold(blurredImage, blocksImage, backgroundSubstructor->getShadowValue() + 1, 255, THRESH_BINARY);
        dilate(blocksImage, blocksImage, dilateElement, Point(-1, -1), 2);

        contoursImage = blocksImage.clone();

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(contoursImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        vector<Rect> rectangles;
        contoursImage = Mat::zeros(blocksImage.size(), CV_8UC3);
        for(int i = 0; i < contours.size(); ++i) {
            drawContours(contoursImage, contours, i, Scalar(255, 0, 0), 2, 8, hierarchy, 0);
            const auto rect = boundingRect(contours[i]);
            if (rect.area() >= 50000) {
                rectangles.push_back(rect);
            }
        }

        uniteRectangles(rectangles);

        for (const auto& rect : rectangles) {
            rectangle(cameraImage, rect, Scalar(0, 255, 0), 8);
        }

        Mat composedImage(cameraImage.size(), cameraImage.type());

        addImageToComposition(composedImage, blurredImage, true, 2);
        addImageToComposition(composedImage, blocksImage, true, 1);
        addImageToComposition(composedImage, contoursImage, false, 3);
        addImageToComposition(composedImage, cameraImage, false, 4);

        imshow("Scene", composedImage);

        int c = waitKey(30);
        if( c == 27 || c == 'q' || c == 'Q' )
            break;
    }
    capture.release();
}

int main(int argc, char* argv[])  {
    namedWindow("Scene");

    processVideo();
    destroyAllWindows();

    return EXIT_SUCCESS;
}
