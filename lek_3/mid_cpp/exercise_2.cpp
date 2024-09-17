#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Open the video capture
    VideoCapture cap("slow_traffic_small.mp4");
    if (!cap.isOpened()) {
        cout << "Error opening video stream" << endl;
        return -1;
    }

    // Read the first frame
    Mat frame, gray_frame, prev_frame, sob_x, sob_y, img_dif, flow_map;
    cap >> frame;
    if (frame.empty()) {
        cout << "Error reading first frame" << endl;
        return -1;
    }

    // Print the number of channels in the first frame
    cout << "First frame channels: " << frame.channels() << endl;

    // Check the number of channels in the first frame
    if (frame.channels() == 3) {
        // Convert the first frame to grayscale
        cvtColor(frame, prev_frame, COLOR_BGR2GRAY);
    } else if (frame.channels() == 1) {
        prev_frame = frame.clone();  // If already grayscale
    } else {
        cout << "Unsupported number of channels in the frame: " << frame.channels() << endl;
        return -1;
    }

    // Create flow_map to store the results
    flow_map = Mat::zeros(prev_frame.rows - 3, prev_frame.cols - 3, CV_32FC2);

    // Sobel kernel size
    int ksize = 3;

    while (true) {
        // Read the current frame
        cap >> frame;
        if (frame.empty()) {
            cout << "End of video" << endl;
            break;
        }

        // Print the number of channels in the current frame
        cout << "Current frame channels: " << frame.channels() << endl;

        // Check the number of channels in the current frame
        if (frame.channels() == 3) {
            // Convert the current frame to grayscale
            cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
        } else if (frame.channels() == 1) {
            gray_frame = frame.clone();  // If already grayscale
        } else {
            cout << "Skipping frame with unsupported number of channels: " << frame.channels() << endl;
            continue;
        }

        // Compute frame difference
        absdiff(gray_frame, prev_frame, img_dif);

        // Compute Sobel gradients
        Sobel(gray_frame, sob_x, CV_32F, 1, 0, ksize);
        Sobel(gray_frame, sob_y, CV_32F, 0, 1, ksize);

        // Iterate over the image in 3x3 blocks
        for (int i = 0; i < gray_frame.rows - 3; i++) {
            for (int j = 0; j < gray_frame.cols - 3; j++) {
                // Extract 3x3 blocks for sobel_x, sobel_y, and img_dif
                Mat brik_x = sob_x(Rect(j, i, 3, 3));
                Mat brik_y = sob_y(Rect(j, i, 3, 3));
                Mat brik_dif = img_dif(Rect(j, i, 3, 3));

                // Ensure that the matrices are continuous before copying
                if (!brik_x.isContinuous()) {
                    brik_x = brik_x.clone();
                }
                if (!brik_y.isContinuous()) {
                    brik_y = brik_y.clone();
                }
                if (!brik_dif.isContinuous()) {
                    brik_dif = brik_dif.clone();
                }

                // Flatten the 3x3 blocks manually without reshape
                Mat flat_brik_x = Mat(9, 1, CV_32F);
                Mat flat_brik_y = Mat(9, 1, CV_32F);
                Mat flat_brik_dif = Mat(9, 1, CV_32F);

                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        flat_brik_x.at<float>(k * 3 + l, 0) = brik_x.at<float>(k, l);
                        flat_brik_y.at<float>(k * 3 + l, 0) = brik_y.at<float>(k, l);
                        flat_brik_dif.at<float>(k * 3 + l, 0) = brik_dif.at<float>(k, l);
                    }
                }

                // Create a 9x2 matrix for Sobel gradients (xy)
                Mat xy(9, 2, CV_32F);
                flat_brik_x.copyTo(xy.col(0));
                flat_brik_y.copyTo(xy.col(1));

                // Perform least squares
                Mat x;
                try {
                    solve(xy, flat_brik_dif, x, DECOMP_SVD);  // Use DECOMP_SVD for robust solving
                } catch (const Exception& e) {
                    cout << "Error during solving: " << e.what() << endl;
                    continue;
                }

                // Store the result in the flow_map
                flow_map.at<Vec2f>(i, j)[0] = x.at<float>(0);
                flow_map.at<Vec2f>(i, j)[1] = x.at<float>(1);
            }
        }

        // Show the current frame and flow map
        imshow("Frame", frame);
        imshow("Flow Map", flow_map);

        // Exit if 'q' is pressed
        if (waitKey(1) == 'q') {
            break;
        }

        // Update previous frame
        prev_frame = gray_frame.clone();
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
