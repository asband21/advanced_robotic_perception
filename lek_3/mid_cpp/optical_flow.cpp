#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Open the video file
    VideoCapture cap("slow_traffic_small.mp4");
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    // Read the first frame
    Mat frame, frame_gray;
    if (!cap.read(frame)) {
        cout << "Cannot read the video file" << endl;
        return -1;
    }

    // Convert to grayscale
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    int height = frame_gray.rows;
    int width = frame_gray.cols;

    // Initialize flow_map as a 3-channel float image
    Mat flow_map = Mat::zeros(height - 3, width - 3, CV_32FC3);

    int iii = 1;
    while (true) {
        cout << "Frame: " << iii << endl;
        iii++;

        // Store previous frame
        Mat prev_gray = frame_gray.clone();

        // Read the next frame
        if (!cap.read(frame)) {
            cout << "Can't receive frame (stream end?). Exiting ..." << endl;
            break;
        }

        // Convert current frame to grayscale
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        // Convert images to float32 for precise computation
        Mat prev_gray_f, frame_gray_f;
        prev_gray.convertTo(prev_gray_f, CV_32F);
        frame_gray.convertTo(frame_gray_f, CV_32F);

        // Compute the difference between frames
        Mat img_dif = frame_gray_f - prev_gray_f;

        // Compute Sobel gradients
        Mat sob_x, sob_y;
        Sobel(frame_gray_f, sob_x, CV_32F, 1, 0, 1);
        Sobel(frame_gray_f, sob_y, CV_32F, 0, 1, 1);

        // Iterate over the image in 3x3 blocks
        for (int i = 0; i < height - 3; i++) {
            for (int j = 0; j < width - 3; j++) {
                // Extract 3x3 blocks from gradients and difference image
                Mat brik_x = sob_x(Rect(j, i, 3, 3)).clone().reshape(1, 9);
                Mat brik_y = sob_y(Rect(j, i, 3, 3)).clone().reshape(1, 9);
                Mat brik_dif = img_dif(Rect(j, i, 3, 3)).clone().reshape(1, 9);

                // Stack brik_x and brik_y to form a 9x2 matrix
                Mat xy;
                hconcat(brik_x.t(), brik_y.t(), xy);

                // Solve the least squares problem: xy * flow = brik_dif.t()
                Mat flow;
                bool success = solve(xy, brik_dif.t(), flow, DECOMP_SVD);
                if (!success || flow.rows < 2) {
                    continue;
                }

                // Store the flow vectors in flow_map
                flow_map.at<Vec3f>(i, j)[0] = flow.at<float>(0);
                flow_map.at<Vec3f>(i, j)[1] = flow.at<float>(1);
                flow_map.at<Vec3f>(i, j)[2] = 0; // Third channel is zero
            }
        }

        // Compute the magnitude of the flow vectors for visualization
        Mat flow_parts[3];
        split(flow_map, flow_parts);
	//Mat magnitude, angle;
        //magnitude(flow_parts[0], flow_parts[1], magnitude, angle);
	Mat magnitude, angle;
	cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle);

        // Normalize and convert to 8-bit image for display
	Mat magnitude_display;
        normalize(magnitude, magnitude_display, 0, 255, NORM_MINMAX);
        magnitude_display.convertTo(magnitude_display, CV_8U);

        // Prepare Sobel images for display
        Mat sob_x_display, sob_y_display;
        sob_x.convertTo(sob_x_display, CV_8U);
        sob_y.convertTo(sob_y_display, CV_8U);

        // Display the images
        imshow("Frame", frame);
        imshow("Flow Magnitude", magnitude_display);
        imshow("Sobel X", sob_x_display);
        imshow("Sobel Y", sob_y_display);

        // Exit if 'q' is pressed
        if (waitKey(1) == 'q') {
            break;
        }
    }

    // Release resources
    cap.release();
    destroyAllWindows();

    return 0;
}
