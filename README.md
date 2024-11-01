# Lane Line Detection

This project implements a lane line detection system for vehicles using computer vision techniques. The code processes video input to identify and visualize lane markings on the road.

## Features

- **Color Space Conversion**: Utilizes HSV and RGB color spaces for effective lane color selection.
- **Canny Edge Detection**: Applies Canny edge detection to highlight potential lane lines.
- **Region of Interest**: Masks the image to focus on the area of interest where lane lines are expected to be found.
- **Hough Transform**: Detects lines in the masked image using the Hough transform algorithm.
- **Visualization**: Overlays detected lane lines on the original video frames for clear visualization.

## How It Works

The lane detection system processes video frames using a series of image processing techniques:

1. **Input Video**: The system reads video files from the `test_videos` directory.
2. **Preprocessing**: Each frame is converted to grayscale and blurred to reduce noise.
3. **Edge Detection**: Canny edge detection is applied to highlight lane boundaries.
4. **Region of Interest**: The algorithm focuses on a specified area of the frame where lanes are expected.
5. **Line Detection**: The Hough Transform detects lines in the processed image.
6. **Overlay**: Detected lane lines are overlaid on the original frame for visualization.
7. **Output Video**: The processed frames are compiled back into a video and saved in the output directory.

### Test Images and Output Images

The system uses test images located in the `test_images` directory to evaluate the performance of the lane detection algorithm. The output images generated during processing are saved in the `output_images` directory. These output images serve as a valuable tool for debugging, allowing to visually inspect the effectiveness of the lane detection process at each step and make necessary adjustments to improve accuracy.

## Requirements

To run this project, you need the following:

- Python 3.x
- OpenCV
- NumPy
- MoviePy
- Matplotlib (optional, for visualization during development)

# How to Run the Lane Line Detection Project

Follow the steps below to run the Lane Line Detection project on your local machine.

1. Clone the repository
2. Naviagte to the correct directory
3. Add the video files under test_videos and images under test_images ( some are already provided here )
4. Install the required packages if it is missing using pip
5. Run the python file
