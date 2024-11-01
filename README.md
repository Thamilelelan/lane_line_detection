# Lane Line Detection

This project implements a lane line detection system for vehicles using computer vision techniques. The code processes video input to identify and visualize lane markings on the road.

## Features

- **Color Space Conversion**: Utilizes HSV and RGB color spaces for effective lane color selection.
- **Canny Edge Detection**: Applies Canny edge detection to highlight potential lane lines.
- **Region of Interest**: Masks the image to focus on the area of interest where lane lines are expected to be found.
- **Hough Transform**: Detects lines in the masked image using the Hough transform algorithm.
- **Visualization**: Overlays detected lane lines on the original video frames for clear visualization.

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
