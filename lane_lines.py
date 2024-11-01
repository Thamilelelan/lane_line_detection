# Importing necessary packages
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
from moviepy.editor import VideoFileClip

def save_image(image, filename):
    cv2.imwrite(filename, image)

def process_images(image_list, process_func, output_dir):
    for i, image in enumerate(image_list):
        processed_image = process_func(image)
        save_image(processed_image, os.path.join(output_dir, f'processed_{i}.jpg'))

def list_images(images, cols=2, rows=5, cmap=None):
    plt.figure(figsize=(10, 11))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        cmap = 'gray' if len(image.shape) == 2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()

# Create output directories if they do not exist
output_image_dir = 'output_images'
output_video_dir = 'output_videos'
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_video_dir, exist_ok=True)

# Reading in the test images
test_images = [plt.imread(img) for img in glob.glob('test_images/*.jpg')]

def gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def HSV_color_selection(image):
    converted_image = convert_hsv(image)
    
    # White color mask
    lower_threshold = np.uint8([0, 0, 200])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    
    # Yellow color mask
    lower_threshold = np.uint8([15, 100, 100])
    upper_threshold = np.uint8([30, 255, 255])
    yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    
    # Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    return masked_image

def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def RGB_color_selection(image):
    # White color mask
    lower_threshold = np.uint8([200, 200, 200])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower_threshold, upper_threshold)
    
    # Yellow color mask
    lower_threshold = np.uint8([180, 180, 0])
    upper_threshold = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower_threshold, upper_threshold)
    
    # Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    return masked_image

# Process and save images
color_selected_images = list(map(HSV_color_selection, test_images))
process_images(color_selected_images, gray_scale, output_image_dir)

def gaussian_smoothing(image, kernel_size=11):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def canny_detector(image, low_threshold=100, high_threshold=200):
    return cv2.Canny(image, low_threshold, high_threshold)

def region_selection(image):
    mask = np.zeros_like(image)   
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.1, rows * 0.95]
    top_left     = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right    = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def hough_transform(image):
    rho = 1
    theta = np.pi / 180
    threshold = 15
    minLineLength = 30
    maxLineGap = 200
    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)

def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    image = np.copy(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

def average_slope_intercept(lines):
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)
    
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if left_weights else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return ((x1, int(y1)), (x2, int(y2)))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

# Video processing functions
def frame_processor(image):
    color_select = HSV_color_selection(image)
    gray = gray_scale(color_select)
    smooth = gaussian_smoothing(gray)
    edges = canny_detector(smooth)
    region = region_selection(edges)
    hough = hough_transform(region)
    result = draw_lane_lines(image, lane_lines(image, hough))
    return result 

def process_video(test_video, output_video):
    input_video = VideoFileClip(os.path.join('test_videos', test_video), audio=False)
    processed = input_video.fl_image(frame_processor)
    processed.write_videofile(os.path.join(output_video_dir, output_video), audio=False)

# Process each video
process_video('solidWhiteRight.mp4', 'solidWhiteRight_output.mp4')
process_video('solidYellowLeft.mp4', 'solidYellowLeft_output.mp4')
process_video('challenge.mp4', 'challenge_output.mp4')


