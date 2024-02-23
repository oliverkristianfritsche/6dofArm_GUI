import cv2
import numpy as np


def detect_reds(image,show = True):
    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the range of red color in HSV
    lower_red = np.array([10, 100, 100])
    upper_red = np.array([25, 255, 255])
    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)

    if show:
        # cv2.imshow('Original', image)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', res)

    return res

def detect_light_orange(image, show=True):
    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define a wider range of orange color in HSV to include lighter shades
    # This range might also capture very light reds or yellows that could appear as orange in low-res images
    lower_orange = np.array([5, 50, 50])  # Lower hue, less saturation, and less value to capture lighter shades
    upper_orange = np.array([30, 255, 255])  # Upper hue extended into yellows
    
    # Threshold the HSV image to get only colors within the specified range
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)

    if show:
        # Display the original, mask, and result side by side
        # cv2.imshow('Original', image)
        cv2.imshow('Red Detect Mask', mask)
        cv2.imshow('Red detect Result', res)

    return res


def detect_white_spheroids(image, show=True):
    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert the blurred image to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Define the range of white color in HSV
    lower_white = np.array([0, 0, 200])  # Lower bound for hue, saturation, and high value
    upper_white = np.array([180, 55, 255])  # Upper bound for hue, saturation, and value
    
    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Apply morphological operations to remove small artifacts
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for spheroid shapes
    spheroids = []
    for contour in contours:
        # Find the largest contour by area
        area = cv2.contourArea(contour)
        # Define minimum and maximum area thresholds
        min_area = 50  # Minimum area for a contour to be considered
        max_area = 5000  # Maximum area for a contour to be considered
        if min_area < area < max_area:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            # Calculate the aspect ratio of the bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Calculate the ratio of the contour area to the bounding rectangle area
            bounding_rect_area = w * h
            area_ratio = area / bounding_rect_area if bounding_rect_area > 0 else 0
            
            # Check if the shape is close to a circle using circularity and aspect ratio
            if 0.75 < aspect_ratio < 1.25 and 0.7 < area_ratio < 0.8:
                spheroids.append((center, radius))
    # Draw the detected spheroids on the image
    res = image.copy()
    for center, radius in spheroids:
        cv2.circle(res, center, radius, (0, 255, 0), 2)
    
    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw only the largest contour
        largest_spheroid = cv2.drawContours(image.copy(), [largest_contour], -1, (0, 255, 0), 2)

        res = cv2.drawContours(res, contours, -1, (0, 255, 0), 2)
        if show:
            # Display the original, mask, and result side by side
            # cv2.imshow('Original', image)
            cv2.imshow('White spheroid Mask', mask)
            cv2.imshow('White spheroid Result', res)
            cv2.imshow('Largest White spheroid', largest_spheroid)
        
        #return centerpoint of the largest spheroid
        return largest_contour
    
    return None
