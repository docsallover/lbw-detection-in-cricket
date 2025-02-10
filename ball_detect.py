import cv2
from cvzone.ColorModule import ColorFinder


def ball_detect(img, color_finder, hsv_values):
    """
    Detects a ball in an image frame based on color.
    Adapted to handle OpenCV-style contours (NumPy arrays of points).

    Args:
        img: The input image frame (BGR format).
        color_finder: An instance of cvzone.ColorFinder for color detection.
        hsv_values: A dictionary containing HSV range for ball color.

    Returns:
        tuple: (img_with_contours, ball_x, ball_y)
               - img_with_contours: Image with contours drawn around the detected ball (or None if no ball detected).
               - ball_x, ball_y: Center coordinates (x, y) of the detected ball (or 0, 0 if no ball detected).
    """
    ball_x = 0
    ball_y = 0
    img_with_contours = None

    if img is None:
        return None, ball_x, ball_y

    imggray_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, mask = color_finder.update(imggray_hsv, hsv_values)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # Use standard cv2.findContours

    # print(f"Type of contours: {type(contours)}") # Keep for debugging - should be NumPy array
    if len(contours) > 0:
        # Find the contour with the largest area using cv2.contourArea
        largest_contour_index = 0
        max_area = 0
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour_index = i
        largest_contour = contours[largest_contour_index]

        # Calculate centroid using moments (OpenCV method for center of contour)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:  # avoid division by zero
            ball_x = int(M["m10"] / M["m00"])
            ball_y = int(M["m01"] / M["m00"])
        else:
            ball_x = 0
            ball_y = 0

        img_with_contours = img.copy()
        cv2.drawContours(
            img_with_contours, [largest_contour], -1, (255, 0, 0), 2
        )  # Draw largest contour
        cv2.circle(
            img_with_contours, (ball_x, ball_y), 5, (255, 0, 0), -1
        )  # Draw center point

    return img_with_contours, ball_x, ball_y


if __name__ == "__main__":
    cap = cv2.VideoCapture(r"lbw.mp4")
    color_finder = ColorFinder(False)
    hsv_vals = {
        "hmin": 10,
        "smin": 44,
        "vmin": 192,
        "hmax": 125,
        "smax": 114,
        "vmax": 255,
    }

    while True:
        frame, img = cap.read()
        if not frame:
            break

        img_contours, x, y = ball_detect(img, color_finder, hsv_vals)

        if img_contours is not None:
            cv2.imshow("Ball Detection", img_contours)
        else:
            cv2.imshow("Ball Detection", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
