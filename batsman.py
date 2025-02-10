import cv2
import cvzone
import numpy as np


def batsman_detect(img, rgb_lower, rgb_upper, canny_threshold1=100, canny_threshold2=200):
    """
    Detects a batsman in an image frame using color-based filtering and edge detection.

    Args:
        img: The input image frame (BGR format).
        rgb_lower: NumPy array defining the lower bound of the RGB color range for batsman. e.g., np.array([112, 0, 181])
        rgb_upper: NumPy array defining the upper bound of the RGB color range for batsman. e.g., np.array([255, 255, 255])
        canny_threshold1: Lower threshold for Canny edge detection.
        canny_threshold2: Upper threshold for Canny edge detection.

    Returns:
        contours: A list of contours detected in the color-masked and edge-processed image,
                  presumed to be the batsman. Returns an empty list if no contours are found.
    """
    img_gray_rgb = cv2.cvtColor(
        img, cv2.COLOR_BGR2RGB
    )  # Convert to RGB for color masking
    img_blur = cv2.GaussianBlur(
        img_gray_rgb, (5, 5), 1
    )  # Gaussian blur for noise reduction

    # Edge Detection (Canny) - you can tune canny_threshold1 and canny_threshold2
    img_canny = cv2.Canny(img_blur, canny_threshold1, canny_threshold2)

    # Morphological Operations (Opening - Dilation followed by Erosion) - Consider experimenting with Closing (Erosion then Dilation)
    kernel = np.ones((5, 5))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)  # Dilate to thicken edges
    img_threshold = cv2.erode(
        img_dilate, kernel, iterations=2
    )  # Erode to remove noise and thin edges (Opening)

    # Color Masking in RGB color space
    mask = cv2.inRange(img_gray_rgb, rgb_lower, rgb_upper)

    # Find contours in the color mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


if __name__ == "__main__":
    cap = cv2.VideoCapture(r"lbw.mp4")  # Adjust path if needed

    # Default RGB color range - you should tune these using the trackbars below
    default_rgb_lower = np.array([112, 0, 181])
    default_rgb_upper = np.array([255, 255, 255])

    # Canny thresholds - you can also tune these via trackbars if needed
    default_canny_threshold1 = 100
    default_canny_threshold2 = 200

    def empty(a):  # Dummy function for trackbar
        pass

    # Create a window for trackbars to tune RGB color range and Canny thresholds
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow(
        "Trackbars", 640, 480
    )  # Increased window height to fit more trackbars
    cv2.createTrackbar("R Min", "Trackbars", default_rgb_lower[0], 255, empty)
    cv2.createTrackbar("G Min", "Trackbars", default_rgb_lower[1], 255, empty)
    cv2.createTrackbar("B Min", "Trackbars", default_rgb_lower[2], 255, empty)
    cv2.createTrackbar("R Max", "Trackbars", default_rgb_upper[0], 255, empty)
    cv2.createTrackbar("G Max", "Trackbars", default_rgb_upper[1], 255, empty)
    cv2.createTrackbar("B Max", "Trackbars", default_rgb_upper[2], 255, empty)
    cv2.createTrackbar(
        "Canny Thresh 1", "Trackbars", default_canny_threshold1, 255, empty
    )  # Canny threshold 1
    cv2.createTrackbar(
        "Canny Thresh 2", "Trackbars", default_canny_threshold2, 255, empty
    )  # Canny threshold 2

    while True:
        frame, img = cap.read()
        if not frame:
            break

        # Get trackbar positions for RGB color range
        rgb_lower = np.array(
            [
                cv2.getTrackbarPos("R Min", "Trackbars"),
                cv2.getTrackbarPos("G Min", "Trackbars"),
                cv2.getTrackbarPos("B Min", "Trackbars"),
            ]
        )
        rgb_upper = np.array(
            [
                cv2.getTrackbarPos("R Max", "Trackbars"),
                cv2.getTrackbarPos("G Max", "Trackbars"),
                cv2.getTrackbarPos("B Max", "Trackbars"),
            ]
        )

        # Get trackbar positions for Canny thresholds
        canny_threshold1 = cv2.getTrackbarPos("Canny Thresh 1", "Trackbars")
        canny_threshold2 = cv2.getTrackbarPos("Canny Thresh 2", "Trackbars")

        # Detect batsman using the function with tunable parameters
        batsman_contours = batsman_detect(
            img, rgb_lower, rgb_upper, canny_threshold1, canny_threshold2
        )

        img_contours = img.copy()  # Copy image to draw contours on
        for cnt in batsman_contours:
            if (
                cv2.contourArea(cnt) > 5000
            ):  # Area filtering - you can adjust this threshold in main.py if needed
                cv2.drawContours(
                    img_contours, cnt, -1, (0, 255, 0), 2
                )  # Draw batsman contours in green

        img_mask = cv2.inRange(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB), rgb_lower, rgb_upper
        )  # Show the mask for tuning

        img_stack = cvzone.stackImages(
            [img, img_mask, img_contours], 3, 0.5
        )  # Stack original, mask, and contours
        cv2.imshow("Batsman Detection Tuning", img_stack)  # Combined window for tuning

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):  # Press 's' to save current RGB values to console
            print("Saved RGB lower:", rgb_lower)
            print("Saved RGB upper:", rgb_upper)
            print("Saved Canny Threshold 1:", canny_threshold1)
            print("Saved Canny Threshold 2:", canny_threshold2)
            print(
                "--- Copy these values to your main.py or default_rgb_lower/upper in batsman.py ---"
            )

    cap.release()
    cv2.destroyAllWindows()
