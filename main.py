from pitch import pitch
from batsman import batsman_detect  # Import the cleaned batsman_detect
from ball_detect import ball_detect  # Import the cleaned ball_detect
import numpy as np
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder

# Initialize ColorFinder and HSV values for ball detection in main.py
mycolorFinder = ColorFinder(False)
hsvVals = {
    "hmin": 10,
    "smin": 44,
    "vmin": 192,
    "hmax": 125,
    "smax": 114,
    "vmax": 255,
}  # Ball HSV values - keep these if they are tuned correctly

# Define TUNED RGB color range for batsman detection - REPLACE these with your TUNED values from batsman.py trackbars!
tuned_rgb_lower = np.array(
    [112, 0, 181]
)  # Example - REPLACE with your tuned lower RGB value
tuned_rgb_upper = np.array(
    [255, 255, 255]
)  # Example - REPLACE with your tuned upper RGB value

# Optional: Tuned Canny thresholds for batsman detection - REPLACE if you tuned these in batsman.py
tuned_canny_threshold1 = 100  # Example - REPLACE if you tuned Canny threshold 1
tuned_canny_threshold2 = 200  # Example - REPLACE if you tuned Canny threshold 2


cap = cv2.VideoCapture(r"lbw.mp4")


def ball_pitch_pad(x, x_prev, prev_x_diff, y, y_prev, prev_y_diff, batLeg):
    """
    Determine whether the ball is a pitch or a pad.

    Args:
        x (int): Current x-coordinate of the ball.
        x_prev (int): Previous x-coordinate of the ball.
        prev_x_diff (int): Previous x-difference between frames.
        y (int): Current y-coordinate of the ball.
        y_prev (int): Previous y-coordinate of the ball.
        prev_y_diff (int): Previous y-difference between frames.
        batLeg (int): y-coordinate of the batsman's leg (used as a threshold).

    Returns:
        tuple: A tuple of (motion_type, x_diff, y_diff).
            - motion_type (str): One of "Motion", "Pad", or "Pitch", indicating the type of motion detected.
            - x_diff (int): The difference in x-coordinates between the current frame and the previous frame.
            - y_diff (int): The difference in y-coordinates between the current frame and the previous frame.
    """
    if x_prev == 0 and y_prev == 0:  # Handle first frame
        return "Motion", 0, 0

    if (
        abs(x - x_prev) > 3 * abs(prev_x_diff) and abs(prev_x_diff) > 0
    ):  # Avoid division by zero if prev_x_diff is zero initially
        if y < batLeg:
            print("Condition 1 triggered: Sudden X motion and below batLeg")
            return "Pad", x - x_prev, y - y_prev

    if y - y_prev < 0 and prev_y_diff > 0:  # Ball moving downwards after upward motion
        print("Condition 2 triggered: Y motion change")
        if y < batLeg:
            return "Pad", x - x_prev, y - y_prev
        else:
            return "Pitch", x - x_prev, y - y_prev

    return "Motion", x - x_prev, y - y_prev


x = 0
y = 0
batLeg = 0
x_prev = 0
y_prev = 0
prev_x_diff = 0
prev_y_diff = 0
lbw_detected = False  # Flag to track if LBW condition is met in any frame

while True:
    x_prev = x
    y_prev = y
    frame, img = cap.read()

    if not frame:
        print("Video ended or error reading frame. Exiting.")
        break

    ballImg = img.copy()
    pitchImg = img.copy()
    batsmanImg = img.copy()

    # Use the cleaned ball_detect function, passing colorFinder and hsvVals
    ballContour, x, y = ball_detect(img, mycolorFinder, hsvVals)
    all = ballContour.copy() if ballContour is not None else img.copy()

    # Use the cleaned batsman_detect function, passing TUNED RGB range and optional Canny thresholds
    batsmanContours = batsman_detect(
        img,
        tuned_rgb_lower,
        tuned_rgb_upper,
        tuned_canny_threshold1,
        tuned_canny_threshold2,
    )
    # Note: batsman_detect now returns just contours, not an image with contours drawn

    pitchContour = pitch(
        img
    )  # Assuming pitch function is in pitch.py and works correctly

    # Drawing Pitch Contours
    for cnt in pitchContour:
        if cv2.contourArea(cnt) > 50000:
            cv2.drawContours(pitchImg, cnt, -1, (0, 255, 0), 10)
            if all is not None:
                cv2.drawContours(all, cnt, -1, (0, 255, 0), 10)

    # Initialize batLeg to a large value initially, updated if batsman is detected
    current_batLeg = 10000  # Initialize to a very large value

    # Drawing Batsman Contours - using the contours returned from cleaned batsman_detect
    for cnt in batsmanContours:  # Iterate through batsmanContours (plural)
        if cv2.contourArea(cnt) > 5000:  # Area filtering for batsman contours
            if (
                min(cnt[:, :, 1]) < y and y != 0
            ):  # Check if batsman contour is above ball AND ball is detected
                batLeg_candidate = max(
                    cnt[:, :, 1]
                )  # Calculate candidate batLeg from current contour
                if (
                    batLeg_candidate < current_batLeg
                ):  # Take the minimum batLeg value found across all batsman contours in frame
                    current_batLeg = batLeg_candidate
                cv2.drawContours(batsmanImg, cnt, -1, (0, 0, 255), 10)
                if all is not None:
                    cv2.drawContours(all, cnt, -1, (0, 0, 255), 10)
    batLeg = (
        current_batLeg  # Update batLeg with the minimum batLeg found in current frame
    )

    output = ball_pitch_pad(x, x_prev, prev_x_diff, y, y_prev, prev_y_diff, batLeg)
    prev_x_diff = output[1]
    prev_y_diff = output[2]
    motion_type = output[0]
    # print(f"Motion Type: {motion_type}, Ball (x,y): ({x},{y}), batLeg: {batLeg}")

    if motion_type == "Pad":
        lbw_detected = True  # Set LBW flag if "Pad" is detected

    imgStackList = [
        ballContour if ballContour is not None else img,
        pitchImg,
        batsmanImg,
        all if all is not None else img,
    ]
    imgStack = cvzone.stackImages(imgStackList, 2, 0.5)
    cv2.imshow("stack", imgStack)

    if cv2.waitKey(1) == ord("k"):
        while True:
            if cv2.waitKey(1) == ord("k"):
                break

    elif cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if lbw_detected:
    print("\nPotential LBW Detected!")  # Print LBW detected message at the end
else:
    print("\nNo LBW Detected (based on Pad condition).")  # Print No LBW message

print(
    "Remember to tune batsman detection RGB values in batsman.py for better accuracy."
)
