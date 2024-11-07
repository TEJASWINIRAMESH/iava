
# Ex 3: Implementation of optical flow algorithm using openCV

# %%
import cv2, numpy as np
from google.colab import files
from IPython.display import Image, display

# Upload video
# video_path = next(iter(files.upload()))
video_path="video_.mp4"

# Load video and first frame
cap = cv2.VideoCapture(video_path)
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Initialize tracking points
p0 = cv2.goodFeaturesToTrack(old_gray, 100, 0.3, 7)
mask = np.zeros_like(old_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute Optical Flow
    p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, winSize=(15, 15), maxLevel=2,
                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    good_new, good_old = p1[st == 1], p0[st == 1]

    # Draw tracks
    for (new, old) in zip(good_new, good_old):
        mask = cv2.line(mask, tuple(new.ravel().astype(int)), tuple(old.ravel().astype(int)), (0, 255, 0), 2)
        frame = cv2.circle(frame, tuple(new.ravel().astype(int)), 5, (0, 0, 255), -1)

    img = cv2.add(frame, mask)
    cv2.imwrite('frame.jpg', img)
    display(Image('frame.jpg'))

    old_gray, p0 = frame_gray.copy(), good_new.reshape(-1, 1, 2)

cap.release()

