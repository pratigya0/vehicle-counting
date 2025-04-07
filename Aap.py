import cv2
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\prati\vehicle-counting\video.mp4")
  # Make sure the path is correct

# Background Subtractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

count_line_pos = 550
min_width_rect = 80
min_height_rect = 80
offset = 6
counter1 = 0

def center_handle(x, y, w, h):
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return cx, cy

detect = []

while True:
    ret, frame1 = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, count_line_pos), (1200, count_line_pos), (255, 127, 0), 3)

    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= min_width_rect and h >= min_height_rect:
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame1, "Vehicle: " + str(counter1), (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (155, 0, 255), 2)

            center = center_handle(x, y, w, h)
            detect.append(center)
            cv2.circle(frame1, center, 4, (0, 0, 255), -1)

            for (cx, cy) in detect:
                if (count_line_pos - offset) < cy < (count_line_pos + offset):
                    counter1 += 1
                    detect.remove((cx, cy))
                    print("Vehicle Counted: " + str(counter1))
                    cv2.line(frame1, (25, count_line_pos), (1200, count_line_pos), (0, 127, 255), 3)

    cv2.putText(frame1, "Total Vehicles: " + str(counter1), (450, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

    cv2.imshow('Vehicle Detection', frame1)

    if cv2.waitKey(1) == 13:  # Press Enter to quit
        break

cap.release()
cv2.destroyAllWindows()
