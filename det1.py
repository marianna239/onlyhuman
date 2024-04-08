import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np
import time

ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    target_class_index = 0  # Assuming "person" class has index 0
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")
    num_people = 0  # Counter for the number of people detected


    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    last_capture_time = time.time()
    capture_interval = 5  # Time interval

    while True:
        ret, frame = cap.read()

        if not ret:
            break  # Break the loop if no frame is retrieved

        # Check if it's time to capture a new image
        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            # Process the frame using your model (replace model() with your actual model function)
            results = model(frame, agnostic_nms=True)[0]
            # boxes = result.boxes
            # print(boxes)
            # print(results)
            # print(results.class_indices)
            # exit()
            for i, result in enumerate(results):
                boxes = result.boxes  # Bounding box coordinates
                print(type(boxes))
                print(boxes['cls'])
                if boxes['cls'] == 0:
                    print("Human Detected")
                else:
                    print("Not Human Detected")
                masks = result.masks  # Segmentation masks
                keypoints = result.keypoints  # Pose keypoints
                probs = result.probs  # Class probabilities
                # Display and save the results (assuming show() and save() are correctly implemented)
                # result.show()  # Display the result
            results.save(filename=f'result_{i}.jpg')  # Save the result
            last_capture_time = current_time  # Update the last capture time

        if cv2.waitKey(30) == 27:  # Break the loop if ESC key is pressed
            break

if __name__ == "__main__":
    main()
