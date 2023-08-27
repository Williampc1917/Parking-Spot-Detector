from typing import List
import cv2
import numpy as np
import json
import logging
from mss import mss
from ultralytics import YOLO
from collections import defaultdict
from twilio.rest import Client

logging.basicConfig(level=logging.INFO)


def load_config(config_path="config.json"):
    """
    Load the configuration for the ParkingSpotTracker from a JSON file.

    This function reads the provided JSON configuration file and extracts
    the 'Parking_Yolov8_Config' key, which contains the necessary
    configuration for the ParkingSpotTracker.

    Parameters:
        config_path (str, optional): The path to the JSON configuration file.
            Defaults to "config.json".

    Returns:
        dict: A dictionary containing the configuration settings for the
        ParkingSpotTracker.

    Raises:
        FileNotFoundError: If the provided config_path doesn't exist.
        KeyError: If the 'Parking_Yolov8_Config' key is not present in the
        JSON file.
    """
    with open(config_path, "r") as file:
        data = json.load(file)
        return data["parking_spot_tracker_Config"]


class ParkingSpotTracker:

    def __init__(self, config_data):
        """
           Initializes the ParkingSpotTracker with necessary configurations.

           This constructor sets up the parking spot tracker based on a given
           configuration. It ensures all required keys are present in the config,
           then loads the YOLO model for object detection and tracking, sets up
           thresholds for movement, parking, and missed frame detections, and
           initializes other necessary parameters.

           Parameters:
               config_data (dict): A dictionary containing configuration parameters.

           Raises:
               ValueError: If any of the required keys are missing from the config.
           """

        required_keys = ["MODEL_PATH",
                         "MOVE_THRESHOLD",
                         "PARKED_FRAMES_THRESHOLD",
                         "MISSED_FRAMES_THRESHOLD",
                         "RESET_TRACKING_FRAMES",
                         "ACCOUNT_SID",
                         "AUTH_TOKEN",
                         "TWILIO_FROM_NUMBER",
                         "TWILIO_TO_NUMBER"
                         ]

        # Validate that all required keys are present in the config
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Key {key} not found in config!")

        # initialize the properties
        self.MOVE_THRESHOLD = config["MOVE_THRESHOLD"]
        self.PARKED_FRAMES_THRESHOLD = config["PARKED_FRAMES_THRESHOLD"]
        self.MISSED_FRAMES_THRESHOLD = config["MISSED_FRAMES_THRESHOLD"]
        self.ACCOUNT_SID = config["ACCOUNT_SID"]
        self.AUTH_TOKEN = config["AUTH_TOKEN"]
        self.model = YOLO(config["MODEL_PATH"])
        self.monitor = mss().monitors[config["MONITOR_NUMBER"]]
        self.confidence = config["CONFIDENCE"]
        self.TWILIO_FROM_NUMBER = config["TWILIO_FROM_NUMBER"]
        self.TWILIO_TO_NUMBER = config["TWILIO_TO_NUMBER"]
        self.track_history = defaultdict(dict)
        self.frame_count = 0
        self.config = config_data

        # Initialize the Twilio client
        self.client = Client(self.ACCOUNT_SID, self.AUTH_TOKEN)

    def capture_screen(self):
        """
            Captures the screen content for the specified monitor.

            Returns:
                numpy.ndarray: A frame capturing the current screen content.
            """
        with mss() as sct:
            screenshot = sct.grab(self.monitor)
            # noinspection PyTypeChecker
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        return frame

    def process_frame(self, frame):
        """
            Processes the captured frame using YOLO and extracts necessary information.

            Parameters:
                frame (numpy.ndarray): Input frame captured from the screen.

            Returns:
                tuple:
                    - boxes (numpy.ndarray): Bounding box coordinates of detected objects.
                    - track_ids (list[int]): Unique tracking IDs for detected objects.
                    - annotated_frame (numpy.ndarray): Frame with detected objects annotated.
            """

        results = self.model.track(frame, persist=True, conf=self.confidence, classes=[2, 3, 7, 5], retina_masks=True)
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        annotated_frame = results[0].plot()
        return boxes, track_ids, annotated_frame

    def visualize_and_track_objects(self, boxes: np.ndarray, track_ids: List[int], annotated_frame):
        """
           Visualize tracking information on the frame and monitor parking status.
           Periodically resets tracking points and manages object tracking history.

           Parameters:
               boxes (numpy.ndarray): Bounding box coordinates of detected objects.
               track_ids (list[int]): Unique tracking IDs for detected objects.
               annotated_frame (numpy.ndarray): Frame with detected objects annotated.

           Returns:
               numpy.ndarray: Frame with updated annotations based on tracked objects and parking status.
           """
        self.frame_count += 1
        reset_tracking_frames = self.config["RESET_TRACKING_FRAMES"]

        if self.frame_count % reset_tracking_frames == 0:
            for track_id in self.track_history:
                if self.track_history[track_id]["positions"]:
                    # Store the last position before clearing
                    last_pos = self.track_history[track_id]["positions"][-1]
                    self.track_history[track_id]["positions"].clear()
                    # Add the last position back, so it can be used for movement assessment
                    self.track_history[track_id]["positions"].append(last_pos)
            logging.info("Tracking points reset after %s frames.", reset_tracking_frames)

        currently_detected = set(track_ids)
        for box, track_id in zip(boxes, track_ids):
            self._update_track_history(box, track_id, annotated_frame)
        return self._handle_parked_cars(annotated_frame, currently_detected)

    def _update_track_history(self, box, track_id, annotated_frame):
        """
            Update the track history for a specific car based on its bounding box and track_id.
            Visualizes the tracking information on the annotated frame, and monitors
            the movement and parking status of the car.

            Parameters:
                box (tuple[float, float, float, float]): Bounding box coordinates (x, y, width, height) of the detected car.
                track_id (int): Unique tracking ID for the detected car.
                annotated_frame (numpy.ndarray): Frame with detected objects annotated.

            Returns:
                None: The method directly modifies the `annotated_frame` and updates the internal `track_history` attribute.
            """
        x, y, w, h = box

        # Initialize tracking history for a new car
        if track_id not in self.track_history:
            self.track_history[track_id] = {
                "positions": [],
                "stationary_count": 0,
                "is_parked": False,
                "missed_frames": 0
            }
        else:
            # Reset the missed_frames counter since the car is detected in this frame
            self.track_history[track_id]["missed_frames"] = 0

        history = self.track_history[track_id]
        current_pos = np.array([float(x), float(y)])

        # Check if the car has moved since the last frame
        if history["positions"]:
            last_pos = np.array(history["positions"][-1])
            distance_moved = np.linalg.norm(current_pos - last_pos)
            if distance_moved < self.MOVE_THRESHOLD:
                history["stationary_count"] += 1
            else:
                history["stationary_count"] = 0

        history["positions"].append((float(x), float(y)))

        # Check for parked status
        if history["stationary_count"] > self.PARKED_FRAMES_THRESHOLD and not history["is_parked"]:
            cv2.putText(annotated_frame, "Parked", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            history["is_parked"] = True
            logging.info(f"Car with ID {track_id} is now parked.")

        # Draw the tracking lines
        points = np.hstack(history["positions"]).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

    def _handle_parked_cars(self, annotated_frame, currently_detected):
        """
            Handles cars that were previously marked as parked but are no longer detected.
            If a parked car is not detected for a specified number of frames, it's
            assumed that the car has left and a notification is sent. This method also
            updates the tracking history to remove cars that have left the scene.

            Parameters:
                annotated_frame (numpy.ndarray): Frame with detected objects annotated.
                currently_detected (set[int]): Set of track_ids for cars currently detected in the frame.

            Returns:
                numpy.ndarray: The same annotated frame passed as an argument, possibly with updates based on the method's actions.
            """
        for track_id in list(self.track_history.keys()):
            if track_id not in currently_detected and self.track_history[track_id]["is_parked"]:
                self.track_history[track_id]["missed_frames"] += 1
                if self.track_history[track_id]["missed_frames"] > self.MISSED_FRAMES_THRESHOLD:
                    del self.track_history[track_id]
                    logging.info(f"Car with ID {track_id} has left the parking spot. Spot is now available!")
                    self.send_notification(f"Car with ID {track_id} has left the parking spot. Spot is now available!")
        return annotated_frame

    def send_notification(self, message):
        """
                Sends a notification message using the Twilio API.

                Utilizes the Twilio API to send a notification message. If the notification
                is sent successfully, a success message is printed to the console. In case of
                an error, a relevant error message is displayed.

                Parameters:
                    message (str): The content of the notification to be sent.

                Returns:
                    None
                """
        try:
            twilio_message = self.client.messages.create(
                from_=self.TWILIO_FROM_NUMBER,
                body=message,
                to=self.TWILIO_TO_NUMBER
            )

            if twilio_message.sid:
                logging.info("Notification sent successfully!")
            else:
                logging.warning("Failed to send notification.")
        except Exception as e:
            logging.error(f"Error sending notification: {e}")

    def monitor_parking_spots(self):
        """
           Continuously monitors parking spots, processes frames, and visualizes the results.

           This method captures the screen to get a frame, processes the frame using the
           YOLO model to detect and track objects, and visualizes the tracking results on
           the frame. The processed frame is displayed in a window named "YOLOv8 Inference with Tracking".
           The loop continues indefinitely until the user presses the 'q' key.

           Returns:
               None
           """
        while True:
            frame = self.capture_screen()
            boxes, track_ids, annotated_frame = self.process_frame(frame)
            final_frame = self.visualize_and_track_objects(boxes, track_ids, annotated_frame)
            cv2.imshow("YOLOv8 Inference with Tracking", final_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    """
        Entry point for the script. This block initializes the parking spot tracker
        and begins the monitoring process using the configuration provided in the JSON file.
        """
    config = load_config()
    parking_tracker = ParkingSpotTracker(config)
    parking_tracker.monitor_parking_spots()
