
# Parking Spot Tracker

The Parking Spot Tracker is designed to detect and monitor available parking spots using a live stream feed. It leverages the YOLOv8 object detection model for its operations and can notify users about parking availability in real-time.


## Demo

![Demonstration of Parking Spot Tracker in action](URL_of_GIF)


## Features

* Utilizes YOLOv8 for robust object detection and tracking.

* Monitors parking spots continuously from a given live stream.

* Sends real-time notifications about parking spot status using Twilio.

* Visualizes object tracking on the display frame.
## Prerequisites

*  YOLOv8: The project uses [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection.

* [Python 3.x](https://www.python.org/downloads/)

* [OpenCV](https://opencv.org/)

* YOLOv8 pre-trained model

* [Twilio API](https://www.twilio.com/docs/quickstart) for notifications
## Installation

1) Clone the repository:

```bash
  git clone <repository_url>
```
2) Navigate to the project directory:

```bash
  cd parking_spot_tracker
```

3) Install the required packages:

```bash
  pip install -r requirements.txt
```

## Configuration

Before you start, configure the config.json file:

* MODEL_PATH: Path to the YOLOv8 pre-trained model. See [YOLOv8](https://github.com/ultralytics/ultralytics) for more details. 

* MONITOR_NUMBER: Screen monitor number for capturing live feed.

* CONFIDENCE: Confidence threshold for YOLOv8 detections. Valid values are between 0.0 and 1.0.  See [YOLOv8](https://github.com/ultralytics/ultralytics) for more details.

* MOVE_THRESHOLD: Threshold distance to determine if a car has moved.

* PARKED_FRAMES_THRESHOLD: Number of frames a car should remain stationary to be marked as parked.

* MISSED_FRAMES_THRESHOLD: Number of frames a parked car can remain undetected before it's considered to have left.

* RESET_TRACKING_FRAMES: Number of frames to reset tracking points.

* ACCOUNT_SID: Your Twilio account SID.

* AUTH_TOKEN: Your Twilio authentication token.

* API_URL: Twilio API url




## Usage

1) Run the script:

```bash
  python src/parking_spot_tracker.py
```

2) To exit the live stream and tracking, press the q key on the monitoring window.
    
## Design Considerations

During the development of the Parking Spot Tracker, certain constraints and considerations influenced the design and implementation:

- **Camera Stream Access**: Ideally, accessing the RTSP stream of the camera directly would have been the most efficient approach. However, due to administrative restrictions on the cameras, this was not feasible. As a workaround, the live feed was captured from a second monitor.
## License

This project is licensed under the MIT License. Refer to the LICENSE file for more details.
## Acknowledgments

* YOLOv8: For providing state-of-the-art object detection capabilities.

* Twilio: For notification system.
