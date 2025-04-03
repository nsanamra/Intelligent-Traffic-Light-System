import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time


class IntelligentTrafficSystem:
    def __init__(self, model_path, lane1_points, lane2_points):
        """
        Initialize the traffic system
        Args:
            model_path (str): Path to weights
            lane1_points (list): Coordinates for lane 1
            lane2_points (list): Coordinates for lane 2
        """
        self.model = YOLO('yolo_UA.pt')
        self.lane1_points = np.array(lane1_points, np.int32)
        self.lane2_points = np.array(lane2_points, np.int32)
        self.lane1_mask = None
        self.lane2_mask = None

        # Traffic light states
        self.current_green = 1  # 1 for lane1, 2 for lane2
        self.min_green_time = 30  # minimum green light duration in seconds
        self.max_green_time = 90  # maximum green light duration in seconds
        self.yellow_time = 3  # yellow light duration in seconds
        self.empty_lane_threshold = 5  # time to wait before switching if lane is empty
        self.last_switch_time = time.time()
        self.yellow_start_time = None

        # Vehicle counting
        self.lane1_count = 0
        self.lane2_count = 0

        # Empty lane detection
        self.empty_lane_start_time = None
        self.is_in_yellow_phase = False

    def create_masks(self, frame_shape):
        """Create binary masks for both lanes"""
        self.lane1_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        self.lane2_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(self.lane1_mask, [self.lane1_points], 255)
        cv2.fillPoly(self.lane2_mask, [self.lane2_points], 255)

    def is_in_lane1(self, bbox):
        """Check if vehicle is in lane 1"""
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        return self.lane1_mask[center_y, center_x] == 255

    def is_in_lane2(self, bbox):
        """Check if vehicle is in lane 2"""
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        return self.lane2_mask[center_y, center_x] == 255

    def calculate_green_time(self, count):
        """Calculate green time based on vehicle count"""
        # Base time of 30 seconds + 5 seconds per vehicle, capped at max_green_time
        green_time = min(self.min_green_time + (count * 5), self.max_green_time)
        return green_time

    def should_switch_signal(self):
        """Determine if signal should switch based on vehicle presence"""
        current_time = time.time()

        # If we're in yellow phase, check if it's complete
        if self.is_in_yellow_phase:
            if current_time - self.yellow_start_time >= self.yellow_time:
                self.is_in_yellow_phase = False
                return True
            return False

        # Check if current green lane is empty and other lane has vehicles
        current_lane_count = self.lane1_count if self.current_green == 1 else self.lane2_count
        other_lane_count = self.lane2_count if self.current_green == 1 else self.lane1_count

        if current_lane_count == 0 and other_lane_count > 0:
            # Start counting empty lane time if not already started
            if self.empty_lane_start_time is None:
                self.empty_lane_start_time = current_time
            # Check if empty lane threshold is reached
            elif current_time - self.empty_lane_start_time >= self.empty_lane_threshold:
                # Start yellow phase
                if not self.is_in_yellow_phase:
                    self.is_in_yellow_phase = True
                    self.yellow_start_time = current_time
        else:
            # Reset empty lane timer if vehicles appear or if checking regular timing
            self.empty_lane_start_time = None

            # Check regular timing
            elapsed_time = current_time - self.last_switch_time
            required_time = self.calculate_green_time(current_lane_count)

            if elapsed_time >= required_time:
                # Start yellow phase
                if not self.is_in_yellow_phase:
                    self.is_in_yellow_phase = True
                    self.yellow_start_time = current_time

        return False

    def get_traffic_light_state(self):
        """Determine current traffic light state and timing"""
        current_time = time.time()

        if self.is_in_yellow_phase:
            remaining_yellow = self.yellow_time - (current_time - self.yellow_start_time)
            return self.current_green, "YELLOW", remaining_yellow

        # Calculate remaining time for current phase
        elapsed_time = current_time - self.last_switch_time
        if self.current_green == 1:
            required_time = self.calculate_green_time(self.lane1_count)
        else:
            required_time = self.calculate_green_time(self.lane2_count)

        remaining_time = required_time - elapsed_time

        # If current lane is empty and other lane has vehicles, show empty lane countdown
        current_lane_count = self.lane1_count if self.current_green == 1 else self.lane2_count
        other_lane_count = self.lane2_count if self.current_green == 1 else self.lane1_count

        if current_lane_count == 0 and other_lane_count > 0 and self.empty_lane_start_time is not None:
            remaining_time = self.empty_lane_threshold - (current_time - self.empty_lane_start_time)

        return self.current_green, "GREEN", max(remaining_time, 0)

    def process_frame(self, frame):
        """Process a frame and update traffic light state"""
        if self.lane1_mask is None:
            self.create_masks(frame.shape)

        # Run inference
        results = self.model(frame, conf=0.25)[0]

        # Reset counts
        self.lane1_count = 0
        self.lane2_count = 0

        # Draw lanes
        cv2.polylines(frame, [self.lane1_points], True, (0, 255, 0), 2)
        cv2.polylines(frame, [self.lane2_points], True, (0, 0, 255), 2)

        # Process detections
        for detection in results.boxes.data:
            bbox = detection[:4].int().cpu().numpy()

            if self.is_in_lane1(bbox):
                self.lane1_count += 1
                color = (0, 255, 0)  # Green for lane 1
            elif self.is_in_lane2(bbox):
                self.lane2_count += 1
                color = (0, 0, 255)  # Red for lane 2
            else:
                continue

            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Check if we should switch signals
        if self.should_switch_signal():
            if not self.is_in_yellow_phase:  # Only switch after yellow phase
                self.current_green = 2 if self.current_green == 1 else 1
                self.last_switch_time = time.time()

        # Get traffic light state
        active_lane, light_state, remaining_time = self.get_traffic_light_state()

        # Display information
        cv2.putText(frame, f'Lane 1 Count: {self.lane1_count}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Lane 2 Count: {self.lane2_count}', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        status_text = f'Lane {active_lane}: {light_state} ({int(remaining_time)}s)'
        cv2.putText(frame, status_text, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        return frame


def main():
    # Define your lane regions
    lane1_points = [
        [8, 876],
        [373, 436],
        [479, 278],
        [502, 172],
        [687, 148],
        [794, 144],
        [935, 308],
        [1034, 547],
        [1104, 987],
        [1107, 1046],
        [7, 886],
    ]

    # Define lane 2 points (adjust these for your specific video)
    lane2_points = [
        [1139, 1054],
        [1235, 538],
        [1493, 418],
        [1644, 369],
        [1704, 358],
        [1740, 462],
        [1773, 544],
        [1819, 570],
        [1848, 599],
        [1719, 761],
        [1660, 884],
        [1623, 1059],
        [1136, 1057],
    ]

    # Initialize traffic system
    traffic_system = IntelligentTrafficSystem('yolo_UA.pt', lane1_points, lane2_points)

    # Open video
    cap = cv2.VideoCapture('123.mp4')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed_frame = traffic_system.process_frame(frame)

        # Display frame
        cv2.imshow('Intelligent Traffic System', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()