import cv2
import numpy as np


class RegionSelector:
    def __init__(self):
        self.points = []
        self.current_frame = None
        self.window_name = "Region Selector"

    def click_event(self, event, x, y, flags, params):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point
            self.points.append([x, y])

            # Draw point
            cv2.circle(self.current_frame, (x, y), 5, (0, 255, 0), -1)

            # Draw polygon if we have at least 2 points
            if len(self.points) >= 2:
                cv2.polylines(self.current_frame,
                              [np.array(self.points)],
                              False, (0, 255, 0), 2)

            cv2.imshow(self.window_name, self.current_frame)

    def select_region(self, video_path):
        """
        Select region from the first frame of the video

        Args:
            video_path (str): Path to the video file

        Returns:
            points (list): List of selected coordinates
            frame_shape (tuple): Shape of the video frame
        """
        # Open video
        cap = cv2.VideoCapture('123.mp4')
        ret, frame = cap.read()
        if not ret:
            print("Error reading video")
            return None, None

        # Store original frame and make a copy for drawing
        original_frame = frame.copy()
        self.current_frame = frame.copy()

        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.click_event)

        print("Click points to define your region. Press:")
        print("'c' - Close the polygon")
        print("'r' - Reset points")
        print("'q' - Quit without saving")

        while True:
            cv2.imshow(self.window_name, self.current_frame)
            key = cv2.waitKey(1) & 0xFF

            # Close polygon
            if key == ord('c') and len(self.points) >= 3:
                # Draw final line to close polygon
                pts = np.array(self.points + [self.points[0]])
                cv2.polylines(self.current_frame, [pts], True, (0, 255, 0), 2)
                cv2.imshow(self.window_name, self.current_frame)
                cv2.waitKey(1)
                break

            # Reset points
            elif key == ord('r'):
                self.points = []
                self.current_frame = original_frame.copy()
                cv2.imshow(self.window_name, self.current_frame)

            # Quit
            elif key == ord('q'):
                self.points = []
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()

        return self.points, frame.shape


def save_coordinates(points, filename="region_coordinates.txt"):
    """Save coordinates to a file"""
    if points:
        with open(filename, 'w') as f:
            for point in points:
                f.write(f"{point[0]},{point[1]}\n")
        print(f"Coordinates saved to {filename}")
        print("\nCoordinates for direct use in code:")
        print("region_points = [")
        for point in points:
            print(f"    [{point[0]}, {point[1]}],")
        print("]")


def main():
    # Initialize region selector
    selector = RegionSelector()

    # Select region
    video_path = "path/to/your/video.mp4"  # Replace with your video path
    points, frame_shape = selector.select_region(video_path)

    # Save coordinates if points were selected
    if points:
        save_coordinates(points)
        print(f"\nFrame shape: {frame_shape}")


if __name__ == "__main__":
    main()