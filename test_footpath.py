# test/test.py
import sys
import os
import cv2
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from visual.footpath.footpath_system import FootpathSystem
from core.event_bus import EventBus
from input.video.video_manager import VideoManager
from input.video.frame_preprocessor import FramePreprocessor
from input.video.video_profiles import get_default_video_profiles
from core.contracts.video_contract import VisualFramePayload

class MockVideoManager:
    """Mock video manager that loads image from test folder"""

    def __init__(self, image_path="test/test_image.jpg"):
        self.image_path = image_path
        self.view_profiles = get_default_video_profiles()
        self.preprocessor = FramePreprocessor(self.view_profiles)

    def capture_frame(self):
        """Load frame from test image"""
        # Load image from test folder
        frame_bgr = cv2.imread(self.image_path)
        if frame_bgr is None:
            print(f"Could not load image: {self.image_path}")
            # Create a simple test image if file doesn't exist
            frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame_bgr, (200, 100), (440, 400), (100, 100, 100), -1)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Generate processed views
        processed_views = self.preprocessor.generate_views(frame_rgb)

        # Create payload
        payload = VisualFramePayload.create_from_frame(
            frame_rgb,
            processed_views,
            metadata={
                "source": "test_image",
                "resolution": frame_rgb.shape[:2],
                "stream_type": "test_image",
            }
        )
        return payload

def main():
    """Test footpath system with local image"""
    print("Testing footpath system with local image...")

    # Create event bus
    event_bus = EventBus()

    # Create mock video manager with test image
    video_manager = MockVideoManager("test/9.jpg")

    # Create footpath system
    footpath_system = FootpathSystem(event_bus)

    # Subscribe to results
    def on_footpath_result(payload):
        print("\n=== FOOTPATH ANALYSIS RESULT ===\n")
        print(f"Message: {payload.message}")
        print(f"Risk Level: {payload.risk_level}")
        print(f"Action: {payload.action}")
        print(f"Direction: {payload.direction}")
        if payload.metrics:
            print("Metrics:")
            for key, value in payload.metrics.items():
                print(f"  {key}: {value}")
        print("================================\n")

    event_bus.subscribe("footpath_result", on_footpath_result)

    # Process footpath request
    footpath_system.activate()
    footpath_system.process_footpath_request(video_manager, alert_system_enabled=False, visualize=True)

if __name__ == "__main__":
    main()
