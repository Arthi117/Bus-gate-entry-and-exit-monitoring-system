"""
create_sample_video.py - Synthetic Test Video Generator for Bus Monitoring System
----------------------------------------------------------------------------------
Generates a realistic test video ('sample_gate_video.mp4') featuring a moving yellow
college bus with a clear license plate ('TN 38 CB 2026') crossing a gate threshold.
Useful for zero-hardware automated testing.
"""

import cv2
import numpy as np


def generate_sample_video(filename: str = "sample_gate_video.mp4", duration_sec: int = 6, fps: int = 30):
    width, height = 640, 480
    total_frames = duration_sec * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    print(f"[INFO] Generating synthetic test video: {filename} ({total_frames} frames)...")
    
    # Bus initial position (Moving from top Y=50 down past gate line Y=250)
    bus_x = 220
    start_y = 40
    end_y = 300
    
    for frame_idx in range(total_frames):
        # Create road background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 100 # Dark grey asphalt
        
        # Draw Gate Line (Horizontal Red Line at Y=250)
        cv2.line(frame, (0, 250), (width, 250), (0, 0, 255), 3)
        cv2.putText(frame, "CAMPUS GATE ENTRY THRESHOLD", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Calculate current bus Y coordinate
        t = frame_idx / float(total_frames)
        current_y = int(start_y + t * (end_y - start_y))
        
        # Draw Bus Body (Yellow College Bus Box)
        bus_w, bus_h = 200, 120
        cv2.rectangle(frame, (bus_x, current_y), (bus_x + bus_w, current_y + bus_h), (0, 215, 255), -1) # Yellow Fill
        cv2.rectangle(frame, (bus_x, current_y), (bus_x + bus_w, current_y + bus_h), (0, 0, 0), 3)       # Black Border
        
        # Draw Bus Windows & Roof Banner
        cv2.rectangle(frame, (bus_x + 15, current_y + 15), (bus_x + bus_w - 15, current_y + 45), (200, 200, 200), -1)
        cv2.putText(frame, "COLLEGE BUS", (bus_x + 35, current_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw License Plate Box (White Box at bottom of bus with text 'TN 38 CB 2026')
        plate_x1, plate_y1 = bus_x + 30, current_y + 75
        plate_x2, plate_y2 = bus_x + 170, current_y + 110
        cv2.rectangle(frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (255, 255, 255), -1) # White Plate Background
        cv2.rectangle(frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 0, 0), 2)       # Black Border
        cv2.putText(frame, "TN 38 CB 2026", (plate_x1 + 10, plate_y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        out.write(frame)

    out.release()
    print(f"[SUCCESS] Sample test video saved successfully: {filename}")


if __name__ == "__main__":
    generate_sample_video()
