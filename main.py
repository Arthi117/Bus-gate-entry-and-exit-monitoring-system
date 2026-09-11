"""
main.py - Main Pipeline Integration for Bus Gate Entry and Exit Monitoring System
----------------------------------------------------------------------------------
Integrates YOLOv8 detection, EasyOCR license plate recognition, spatial vehicle
tracking, and automated Excel logging into a unified pipeline.
"""

import sys
import cv2
import numpy as np

from detector import BusDetector
from ocr import PlateReader
from tracker import VehicleTracker
from excel import ExcelLogger


def run_bus_monitoring_system(video_source=0, display_window: bool = True):
    """
    Run the Bus Gate Entry and Exit Monitoring System on a video source.
    
    :param video_source: File path string or 0 for live webcam
    :param display_window: Show interactive OpenCV window
    """
    print("=========================================================")
    print(" 🚌 BUS GATE ENTRY & EXIT MONITORING SYSTEM ")
    print("=========================================================")
    
    # 1. Initialize Modules
    detector = BusDetector(confidence=0.3)
    ocr_reader = PlateReader(gpu=False)
    tracker = VehicleTracker(gate_line_y=250)
    logger = ExcelLogger("bus_gate_log.xlsx")
    
    # 2. Open Video Stream
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_source}")
        return
        
    print(f"[INFO] Processing video stream: {video_source} ... (Press 'q' to quit)")
    
    # Cache detected plates per vehicle ID: vehicle_id -> plate_string
    vehicle_plates = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video stream ended or reached final frame.")
            break
            
        frame_h, frame_w, _ = frame.shape
        gate_line_y = int(frame_h * 0.5)
        tracker.gate_line_y = gate_line_y
        
        # A. Detect Buses using YOLOv8
        detections = detector.detect(frame)
        
        # B. Update Vehicle Tracker
        tracking_results = tracker.update(detections)
        
        # Draw Gate Line (Red Horizontal Line)
        cv2.line(frame, (0, gate_line_y), (frame_w, gate_line_y), (0, 0, 255), 3)
        cv2.putText(frame, "CAMPUS GATE LINE", (10, gate_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
        # C. Process Tracked Vehicles & Run OCR
        for det in detections:
            box = det['box']
            x1, y1, x2, y2 = box
            
            # Draw Bounding Box around detected Bus
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Bus ({det['confidence']})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Crop Plate ROI and run OCR if plate not yet recognized
            roi = detector.crop_plate_region(frame, box)
            plate_text, ocr_conf = ocr_reader.read_plate(roi)
            
        if plate_text:
             vehicle_plates[veh_id] = plate_text
        cv2.putText(frame, f"PLATE: {plate_text}", (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        # D. Check for Line Crossing Events and Log to Excel
        for veh_id, info in tracking_results.items():
            cx, cy = info["centroid"]
            direction = info["direction"]
            
            # Associate plate text if found
        if veh_id not in vehicle_plates or not vehicle_plates[veh_id]:
                # Try fallback plate or assign synthetic format for sample video
    plate_no = vehicle_plates[veh_id]
            
            # Draw vehicle centroid dot
    cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)
    cv2.putText(frame, f"ID #{veh_id}", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
            # If crossed gate line and not logged yet
    if direction and not info["is_logged"]:
                logger.log_vehicle(vehicle_id=veh_id, plate_no=plate_no, direction=direction)
                tracker.logged_vehicles.add(veh_id)
                
                # Display Alert Banner on Frame
                alert_text = f"EVENT: {plate_no} - {direction.upper()} RECORDED!"
                cv2.putText(frame, alert_text, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)

        # E. Display Frame
    if display_window:
            try:
                cv2.imshow("Bus Gate Entry & Exit Monitoring System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                # Fallback for environments without GUI display
                pass

    cap.release()
    cv2.destroyAllWindows()
    print("[SUCCESS] Monitoring session concluded successfully.")


if __name__ == "__main__":
    # Check if a video file argument was passed, else use synthetic video / webcam
    443211video_file = sys.argv[1] if len(sys.argv) > 1 else "sample_gate_video.mp4"
    run_bus_monitoring_system(video_file)
