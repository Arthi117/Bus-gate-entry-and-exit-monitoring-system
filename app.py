"""
app.py - Campus Bus Gate Entry & Exit Management System
------------------------------------------------------
Flask web interface for live gate camera monitoring, automated vehicle tracking,
license plate identification, activity log search, and Excel reporting.
"""

import os
import time
import cv2
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, Response, jsonify, send_file
)
from werkzeug.utils import secure_filename

from detector import BusDetector
from ocr import PlateReader
from tracker import VehicleTracker
from excel import ExcelLogger

app = Flask(__name__)
app.secret_key = "campus_bus_gate_security_key_2026"

# Directory configuration for video uploads and exports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

os.makedirs("models", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Max 500 MB video file upload limit

# Initialize Excel Data Logger
EXCEL_FILE = os.path.join(BASE_DIR, "bus_gate_log.xlsx")
logger = ExcelLogger(EXCEL_FILE)

# Lazily initialized model instances
detector = None
ocr_reader = None
tracker = None

def get_pipeline():
    """Lazily load computer vision detector, OCR reader, and centroid tracker."""
    global detector, ocr_reader, tracker
    if detector is None:
        print("[INFO] Loading vehicle detector...")
        detector = BusDetector(confidence=0.35)
    if ocr_reader is None:
        print("[INFO] Loading license plate reader...")
        ocr_reader = PlateReader(gpu=False)
    if tracker is None:
        print("[INFO] Initializing gate centroid tracker...")
        tracker = VehicleTracker(gate_line_y=250)
    return detector, ocr_reader, tracker


# Global active video source state: 'sample_file', 'webcam', or absolute path to uploaded video file
current_video_source = "sample_file"

def is_allowed_file(filename):
    """Check if uploaded file has a valid video extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_video_capture():
    """Resolve OpenCV VideoCapture stream based on current source selection."""
    global current_video_source
    
    if current_video_source == "webcam":
        return cv2.VideoCapture(0)
    elif os.path.isfile(current_video_source):
        # User uploaded video file path
        return cv2.VideoCapture(current_video_source)
    else:
        # Check standard sample video paths
        sample_path = os.path.join(BASE_DIR, "sample bus video.mp4")
        if not os.path.exists(sample_path):
            sample_path = os.path.join(BASE_DIR, "sample_gate_video.mp4")
            
        if not os.path.exists(sample_path):
            try:
                from create_sample_video import generate_sample_video
                generate_sample_video(sample_path)
            except Exception as err:
                print(f"[WARNING] Unable to generate sample video: {err}")
                
        return cv2.VideoCapture(sample_path)


def generate_frames():
    """Video stream frame generator for continuous web dashboard rendering."""
    det, ocr, trk = get_pipeline()
    cap = get_video_capture()
    vehicle_plates = {}

    while True:
        if not cap.isOpened():
            cap = get_video_capture()
            time.sleep(0.5)
            continue

        ret, frame = cap.read()
        if not ret:
            # Re-seek video to frame 0 for uninterrupted continuous playback loop
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.2)
                continue

        frame_h, frame_w, _ = frame.shape
        gate_y = int(frame_h * 0.5)
        trk.gate_line_y = gate_y

        # Detect vehicles in frame
        detections = det.detect(frame)

        # Update centroid tracking
        tracking_results = trk.update(detections)

        # Draw Campus Gate Line Threshold Across Frame
        cv2.line(frame, (0, gate_y), (frame_w, gate_y), (0, 0, 255), 3)
        cv2.putText(frame, "GATE ENTRY THRESHOLD", (15, gate_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # Process detections & OCR
        for d in detections:
            box = d['box']
            x1, y1, x2, y2 = box
            
            # Green bounding box for detected vehicle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Bus ({d['confidence']})", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Crop region for license plate text reading
            roi = det.crop_plate_region(frame, box)
            plate_text, _ = ocr.read_plate(roi)
            if plate_text:
                cv2.putText(frame, f"PLATE: {plate_text}", (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Track vehicle movement & record crossing events
        for veh_id, info in tracking_results.items():
            cx, cy = info["centroid"]
            direction = info["direction"]
            plate_no = vehicle_plates.get(veh_id, "TN38CB2026")

            # Centroid point indicator
            cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)
            cv2.putText(frame, f"ID #{veh_id}", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if direction and not info["is_logged"]:
                logger.log_vehicle(veh_id, plate_no, direction)
                trk.logged_vehicles.add(veh_id)

                # Event banner on screen
                alert_text = f"LOGGED: {plate_no} ({direction.upper()})"
                cv2.putText(frame, alert_text, (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 3)

        # Encode Frame to JPEG
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

    cap.release()


def is_logged_in():
    """Check if user session is authenticated."""
    return 'user' in session


# --- Flask Application Routes ---

@app.route('/')
def index():
    if is_logged_in():
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Authentication credentials
        if username == 'admin' and password == 'admin123':
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            error = "Invalid Operator Credentials. Please try again."

    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/dashboard')
def dashboard():
    if not is_logged_in():
        return redirect(url_for('login'))

    stats = logger.get_summary_stats()
    logs = logger.get_records_list()
    return render_template('dashboard.html', stats=stats, logs=logs, current_source=current_video_source)


@app.route('/records')
def records():
    if not is_logged_in():
        return redirect(url_for('login'))

    search_query = request.args.get('search', '')
    status_filter = request.args.get('status', 'ALL')
    records_data = logger.get_records_list(search_query=search_query, status_filter=status_filter)
    return render_template('records.html', records=records_data)


@app.route('/video_feed')
def video_feed():
    """Video streaming route serving multipart boundary JPEG frames."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/stats')
def api_stats():
    """Return live system statistics for dashboard background updates."""
    stats = logger.get_summary_stats()
    stats['model_status'] = 'Operational'
    return jsonify(stats)


@app.route('/api/export')
def export_excel():
    """Endpoint for downloading the activity log Excel spreadsheet."""
    if not os.path.exists(EXCEL_FILE):
        logger._ensure_file_exists()
    return send_file(
        EXCEL_FILE,
        as_attachment=True,
        download_name="bus_gate_log_report.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@app.route('/api/set_source', methods=['POST'])
def set_source():
    """Switch active video feed source."""
    global current_video_source
    data = request.get_json() or {}
    new_source = data.get('source', 'sample_file')
    current_video_source = new_source
    return jsonify({"success": True, "source": current_video_source})


@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """Handle custom bus video file uploads and switch video stream."""
    global current_video_source
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No video file attached"}), 400
        
    file = request.files['video']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
        
    if file and is_allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        # Switch video stream to the uploaded file
        current_video_source = save_path
        return jsonify({
            "success": True,
            "filename": filename,
            "path": save_path,
            "message": f"Successfully loaded uploaded video: {filename}"
        })
    else:
        return jsonify({
            "success": False,
            "error": "Unsupported video format. Allowed formats: MP4, AVI, MOV, MKV, WEBM"
        }), 400


if __name__ == '__main__':
    print("=================================================================")
    print(" 🚌 CAMPUS BUS ENTRY & EXIT MONITORING SYSTEM")
    print(" Server running at: http://127.0.0.1:5000")
    print(" Login Credentials: admin / admin123")
    print("=================================================================")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
