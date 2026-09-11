"""
ocr.py - OCR Module for Bus Monitoring System
---------------------------------------------
Reads vehicle registration numbers using EasyOCR.
"""

import re
import cv2
import easyocr
import numpy as np


class PlateReader:

    def __init__(self, languages=None, gpu=False):

        if languages is None:
            languages = ["en"]

        print("[INFO] Loading EasyOCR...")

        try:
            self.reader = easyocr.Reader(languages, gpu=gpu)
            print("[INFO] EasyOCR loaded successfully.")
        except Exception as e:
            print("[ERROR] EasyOCR initialization failed:", e)
            self.reader = None

    # ---------------------------------------------------------
    # Image Preprocessing
    # ---------------------------------------------------------
    def preprocess_image(self, image):

        if image is None or image.size == 0:
            return None

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize
        gray = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )

        # Noise removal
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Threshold
        thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]

        return thresh

    # ---------------------------------------------------------
    # Clean OCR Output
    # ---------------------------------------------------------
    def clean_text(self, text):

        text = text.upper()

        text = re.sub(r"[^A-Z0-9]", "", text)

        return text

    # ---------------------------------------------------------
    # Validate Indian Registration Number
    # Example:
    # TN38AB1234
    # KA01MN5678
    # ---------------------------------------------------------
    def valid_plate(self, text):

        pattern = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$"

        return re.match(pattern, text) is not None

    # ---------------------------------------------------------
    # OCR
    # ---------------------------------------------------------
    def read_plate(self, image):

        if self.reader is None:
            return "", 0.0

        processed = self.preprocess_image(image)

        if processed is None:
            return "", 0.0

        try:
            results = self.reader.readtext(
                processed,
                detail=1,
                paragraph=False
            )
        except Exception as e:
            print("[OCR ERROR]", e)
            return "", 0.0

        best_plate = ""
        best_conf = 0.0

        for _, text, conf in results:

            plate = self.clean_text(text)

            if self.valid_plate(plate):

                if conf > best_conf:
                    best_plate = plate
                    best_conf = conf

        # Fallback if no valid Indian plate found
        if best_plate == "":

            for _, text, conf in results:

                plate = self.clean_text(text)

                if len(plate) >= 5 and conf > best_conf:
                    best_plate = plate
                    best_conf = conf

        return best_plate, round(float(best_conf), 2)


# -----------------------------------------------------------------
# Test
# -----------------------------------------------------------------
if __name__ == "__main__":

    img = np.ones((120, 350, 3), dtype=np.uint8) * 255

    cv2.putText(
        img,
        "TN38AB1234",
        (15, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 0),
        3,
    )

    ocr = PlateReader(gpu=False)

    plate, confidence = ocr.read_plate(img)

    print("\nDetected Plate :", plate)
    print("Confidence     :", confidence)