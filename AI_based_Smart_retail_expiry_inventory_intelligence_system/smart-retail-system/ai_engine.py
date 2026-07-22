import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
import easyocr
import re
from datetime import datetime, date
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COCO Class index map for common retail/grocery items
COCO_CLASSES = {
    44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife",
    50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich",
    55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza",
    60: "donut", 61: "cake"
}

class RetailAIEngine:
    def __init__(self):
        logger.info("Initializing Object Detection Model (MobileNetV3 SSD)...")
        # Load lightweight MobileNet SSD for CPU/GPU efficiency
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.detector = ssdlite320_mobilenet_v3_large(weights=self.weights)
        self.detector.eval()
        
        # Check device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector.to(self.device)
        
        logger.info(f"Initializing EasyOCR Engine on {self.device}...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        # Regex patterns for various global expiry formats
        # Matches: DD/MM/YYYY, DD-MM-YYYY, MM/YYYY, YYYY/MM/DD, DD.MM.YY etc.
        self.date_patterns = [
            r'\b(0[1-9]|[12]\d|3[01])[-/\.](0[1-9]|1[0-2])[-/\.](20\d{2}|\d{2})\b', # DD/MM/YYYY or DD/MM/YY
            r'\b(0[1-9]|1[0-2])[-/\.](20\d{2}|\d{2})\b',                           # MM/YYYY or MM/YY
            r'\b(20\d{2})[-/\.](0[1-9]|1[0-2])[-/\.](0[1-9]|[12]\d|3[01])\b'       # YYYY/MM/DD
        ]

    def parse_expiry_date(self, text):
        """
        Applies regular expression patterns to OCR output stream to locate standard date metrics.
        """
        text = text.upper().strip()
        # Clean up typical optical noise characters
        text = re.sub(r'[^A-Z0-9/\-\.]', ' ', text)
        
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # If match is tuple (e.g. from regex groups), format it
                if isinstance(match, tuple):
                    parts = [p for p in match if p]
                else:
                    parts = re.split(r'[-/\.]', match)
                
                try:
                    if len(parts) == 3:
                        # DD, MM, YYYY or YYYY, MM, DD
                        if len(parts[0]) == 4: # YYYY-MM-DD
                            year = int(parts[0])
                            month = int(parts[1])
                            day = int(parts[2])
                        else: # DD-MM-YYYY
                            day = int(parts[0])
                            month = int(parts[1])
                            year = int(parts[2])
                            if year < 100: # Handle 2 digit year
                                year += 2000
                        return date(year, month, day)
                        
                    elif len(parts) == 2:
                        # MM-YYYY
                        month = int(parts[0])
                        year = int(parts[1])
                        if year < 100:
                            year += 2000
                        # Default to last day of the month for safety
                        day = 28 if month == 2 else 30
                        if month in [1, 3, 5, 7, 8, 10, 12]:
                            day = 31
                        return date(year, month, day)
                except ValueError:
                    continue
        return None

    def analyze_frame(self, frame):
        """
        Performs object detection and localized OCR on a video frame.
        :param frame: Standard OpenCV numpy image (BGR format).
        :return: (annotated_frame, list_of_detected_items)
        """
        h, w, _ = frame.shape
        # Convert to RGB and normalize for PyTorch object detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_img = torchvision.transforms.functional.to_tensor(rgb_frame).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.detector(tensor_img)[0]
            
        annotated_frame = frame.copy()
        detected_items = []
        
        # High confidence detection boundary
        threshold = 0.55
        
        for idx in range(len(predictions['scores'])):
            score = predictions['scores'][idx].item()
            if score > threshold:
                label_id = int(predictions['labels'][idx].item())
                # Default to generic "item" if label matches outside specific food inventory classes
                class_name = COCO_CLASSES.get(label_id, "Retail Item")
                
                box = predictions['boxes'][idx].cpu().numpy().astype(int)
                xmin, ymin, xmax, ymax = box
                
                # Clip box to image boundaries
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(w, xmax), min(h, ymax)
                
                # Draw main object prediction box
                cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # Perform cropped region-of-interest OCR scanning to find expiry indicators
                roi = rgb_frame[ymin:ymax, xmin:xmax]
                detected_expiry = None
                
                if roi.size > 0:
                    ocr_results = self.ocr_reader.readtext(roi)
                    for (bbox, text, ocr_score) in ocr_results:
                        if ocr_score > 0.4:
                            parsed_date = self.parse_expiry_date(text)
                            if parsed_date:
                                detected_expiry = parsed_date
                                # Highlight the OCR bounding box location relative to primary image frame
                                rx1 = xmin + int(bbox[0][0])
                                ry1 = ymin + int(bbox[0][1])
                                rx2 = xmin + int(bbox[2][0])
                                ry2 = ymin + int(bbox[2][1])
                                cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
                                break
                
                expiry_str = detected_expiry.strftime("%Y-%m-%d") if detected_expiry else "Unknown"
                label_text = f"{class_name.capitalize()} (Exp: {expiry_str})"
                
                cv2.putText(annotated_frame, label_text, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                detected_items.append({
                    "item_name": class_name,
                    "confidence": float(score),
                    "expiry_date": detected_expiry
                })
                
        return annotated_frame, detected_items