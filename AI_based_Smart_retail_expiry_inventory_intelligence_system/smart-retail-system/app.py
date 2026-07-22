import os
import base64
import numpy as np
import cv2
from datetime import datetime, date
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from ai_engine import RetailAIEngine

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///inventory.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize deep learning AI interface
ai_engine = RetailAIEngine()

# Database Model for Inventory Management
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    scan_date = db.Column(db.Date, default=date.today)
    expiry_date = db.Column(db.Date, nullable=True)
    status = db.Column(db.String(20), default="Unknown") # Safe, Expiring Soon, Expired
    days_left = db.Column(db.Integer, nullable=True)

    def calculate_status(self):
        if not self.expiry_date:
            self.status = "Unknown"
            self.days_left = None
            return
        
        today_val = date.today()
        delta = (self.expiry_date - today_val).days
        self.days_left = delta
        
        if delta < 0:
            self.status = "Expired"
        elif delta <= 7:
            self.status = "Expiring Soon"
        else:
            self.status = "Safe"

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/inventory', methods=['GET'])
def get_inventory():
    items = Product.query.order_by(Product.expiry_date.asc()).all()
    # Refresh statuses dynamically to reflect chronological shifts
    inventory_list = []
    for item in items:
        item.calculate_status()
        db.session.commit()
        inventory_list.append({
            "id": item.id,
            "name": item.name.capitalize(),
            "scan_date": item.scan_date.strftime("%Y-%m-%d"),
            "expiry_date": item.expiry_date.strftime("%Y-%m-%d") if item.expiry_date else "N/A",
            "status": item.status,
            "days_left": item.days_left
        })
    return jsonify(inventory_list)

@app.route('/api/scan', methods=['POST'])
def scan_frame():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "Missing image dataset"}), 400

    # Parse standard Base64 dynamic web-camera payload
    image_data = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid frame decode format"}), 400

    # Run AI analysis
    annotated, detections = ai_engine.analyze_frame(frame)

    # Encode inference results frame to render seamlessly on front-end overlay
    _, buffer = cv2.imencode('.jpg', annotated)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    results_saved = []
    
    # Auto-commit detected objects containing active expiry signals to persistent inventory db
    for item in detections:
        # Avoid duplicate database spamming logic - simplify logic for active alerts
        if item['expiry_date'] is not None:
            existing = Product.query.filter_by(
                name=item['item_name'], 
                expiry_date=item['expiry_date']
            ).first()
            
            if not existing:
                new_product = Product(
                    name=item['item_name'],
                    expiry_date=item['expiry_date']
                )
                new_product.calculate_status()
                db.session.add(new_product)
                db.session.commit()
                results_saved.append({
                    "name": item['item_name'],
                    "expiry_date": item['expiry_date'].strftime("%Y-%m-%d"),
                    "status": new_product.status
                })

    return jsonify({
        "annotated_image": f"data:image/jpeg;base64,{encoded_image}",
        "detections": [
            {
                "name": det['item_name'],
                "confidence": round(det['confidence'], 2),
                "expiry": det['expiry_date'].strftime("%Y-%m-%d") if det['expiry_date'] else "Not Detected"
            } for det in detections
        ],
        "items_added": results_saved
    })

@app.route('/api/delete/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    item = Product.query.get(item_id)
    if not item:
        return jsonify({"error": "Item not found"}), 404
    db.session.delete(item)
    db.session.commit()
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)