from flask import Blueprint, request, jsonify, render_template
import sys
import os
import cv2  # Added import
from utils.segmentation import segment_lung
from utils.classification import classify_disease
from utils.report_generator import generate_report

# Ensure directories exist
os.makedirs('app/static/uploaded_images', exist_ok=True)
os.makedirs('app/static/output_images', exist_ok=True)

main = Blueprint('main', __name__)

@main.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['xray']
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']

        # Create paths using os.path.join for cross-platform compatibility
        upload_dir = 'app/static/uploaded_images'
        output_dir = 'app/static/output_images'
        
        xray_path = os.path.join(upload_dir, file.filename)
        mask_path = os.path.join(output_dir, f'mask_{file.filename}')
        report_path = os.path.join(output_dir, f'report_{os.path.splitext(file.filename)[0]}.pdf')

        file.save(xray_path)

        # Segment the lung
        mask = segment_lung(xray_path)
        cv2.imwrite(mask_path, mask)

        # Classify disease
        disease, confidence, severity = classify_disease(xray_path)
        
        # Generate report
        generate_report(name, age, gender, xray_path, mask_path, disease, severity, report_path)

        return jsonify({
            "success": True,
            "disease": disease,
            "severity": severity,
            "confidence": f"{confidence:.2f}",
            "segmented_image": f'mask_{file.filename}',
            "pdf_report": f'report_{os.path.splitext(file.filename)[0]}.pdf'
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


