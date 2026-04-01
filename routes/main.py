from flask import Blueprint, request, jsonify, render_template
import sys
import os
from tensorflow.keras.models import load_model
import cv2  # Added import
from utils.segmentation import segment_image
from utils.classification import classify_disease
from utils.report_generator import generate_report

unet_model = load_model("app/model/unet_fixed.h5")
classifier_model = load_model("app/model/classifier_model.h5")
# Ensure directories exist
os.makedirs('app/static/uploaded_images', exist_ok=True)
os.makedirs('app/static/output_images', exist_ok=True)

main = Blueprint('main', __name__)
#added a simple home route for testing
@main.route("/")
def home():
    return "Lung Detection API Running "

@main.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['xray']
        name = request.form['name']
        age = float(request.form['age'])
        gender = request.form['gender']
    
        # Create paths using os.path.join for cross-platform compatibility
        upload_dir = 'app/static/uploaded_images'
        output_dir = 'app/static/output_images'
        
        xray_path = os.path.join(upload_dir, file.filename)
        mask_path = os.path.join(output_dir, f'mask_{file.filename}')
        report_path = os.path.join(output_dir, f'report_{os.path.splitext(file.filename)[0]}.pdf')

        file.save(xray_path)
        

        # Segment the lung
        mask = segment_image(xray_path)
        cv2.imwrite(mask_path, mask)

        # Classify disease
        disease, confidence, severity = classify_disease(xray_path)
        
        # Generate report
        generate_report(name, age, gender, xray_path, mask_path, disease, severity, report_path)
        # Determine the comment based on severity
        if severity.lower() == "low":
          comment = "Mild condition detected. No immediate risk, but monitoring is advised."
        elif severity.lower() == "medium":
          comment = "Moderate infection detected. Please consult a doctor."
        else:
          comment = "Severe condition detected. Immediate medical attention required."

# Return the JSON response
        return jsonify({
            "success": True,
            "disease": disease,
            "severity": severity,
            "confidence": f"{confidence:.2f}",
           "comment": comment,  # Include the computed comment here
           "segmented_image": f"mask_{file.filename}",
          "pdf_report": f"report_{os.path.splitext(file.filename)[0]}.pdf"
    })
       
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500




# from flask import Blueprint, app, request, jsonify
# import os

# from tensorflow.keras.models import load_model
# from utils.segmentation import segment_image
# from utils.classification import classify_disease
# from utils.report_generator import generate_report

# main = Blueprint('main', __name__)

# #  Load models ONLY ONCE (VERY IMPORTANT)
# print(" Loading models...")

# # unet_model = load_model("app/model/unet_model.h5")
# unet_model = load_model("app/model/unet_model.keras")
# classifier_model = load_model("app/model/classifier_model.keras")


# UPLOAD_FOLDER = "app/static/uploads"
# OUTPUT_FOLDER = "app/static/output_images"

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# @main.route('/analyze', methods=['POST'])
# def analyze():

#     try:
#         file = request.files['xray']
#         name = request.form.get('name')
#         age = request.form.get('age')
#         gender = request.form.get('gender')

#         # Save uploaded file
#         file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#         file.save(file_path)

#         #  Segmentation
#         mask_filename = f"mask_{file.filename}"
#         mask_path = os.path.join(OUTPUT_FOLDER, mask_filename)

#         segment_image(file_path, mask_path, unet_model)

#         # Classification
#         disease, confidence, severity = classify_disease(file_path, classifier_model)

#         #  Comment logic
#         if severity.lower() == "low":
#             comment = "Mild condition detected. No immediate risk, but monitoring is advised."
#         elif severity.lower() == "medium":
#             comment = "Moderate infection detected. Please consult a doctor."
#         else:
#             comment = "Severe condition detected. Immediate medical attention required."

#         #  PDF Report
#         pdf_filename = f"report_{os.path.splitext(file.filename)[0]}.pdf"
#         pdf_path = os.path.join(OUTPUT_FOLDER, pdf_filename)

#         generate_report(
#             name, age, gender,
#             file_path, mask_path,
#             disease, severity,
#             pdf_path
#         )

#         return jsonify({
#             "success": True,
#             "disease": disease,
#             "severity": severity,
#             "confidence": f"{confidence:.2f}",
#             "comment": comment,
#             "segmented_image": mask_filename,
#             "pdf_report": pdf_filename
#         })

#     except Exception as e:
#         return jsonify({"success": False, "error": str(e)})