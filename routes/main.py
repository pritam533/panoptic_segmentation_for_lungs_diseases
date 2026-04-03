# import os
# from flask import Blueprint, request, jsonify, render_template
# from tensorflow.keras.models import load_model
# import cv2

# from utils.segmentation import segment_image
# from utils.classification import classify_disease
# from utils.report_generator import generate_report
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# #  Correct SavedModel paths (VERY IMPORTANT)

# from tensorflow.keras.models import load_model

# unet_model = load_model("app/model/unet_model.h5", compile=False)
# classifier_model = load_model("app/model/classifier_model.h5", compile=False)

# print(" Models loaded successfully")
# # UNET_PATH = os.path.join(BASE_DIR, "..", "app", "model", "unet_saved_model")
# # CLASSIFIER_PATH = os.path.join(BASE_DIR, "..", "app", "model", "classifier_saved_model")

# # Load Segmentation Model
# try:
#     unet_model = load_model(UNET_PATH)
#     print("Segmentation model loaded successfully")
# except Exception as e:
#     print(" Segmentation model error:", str(e))

# #  Load Classification Model
# try:
#     classifier_model = load_model(CLASSIFIER_PATH)
#     print(" Classification model loaded successfully")
# except Exception as e:
#     print(" Classification model error:", str(e))


# # Ensure directories exist
# os.makedirs('app/static/uploaded_images', exist_ok=True)
# os.makedirs('app/static/output_images', exist_ok=True)

# main = Blueprint('main', __name__)

# @main.route('/')
# def index():
#     return render_template('index.html')

# @main.route('/analyze', methods=['POST'])
# def analyze():
#     try:
#         file = request.files['xray']
#         name = request.form['name']
#         age = float(request.form['age'])
#         gender = request.form['gender']

#         upload_dir = 'app/static/uploaded_images'
#         output_dir = 'app/static/output_images'

#         xray_path = os.path.join(upload_dir, file.filename)
#         mask_path = os.path.join(output_dir, f'mask_{file.filename}')
#         report_path = os.path.join(output_dir, f'report_{os.path.splitext(file.filename)[0]}.pdf')

#         file.save(xray_path)

#         #  Pass model explicitly
#         mask = segment_image(xray_path, unet_model)
#         cv2.imwrite(mask_path, mask)

#         disease, confidence, severity = classify_disease(xray_path, classifier_model)

#         generate_report(name, age, gender, xray_path, mask_path, disease, severity, report_path)

#         if severity.lower() == "low":
#             comment = "Mild condition detected. No immediate risk, but monitoring is advised."
#         elif severity.lower() == "medium":
#             comment = "Moderate infection detected. Please consult a doctor."
#         else:
#             comment = "Severe condition detected. Immediate medical attention required."

#         return jsonify({
#             "success": True,
#             "disease": disease,
#             "severity": severity,
#             "confidence": f"{confidence:.2f}",
#             "comment": comment,
#             "segmented_image": f"mask_{file.filename}",
#             "pdf_report": f"report_{os.path.splitext(file.filename)[0]}.pdf"
#         })

#     except Exception as e:
#         print(" ERROR in /analyze:", str(e))  #  DEBUG LINE
#         return jsonify({
#             "success": False,
#             "error": str(e)
#         }), 500






import os
from flask import Blueprint, request, jsonify, render_template
import cv2
import time
from utils.segmentation import segment_image
from utils.classification import classify_disease
from utils.report_generator import generate_report

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#  Global models
unet_model = None
classifier_model = None

#  Load models safely
def load_models():
    global unet_model, classifier_model
    
    if unet_model is None:
        from tensorflow.keras.models import load_model
        
        print(" Loading models...")

        try:
            unet_model = load_model("app/model/unet_model.h5", compile=False)
            print(" UNet loaded")
            unet_model = load_model("app/model/unet_model.h5", compile=False)
           classifier_model = load_model("app/model/classifier_model.h5", compile=False)
           print(" Models loaded")

# ✅ LOAD ON STARTUP (IMPORTANT)
load_models()
        # except Exception as e:
        #     print("❌ UNet load error:", e)

        # try:
        #     classifier_model = load_model("app/model/classifier_model.h5", compile=False)
        #     print("✅ Classifier loaded")
        # except Exception as e:
        #     print("❌ Classifier load error:", e)


# Ensure directories exist
os.makedirs('app/static/uploaded_images', exist_ok=True)
os.makedirs('app/static/output_images', exist_ok=True)

# ✅ Blueprint
main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')


@main.route('/analyze', methods=['POST'])
def analyze():


    start = time.time()

    print("🧠 Running segmentation...")
    mask = segment_image(xray_path, unet_model)

    print("⏱ Segmentation time:", time.time() - start)
    try:
        # ✅ VERY IMPORTANT: LOAD MODELS HERE
        load_models()

         
        if unet_model is None or classifier_model is None:
            return jsonify({
                "success": False,
                "error": "Model not loaded properly"
            }), 500

        file = request.files['xray']
        name = request.form['name']
        age = float(request.form['age'])
        gender = request.form['gender']

        upload_dir = 'app/static/uploaded_images'
        output_dir = 'app/static/output_images'

        xray_path = os.path.join(upload_dir, file.filename)
        mask_path = os.path.join(output_dir, f'mask_{file.filename}')
        report_path = os.path.join(output_dir, f'report_{os.path.splitext(file.filename)[0]}.pdf')

        file.save(xray_path)

        # ✅ Segmentation
        mask = segment_image(xray_path, unet_model)
        cv2.imwrite(mask_path, mask)

        # ✅ Classification
        disease, confidence, severity = classify_disease(xray_path, classifier_model)

        # ✅ Report
        generate_report(name, age, gender, xray_path, mask_path, disease, severity, report_path)

        # ✅ Comments
        if severity.lower() == "low":
            comment = "Mild condition detected. No immediate risk, but monitoring is advised."
        elif severity.lower() == "medium":
            comment = "Moderate infection detected. Please consult a doctor."
        else:
            comment = "Severe condition detected. Immediate medical attention required."

        return jsonify({
            "success": True,
            "disease": disease,
            "severity": severity,
            "confidence": f"{confidence:.2f}",
            "comment": comment,
            "segmented_image": f"mask_{file.filename}",
            "pdf_report": f"report_{os.path.splitext(file.filename)[0]}.pdf"
        })

    except Exception as e:
        print("❌ ERROR in /analyze:", str(e))
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500







# import os
# from flask import Blueprint, request, jsonify, render_template
# import cv2
# from flask import Flask

# app = Flask(__name__, template_folder='templates')

# app.register_blueprint(main)
# from utils.segmentation import segment_image
# from utils.classification import classify_disease
# from utils.report_generator import generate_report



# #  Lazy loading variables
# unet_model = None
# classifier_model = None

# #  Load models only when needed
# def load_models():
#     global unet_model, classifier_model
    
#     if unet_model is None:
#         from tensorflow.keras.models import load_model
        
#         print(" Loading models...")
#         unet_model = load_model("app/model/unet_model.h5", compile=False)
#         classifier_model = load_model("app/model/classifier_model.h5", compile=False)
#         print(" Models loaded successfully")

# # Ensure directories exist
# os.makedirs('app/static/uploaded_images', exist_ok=True)
# os.makedirs('app/static/output_images', exist_ok=True)

# main = Blueprint('main', __name__)

# @main.route('/')
# def index():
#     return "API is running "





# import os
# from flask import Blueprint, request, jsonify, render_template
# from tensorflow.keras.models import load_model
# import cv2

# from utils.segmentation import segment_image
# from utils.classification import classify_disease
# from utils.report_generator import generate_report
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# #  Correct SavedModel paths (VERY IMPORTANT)

# from tensorflow.keras.models import load_model

# unet_model = load_model("app/model/unet_model.h5", compile=False)
# classifier_model = load_model("app/model/classifier_model.h5", compile=False)

# print(" Models loaded successfully")
# # UNET_PATH = os.path.join(BASE_DIR, "..", "app", "model", "unet_saved_model")
# # CLASSIFIER_PATH = os.path.join(BASE_DIR, "..", "app", "model", "classifier_saved_model")

# # Load Segmentation Model
# try:
#     unet_model = load_model(UNET_PATH)
#     print("Segmentation model loaded successfully")
# except Exception as e:
#     print(" Segmentation model error:", str(e))

# #  Load Classification Model
# try:
#     classifier_model = load_model(CLASSIFIER_PATH)
#     print(" Classification model loaded successfully")
# except Exception as e:
#     print(" Classification model error:", str(e))


# # Ensure directories exist
# os.makedirs('app/static/uploaded_images', exist_ok=True)
# os.makedirs('app/static/output_images', exist_ok=True)

# main = Blueprint('main', __name__)
# import os
# from flask import Blueprint, request, jsonify, render_template
# import cv2

# from utils.segmentation import segment_image
# from utils.classification import classify_disease
# from utils.report_generator import generate_report

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# unet_model = None
# classifier_model = None

# def load_models():
#     global unet_model, classifier_model
#     if unet_model is None:
#         from tensorflow.keras.models import load_model
#         print("🔄 Loading models...")
#         unet_model = load_model("app/model/unet_model.h5", compile=False)
#         classifier_model = load_model("app/model/classifier_model.h5", compile=False)
#         print("✅ Models loaded")

# os.makedirs('app/static/uploaded_images', exist_ok=True)
# os.makedirs('app/static/output_images', exist_ok=True)

# # ✅ Blueprint only here
# main = Blueprint('main', __name__)

# @main.route('/')
# def index():
#     return render_template('index.html')
# # @main.route('/')
# # def index():
# #     return render_template('index.html')

# @main.route('/analyze', methods=['POST'])
# def analyze():
#     try:
#         file = request.files['xray']
#         name = request.form['name']
#         age = float(request.form['age'])
#         gender = request.form['gender']

#         upload_dir = 'app/static/uploaded_images'
#         output_dir = 'app/static/output_images'

#         xray_path = os.path.join(upload_dir, file.filename)
#         mask_path = os.path.join(output_dir, f'mask_{file.filename}')
#         report_path = os.path.join(output_dir, f'report_{os.path.splitext(file.filename)[0]}.pdf')

#         file.save(xray_path)

#         #  Pass model explicitly
#         mask = segment_image(xray_path, unet_model)
#         cv2.imwrite(mask_path, mask)

#         disease, confidence, severity = classify_disease(xray_path, classifier_model)

#         generate_report(name, age, gender, xray_path, mask_path, disease, severity, report_path)

#         if severity.lower() == "low":
#             comment = "Mild condition detected. No immediate risk, but monitoring is advised."
#         elif severity.lower() == "medium":
#             comment = "Moderate infection detected. Please consult a doctor."
#         else:
#             comment = "Severe condition detected. Immediate medical attention required."

#         return jsonify({
#             "success": True,
#             "disease": disease,
#             "severity": severity,
#             "confidence": f"{confidence:.2f}",
#             "comment": comment,
#             "segmented_image": f"mask_{file.filename}",
#             "pdf_report": f"report_{os.path.splitext(file.filename)[0]}.pdf"
#         })

#     except Exception as e:
#         print(" ERROR in /analyze:", str(e))  #  DEBUG LINE
#         return jsonify({
#             "success": False,
#             "error": str(e)
#         }), 500








# from flask import Blueprint, request, jsonify, render_template
# import sys
# import os
# from tensorflow.keras.models import load_model
# import cv2  # Added import
# from utils.segmentation import segment_image
# from utils.classification import classify_disease
# from utils.report_generator import generate_report

# unet_model = load_model("app/model/unet_fixed.h5")
# classifier_model = load_model("app/model/classifier_model.h5")
# # Ensure directories exist
# os.makedirs('app/static/uploaded_images', exist_ok=True)
# os.makedirs('app/static/output_images', exist_ok=True)

# main = Blueprint('main', __name__)
# #added a simple home route for testing
# @main.route("/")
# def home():
#     return "Lung Detection API Running "

# @main.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')

# @main.route('/analyze', methods=['POST'])
# def analyze():
#     try:
#         file = request.files['xray']
#         name = request.form['name']
#         age = float(request.form['age'])
#         gender = request.form['gender']
    
#         # Create paths using os.path.join for cross-platform compatibility
#         upload_dir = 'app/static/uploaded_images'
#         output_dir = 'app/static/output_images'
        
#         xray_path = os.path.join(upload_dir, file.filename)
#         mask_path = os.path.join(output_dir, f'mask_{file.filename}')
#         report_path = os.path.join(output_dir, f'report_{os.path.splitext(file.filename)[0]}.pdf')

#         file.save(xray_path)
        

#         # Segment the lung
#         mask = segment_image(xray_path)
#         cv2.imwrite(mask_path, mask)

#         # Classify disease
#         disease, confidence, severity = classify_disease(xray_path)
        
#         # Generate report
#         generate_report(name, age, gender, xray_path, mask_path, disease, severity, report_path)
#         # Determine the comment based on severity
#         if severity.lower() == "low":
#           comment = "Mild condition detected. No immediate risk, but monitoring is advised."
#         elif severity.lower() == "medium":
#           comment = "Moderate infection detected. Please consult a doctor."
#         else:
#           comment = "Severe condition detected. Immediate medical attention required."

# # Return the JSON response
#         return jsonify({
#             "success": True,
#             "disease": disease,
#             "severity": severity,
#             "confidence": f"{confidence:.2f}",
#            "comment": comment,  # Include the computed comment here
#            "segmented_image": f"mask_{file.filename}",
#           "pdf_report": f"report_{os.path.splitext(file.filename)[0]}.pdf"
#     })
       
#     except Exception as e:
#         return jsonify({
#             "success": False,
#             "error": str(e)
#         }), 500



# import os
# from flask import Blueprint, request, jsonify, render_template
# from tensorflow.keras.models import load_model
# import cv2

# from utils.segmentation import segment_image
# from utils.classification import classify_disease
# from utils.report_generator import generate_report

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # unet_model = load_model(os.path.join(BASE_DIR, "..", "app", "model", "unet_compatible.h5"), compile=False)
# # classifier_model = load_model(os.path.join(BASE_DIR, "..", "app", "model", "classifier_compatible.h5"), compile=False)

# # unet_model = load_model("app/model/unet_compatible.h5", compile=False)
# try:
#     unet_model = load_model("app/model/unet_compatible.h5", compile=False)
#     print(" Segmentation model loaded")
# except Exception as e:
#     print(" Segmentation model error:", str(e))
# # classifier_model = load_model("app/model/classifier_compatible.h5", compile=False)
# try:
#     classifier_model = load_model("app/model/classifier_model.h5", compile=False)
#     print("✅ Classification model loaded")
# except Exception as e:
#     print("❌ Classification model error:", str(e))


# os.makedirs('app/static/uploaded_images', exist_ok=True)
# os.makedirs('app/static/output_images', exist_ok=True)

# main = Blueprint('main', __name__)

# # @main.route("/")
# # def home():
# #     return "Lung Detection API Running"

# @main.route('/analyze', methods=['POST'])
# def analyze():
#     try:
#         file = request.files['xray']
#         name = request.form['name']
#         age = float(request.form['age'])
#         gender = request.form['gender']

#         upload_dir = 'app/static/uploaded_images'
#         output_dir = 'app/static/output_images'

#         xray_path = os.path.join(upload_dir, file.filename)
#         mask_path = os.path.join(output_dir, f'mask_{file.filename}')
#         report_path = os.path.join(output_dir, f'report_{os.path.splitext(file.filename)[0]}.pdf')

#         file.save(xray_path)

#         mask = segment_image(xray_path, unet_model)
#         cv2.imwrite(mask_path, mask)

#         disease, confidence, severity = classify_disease(xray_path, classifier_model)

#         generate_report(name, age, gender, xray_path, mask_path, disease, severity, report_path)

#         if severity.lower() == "low":
#             comment = "Mild condition detected. No immediate risk, but monitoring is advised."
#         elif severity.lower() == "medium":
#             comment = "Moderate infection detected. Please consult a doctor."
#         else:
#             comment = "Severe condition detected. Immediate medical attention required."

#         return jsonify({
#             "success": True,
#             "disease": disease,
#             "severity": severity,
#             "confidence": f"{confidence:.2f}",
#             "comment": comment,
#             "segmented_image": f"mask_{file.filename}",
#             "pdf_report": f"report_{os.path.splitext(file.filename)[0]}.pdf"
#         })

#     except Exception as e:
#         return jsonify({
#             "success": False,
#             "error": str(e)
#         }), 500



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
