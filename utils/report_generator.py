# from fpdf import FPDF
# from datetime import datetime

# class PDFReport(FPDF):
#     def header(self):
#         self.set_font('Arial', 'B', 14)
#         self.cell(0, 10, 'Lung Disease Detection Report', ln=True, align='C')
#         self.ln(10)

#     # def add_patient_info(self, name, age, gender):
#     #     self.set_font('Arial', '', 12)
#     #     self.cell(0, 10, f'Name: {name}    Age: {age}    Gender: {gender}', ln=True)
#     #     self.cell(0, 10, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True)
#     #     self.ln(5)
#     def add_patient_info(self, name, age, gender):
#          self.set_font('Arial', '', 12)
    
#          self.cell(0, 10, f'Name: {name}', ln=True)
#          self.cell(0, 10, f'Age: {age}', ln=True)
#          self.cell(0, 10, f'Gender: {gender}', ln=True)
    
#          self.cell(0, 10, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True)
#          self.ln(5)

#     def add_images(self, xray_path, mask_path):
#         self.cell(0, 10, 'X-ray Image:', ln=True)
#         self.image(xray_path, w=90)
#         self.ln(5)
#         self.cell(0, 10, 'Segmented Output:', ln=True)
#         self.image(mask_path, w=90)
#         self.ln(10)

#     def add_diagnosis(self, disease, severity, comment):
#         self.set_font('Arial', 'B', 12)
#         self.cell(0, 10, f'Diagnosis: {disease}', ln=True)
#         self.cell(0, 10, f'Severity: {severity}', ln=True)
#         self.set_font('Arial', '', 12)
#         self.multi_cell(0, 10, f'Comments:\n{comment}')
#         self.ln(10)

#     def footer(self):
#         self.set_y(-15)
#         self.set_font('Arial', 'I', 10)
#         self.cell(0, 10, 'NOTE: this is an AI-generated report. Please consult a certified radiologist. Thank you for using our service.', align='C')


# def get_disease_comment(disease, severity):
#     comments = {
#         "COVID-19": {
#             "Mild": "Ground-glass opacities are faintly visible, typically associated with mild COVID-19. Recommend monitoring and follow-up scan.",
#             "Moderate": "Bilateral opacities with peripheral distribution indicate moderate COVID-19 pneumonia. Suggest home isolation and treatment unless symptoms worsen.",
#             "Severe": "Extensive ground-glass opacities and consolidation detected. Suggest immediate hospitalization for advanced COVID-19 treatment."
#         },
#         "Pneumonia": {
#             "Mild": "Localized patchy infiltrates suggest early-stage pneumonia. Recommend antibiotics and follow-up.",
#             "Moderate": "Lung lobes show dense consolidation, typical of moderate pneumonia. Clinical treatment and rest advised.",
#             "Severe": "Widespread opacities indicate severe pneumonia. Urgent clinical attention required."
#         },
#         "Tuberculosis": {
#             "Mild": "Apical scarring may indicate early-stage tuberculosis. Recommend sputum test and anti-TB medication.",
#             "Moderate": "Fibronodular lesions seen, likely due to active TB. Suggest starting anti-tubercular therapy immediately.",
#             "Severe": "Extensive cavitary lesions observed, typical of severe TB. Hospitalization and long-term treatment needed."
#         },
#         "Normal": {
#             "Mild": "Lungs appear healthy. No abnormal findings.",
#             "Moderate": "Slight deviations detected but not indicative of major disease.",
#             "Severe": "Abnormal scan reported, but not matching major disease pattern. Recommend detailed physical exam."
#         }
#     }

#     return comments.get(disease, {}).get(severity, "No specific comment available.")


# def generate_report(name, age, gender, xray_path, mask_path, disease, severity, output_path):
#     comment = get_disease_comment(disease, severity)
#     pdf = PDFReport()
#     pdf.add_page()
#     pdf.add_patient_info(name, age, gender)
#     pdf.add_images(xray_path, mask_path)
#     pdf.add_diagnosis(disease, severity, comment)
#     pdf.output(output_path)




from fpdf import FPDF
from datetime import datetime

class PDFReport(FPDF):

    def header(self):
        self.set_fill_color(30, 144, 255)  # Blue header
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 16)
        self.cell(0, 12, ' Lung Disease Detection Report', 0, 1, 'C', fill=True)
        self.ln(5)

        # Reset text color
        self.set_text_color(0, 0, 0)

    #  Patient Info Box
    def add_patient_info(self, name, age, gender):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, 'Patient Information', ln=True)

        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.5)

        x_start = self.get_x()
        y_start = self.get_y()

        self.set_font('Arial', '', 11)

        self.cell(0, 8, f'Name   : {name}', ln=True)
        self.cell(0, 8, f'Age    : {age}', ln=True)
        self.cell(0, 8, f'Gender : {gender}', ln=True)
        self.cell(0, 8, f'Date   : {datetime.now().strftime("%d-%m-%Y %H:%M")}', ln=True)

        y_end = self.get_y()

        # Draw box
        self.rect(x_start, y_start, 190, y_end - y_start)
        self.ln(5)

    #  Images Side-by-Side
    def add_images(self, xray_path, mask_path):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, 'Analysis Images', ln=True)

        y_before = self.get_y()

        self.image(xray_path, x=15, y=y_before, w=80)
        self.image(mask_path, x=110, y=y_before, w=80)

        self.ln(85)

    #  Diagnosis Section
    def add_diagnosis(self, disease, severity, comment):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, 'Diagnosis Summary', ln=True)

        #  Severity Color
        if severity.lower() == "low":
            self.set_text_color(0, 128, 0)  # Green
        elif severity.lower() == "medium":
            self.set_text_color(255, 165, 0)  # Orange
        else:
            self.set_text_color(220, 20, 60)  # Red

        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, f'Disease Detected : {disease}', ln=True)
        self.cell(0, 8, f'Severity Level   : {severity}', ln=True)

        # Reset color
        self.set_text_color(0, 0, 0)

        self.ln(3)

        # Comment Box
        self.set_font('Arial', 'B', 11)
        self.cell(0, 8, 'AI Medical Comment', ln=True)

        self.set_fill_color(245, 245, 245)  # Light gray background
        self.set_font('Arial', '', 11)

        self.multi_cell(0, 8, comment, border=1, fill=True)
        self.ln(5)


     #  Footer
    def footer(self):
      self.set_y(-20)

    # Main Note
      self.set_font('Arial', 'I', 9)
      self.set_text_color(120, 120, 120)

      self.multi_cell(
        0,
        5,
        'NOTE: This is an AI-generated report. Please consult a certified radiologist for medical advice.',
        align='C'
      )

    # Copyright Line (Low opacity effect)
      self.ln(2)
      self.set_font('Arial', 'I', 8)
      self.set_text_color(180, 180, 180)  # Light gray = low opacity look

      self.cell(
        0,
        5,
        '© 2024 Preetam Verma. All Rights Reserved.',
        align='C'
     )

#  Disease Comment Logic
def get_disease_comment(disease, severity):
    comments = {
        "COVID-19": {
            "Mild": "Mild ground-glass opacities detected. Regular monitoring is recommended.",
            "Moderate": "Moderate infection signs observed. Medical consultation advised.",
            "Severe": "Severe lung involvement detected. Immediate hospitalization required."
        },
        "Pneumonia": {
            "Mild": "Early-stage pneumonia detected. Medication and rest advised.",
            "Moderate": "Clear signs of pneumonia. Proper medical treatment needed.",
            "Severe": "Severe pneumonia detected. Urgent care required."
        },
        "Tuberculosis": {
            "Mild": "Initial TB signs visible. Further testing recommended.",
            "Moderate": "Active TB patterns detected. Start treatment immediately.",
            "Severe": "Advanced TB detected. Long-term treatment required."
        },
        "Normal": {
            "Mild": "Lungs appear healthy. No abnormalities detected.",
            "Moderate": "Minor variations observed but not critical.",
            "Severe": "Unexpected pattern detected. Further evaluation suggested."
        }
    }

    return comments.get(disease, {}).get(severity, "No specific comment available.")


#  Main Function
def generate_report(name, age, gender, xray_path, mask_path, disease, severity, output_path):
    comment = get_disease_comment(disease, severity)

    pdf = PDFReport()
    pdf.add_page()

    pdf.add_patient_info(name, age, gender)
    pdf.add_images(xray_path, mask_path)
    pdf.add_diagnosis(disease, severity, comment)

    pdf.output(output_path)