from fpdf import FPDF
from datetime import datetime

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Lung Disease Detection Report', ln=True, align='C')
        self.ln(10)

    def add_patient_info(self, name, age, gender):
        self.set_font('Arial', '', 12)
        self.cell(0, 10, f'Name: {name}    Age: {age}    Gender: {gender}', ln=True)
        self.cell(0, 10, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True)
        self.ln(5)

    def add_images(self, xray_path, mask_path):
        self.cell(0, 10, 'X-ray Image:', ln=True)
        self.image(xray_path, w=90)
        self.ln(5)
        self.cell(0, 10, 'Segmented Output:', ln=True)
        self.image(mask_path, w=90)
        self.ln(10)

    def add_diagnosis(self, disease, severity, comment):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, f'Diagnosis: {disease}', ln=True)
        self.cell(0, 10, f'Severity: {severity}', ln=True)
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, f'Comments:\n{comment}')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, 'AI-generated report. Please consult a certified radiologist.', align='C')


def get_disease_comment(disease, severity):
    comments = {
        "COVID-19": {
            "Mild": "Ground-glass opacities are faintly visible, typically associated with mild COVID-19. Recommend monitoring and follow-up scan.",
            "Moderate": "Bilateral opacities with peripheral distribution indicate moderate COVID-19 pneumonia. Suggest home isolation and treatment unless symptoms worsen.",
            "Severe": "Extensive ground-glass opacities and consolidation detected. Suggest immediate hospitalization for advanced COVID-19 treatment."
        },
        "Pneumonia": {
            "Mild": "Localized patchy infiltrates suggest early-stage pneumonia. Recommend antibiotics and follow-up.",
            "Moderate": "Lung lobes show dense consolidation, typical of moderate pneumonia. Clinical treatment and rest advised.",
            "Severe": "Widespread opacities indicate severe pneumonia. Urgent clinical attention required."
        },
        "Tuberculosis": {
            "Mild": "Apical scarring may indicate early-stage tuberculosis. Recommend sputum test and anti-TB medication.",
            "Moderate": "Fibronodular lesions seen, likely due to active TB. Suggest starting anti-tubercular therapy immediately.",
            "Severe": "Extensive cavitary lesions observed, typical of severe TB. Hospitalization and long-term treatment needed."
        },
        "Normal": {
            "Mild": "Lungs appear healthy. No abnormal findings.",
            "Moderate": "Slight deviations detected but not indicative of major disease.",
            "Severe": "Abnormal scan reported, but not matching major disease pattern. Recommend detailed physical exam."
        }
    }

    return comments.get(disease, {}).get(severity, "No specific comment available.")


def generate_report(name, age, gender, xray_path, mask_path, disease, severity, output_path):
    comment = get_disease_comment(disease, severity)
    pdf = PDFReport()
    pdf.add_page()
    pdf.add_patient_info(name, age, gender)
    pdf.add_images(xray_path, mask_path)
    pdf.add_diagnosis(disease, severity, comment)
    pdf.output(output_path)
