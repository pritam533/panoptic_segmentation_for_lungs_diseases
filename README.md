#  Panoptic Image Segmentation for Lung Disease Detection
This project was developed by students of VIT Bhopal as part of Project Exhibition 2 at VIT Bhopal University, under the Department of Computer Science and Engineering.

This project uses deep learning (U-Net) to segment lungs and detect lung diseases from X-ray images. It includes a web-based interface where users can upload an X-ray, view segmentation results, and get a downloadable PDF report with predictions.


## Project Objective
To develop an intelligent web-based platform that leverages deep learning (U-Net architecture) for panoptic segmentation and classification of lung diseases (COVID-19, Pneumonia, Lung Opacity) from chest X-ray images. The system provides:

Real-time segmentation output
Disease prediction with severity estimation
AI-generated doctor-style comments
A downloadable PDF report 

## Purpose
The objective of this project is to leverage deep learning for early and accurate detection of lung diseases like COVID-19, Pneumonia, and Lung Opacity through X-ray images using panoptic segmentation techniques. The project combines medical AI, web application development, and report automation to provide a complete diagnostic support system.

## Formats
    - All the images are in Portable Network Graphics (PNG) file format and resolution are 299*299 pixels.

## Features

- Upload Chest X-ray images (via UI)
- Automatic lung & disease-affected region segmentation using U-Net
- Disease classification and severity estimation
- PDF Report generation (with image, result, and patient info)
- Email report to the patient (optional)
- Manual segmentation feature (optional)
- Full Flask + HTML + JS (AJAX) stack


## Model Info
Architecture: U-Net

Input: Chest X-ray images (typically 256x256 or 512x512)
Output: Segmentation mask (disease regions)
Dataset Used: COVID-19 Radiography Dataset from Kaggle

## Future Improvements
Panoptic segmentation with instance + semantic labeling
More advanced disease classification (multi-label)
Responsive frontend dashboard
Live segmentation feedback

## Credits
U-Net architecture: Olaf Ronneberger et al.
COVID dataset: Kaggle contributors
PDF & Email: ReportLab, pdfkit, smtplib

## TEAM MEMBER 
1. PREETAM VERMA 22MIM10115 (LEAD THE PROJECT)
2. VIPIN VERMA 22MIM10118
3. VISHAL VERMA 22MIM10126
4. AYUSH VERMA 22MIM10123
5. DEVENDRA VERMA 22MIM10108

## Team Members & Contribution

## Name	        Roll          Number	          Contribution
Preetam Verma	22MIM10115	  Team Lead,          Model Integration, Backend & Deployment, Testing & Debugging
Vipin Verma	    22MIM10118	  Data Preprocessing, Model Training
Vishal Verma	22MIM10126	  UI/UX Design,       Frontend Development
Ayush Verma	    22MIM10123	  Data Preprocessing, Model Training
Devendra Verma	22MIM10108	  Report Generation,  Testing & Debugging

## Faculty Guidance
The project was carried out under the guidance of 
Supervisor: Mrs. Garima Jain
Reviewers: Dr. Harshlata Vishwakarma, Dr. Priscilla Dinkar Moyya



##  ABOUT dataset 

COVID-19 CHEST X-RAY DATABASE

A team of researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have created a database of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images. This COVID-19, normal and other lung infection dataset is released in stages. In the first release we have released 219 COVID-19, 1341 normal and 1345 viral pneumonia chest X-ray (CXR) images. In the first update, we have increased the COVID-19 class to 1200 CXR images. In the 2nd update, we have increased the database to 3616 COVID-19 positive cases along with 10,192 Normal, 6012 Lung Opacity (Non-COVID lung infection) and 1345 Viral Pneumonia images and corresponding lung masks. We will continue to update this database as soon as we have new x-ray images for COVID-19 pneumonia patients.  


COVID-19 data:
-----------------------
COVID data are collected from different publicly accessible dataset, online sources and published papers.
-2473 CXR images are collected from padchest dataset[1].
-183 CXR images from a Germany medical school[2].
-559 CXR image from SIRM, Github, Kaggle & Tweeter[3,4,5,6]
-400 CXR images from another Github source[7].


Normal images:
---------------------------------------- 
10192 Normal data are collected from from three different dataset.
-8851 RSNA [8]
-1341 Kaggle [9]


Lung opacity images:
---------------------------------------- 
6012 Lung opacity CXR images are collected from Radiological Society of North America (RSNA) CXR dataset  [8]

Viral Pneumonia images:
---------------------------------------- 
1345 Viral Pneumonia data are collected from  the Chest X-Ray Images (pneumonia) database [9]

Please cite the follwoing two articles if you are using this dataset:
-M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.
-Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. arXiv preprint arXiv:2012.02238.

Reference:
[1]https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590858128006-9e640421-6711
[2]https://github.com/ml-workgroup/covid-19-image-repository/tree/master/png
[3]https://sirm.org/category/senza-categoria/covid-19/
[4]https://eurorad.org
[5]https://github.com/ieee8023/covid-chestxray-dataset
[6]https://figshare.com/articles/COVID-19_Chest_X-Ray_Image_Repository/12580328
[7]https://github.com/armiro/COVID-CXNet  
[8]https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
[9] https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

