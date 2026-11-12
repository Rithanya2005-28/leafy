# PRODUCT REQUIREMENTS DOCUMENT (PRD)

# Project: Basic Plant Leaf Classification System

**Project Type:** AI/ML Web Application  
**Project Level:** Beginner-Friendly  
**Development Environment:** Antigravity  
**Primary Language:** Python

---

## 1. Project Overview

The Basic Plant Leaf Classification System is a web-based application that identifies different types of plant leaves from uploaded images using Deep Learning.

Users can upload a leaf image through a simple website. The system processes the image, extracts useful visual information and predicts the plant species using a Convolutional Neural Network (CNN).

The system also includes basic compiler concepts, client-server communication, image processing and cybersecurity features.

## 2. Problem Statement

Identifying different plant species manually can be difficult, especially for beginners and people unfamiliar with plant varieties.

This project aims to develop a simple automated system that identifies plant leaves from images and displays the predicted plant name.

## 3. Project Objectives

- Identify different plant leaf types using images.
- Apply digital image processing techniques.
- Develop a CNN-based classification model.
- Implement basic compiler design concepts.
- Establish communication between frontend and backend.
- Protect uploaded images using basic security techniques.

## 4. Plant Categories

Initially, support these five plant categories:

1. Mango
2. Neem
3. Guava
4. Tulsi
5. Banana

The system should allow additional categories to be added later.

Use a properly labelled image dataset for training and testing. Do not generate fake predictions or fake accuracy values.

## 5. Functional Requirements

### FR1: User Interface

Create a simple and attractive web application.

Features:
- Home page
- Upload leaf image option
- Image preview
- Classify button
- Prediction result
- Confidence score
- Reset button

### FR2: Image Upload

- Accept JPG, JPEG and PNG images.
- Validate file type and size.
- Display the uploaded image.
- Show an error message for invalid files.

### FR3: Image Processing (Digital Image Processing)

Use OpenCV to perform:

- Image acquisition
- Image resizing (224 × 224)
- RGB colour conversion
- Noise reduction using Gaussian Blur
- Image normalization
- Display the processed image

### FR4: Deep Learning Classification

Develop a CNN model using TensorFlow/Keras.

The model should:

- Accept preprocessed leaf images.
- Extract image features.
- Classify the leaf into one of the five plant categories.
- Display the predicted plant name.
- Display the model's confidence score.
- Handle low-confidence predictions appropriately.

Split the dataset into training, validation and testing sets.

Display actual model evaluation metrics such as accuracy and a confusion matrix.

### FR5: Compiler Design Integration

Implement a small rule-based command parser for leaf classification.

Example command:

`CLASSIFY mango_leaf.jpg`

Implement the following compiler concepts:

1. Lexical Analysis – Identify command tokens.
2. Syntax Analysis – Check the command structure.
3. Semantic Analysis – Validate the command and image filename.
4. Execution – Pass the valid image to the classification module.

Show meaningful error messages for invalid commands.

This module should be simple and should not require developing a complete programming language.

### FR6: Computer Networks Integration

Implement client-server communication.

- Frontend sends uploaded images to the Flask backend.
- Backend receives images through HTTP POST requests.
- REST API handles classification requests.
- Backend sends prediction results to the frontend.
- Display server responses and error messages.

Use Flask for backend communication.

### FR7: Cryptography and Cybersecurity Integration

Implement basic security features:

1. Use SHA-256 hashing to generate a hash of the uploaded image.
2. Verify image integrity using the generated hash.
3. Validate uploaded file extensions and MIME types.
4. Restrict file upload size.
5. Prevent unsafe file paths and filenames.
6. Use secure password hashing if user authentication is implemented.

Do not store passwords as plain text.

## 6. Technology Stack

**Frontend**
- HTML
- CSS
- JavaScript
- Bootstrap (optional)

**Backend**
- Python
- Flask

**Machine Learning**
- TensorFlow
- Keras
- CNN

**Image Processing**
- OpenCV
- NumPy

**Security**
- hashlib
- SHA-256
- Werkzeug password hashing (if login is implemented)

**Database**
- SQLite (optional, for storing classification history)

## 7. System Architecture

The application should follow this workflow:

1. User opens the website.
2. User uploads a plant leaf image.
3. Frontend sends the image to Flask backend through HTTP.
4. Backend validates the uploaded image.
5. SHA-256 hash is generated for integrity verification.
6. OpenCV preprocesses the image.
7. CNN model predicts the plant category.
8. Compiler module validates classification commands when used.
9. Backend returns the prediction.
10. Frontend displays the plant name and confidence score.

## 8. Expected Output

For example:

**Input:** Mango leaf image

**Output:**
- Predicted Plant: Mango
- Confidence Score: Actual model output
- Image Status: Successfully Processed
- Image Integrity: Verified

Include the uploaded image alongside the result.

## 9. Suggested Project Folder Structure

```text
Plant-Leaf-Classification/
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── model/
│   └── leaf_classifier.keras
│
├── dataset/
│   ├── Mango/
│   ├── Neem/
│   ├── Guava/
│   ├── Tulsi/
│   └── Banana/
│
├── compiler/
│   └── command_parser.py
│
├── security/
│   └── image_security.py
│
├── image_processing/
│   └── preprocessing.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
└── uploads/
```

## 10. User Interface Design

Create a clean, modern and beginner-friendly interface.

Design requirements:

- Nature-inspired green and white colour theme.
- Simple navigation bar.
- Large project title: "Plant Leaf Classification System".
- Upload image section in the centre.
- Image preview before classification.
- Green classification button.
- Attractive result card.
- Display predicted plant name and confidence.
- Responsive layout for desktop and mobile.
- Clear error messages.

Avoid unnecessary animations and complicated dashboard features.

## 11. Non-Functional Requirements

- Easy to use.
- Fast image processing.
- Accurate predictions based on the trained model.
- Secure image upload.
- Maintainable Python code.
- Clear separation of frontend and backend.
- Proper error handling.
- Easy to run using VS Code.

## 12. Expected Deliverables

1. Complete working web application.
2. Python Flask backend.
3. CNN model and training code.
4. OpenCV image preprocessing module.
5. Basic compiler command parser.
6. SHA-256 image integrity verification.
7. Frontend using HTML, CSS and JavaScript.
8. Requirements file.
9. README with installation and execution instructions.
10. Project screenshots and testing results.

## 13. Implementation Instructions for Antigravity

Follow these instructions carefully:

1. First, create the complete project folder structure.
2. Develop the frontend and backend separately.
3. Implement each of the five subject-related modules.
4. Connect all modules into one working application.
5. Use simple and understandable Python code.
6. Add comments wherever necessary.
7. Do not create unnecessary advanced features.
8. Do not use dummy predictions or hardcoded accuracy.
9. If the dataset or trained model is missing, explain how to provide or train it.
10. Test all major features and fix errors before completing the project.
11. Provide exact terminal commands to install dependencies and run the application.
12. Do not overwrite or delete existing user files without permission.

## 14. Final Expected Result

A fully functional Plant Leaf Classification System that identifies five different plant leaf categories from uploaded images.

The application must demonstrate the practical implementation of:

- Compiler Design
- Computer Networks
- Digital Image Processing
- Deep Learning
- Cryptography and Cybersecurity

**The final application should be simple, functional, easy to demonstrate and suitable for an undergraduate engineering semester project.**