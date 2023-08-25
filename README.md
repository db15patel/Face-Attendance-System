# Attendance System with Face Recognition 👥📷

Welcome to the **Attendance System with Face Recognition** repository! This project serves as a Proof of Concept (POC) web application that showcases the application of facial recognition technology for efficient attendance management. By leveraging the power of facial recognition, this web application provides a streamlined solution for tracking attendance within an organization. Specifically designed for companies, it offers an integrated platform to manage employee attendance effectively.

## Table of Contents

- [Introduction](#introduction)
- [Functionality Supported](#functionality-supported)
- [Technologies Used](#technologies-used)
- [Recognition Process](#recognition-process)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction 📝

This repository houses a web application developed to demonstrate the feasibility of automating attendance management using facial recognition technology. It offers administrators the ability to register new employees, upload their photos for training the model, and subsequently track attendance efficiently. The system enhances employee experience by allowing them to view their personal attendance records.

## Functionality Supported 🚀

### Admin Panel 🛠️

- **Admin and Employee Login:** Secure login for administrators and employees.
- **Register New Employees:** Admins can register new employees within the system.
- **Add Employee Photos:** Admins can upload employee photos to train the facial recognition model.
- **Train the Model:** Admins can train the model to recognize registered employees.
- **Attendance Reports:** Admins can access detailed attendance reports with filtering options.

### Employee Panel 👤

- **View Personal Attendance Reports:** Employees can view their own attendance records.

## Technologies Used 🛠️

The project leverages various technologies for accurate and efficient face recognition:

- **OpenCV:** Open-source computer vision and machine learning library.
- **Dlib:** C++ library containing machine learning algorithms.
- **face_recognition:** A library by Adam Geitgey for simplified face recognition.
- **Django:** Python framework for web development.
- **scikit-learn:** Machine learning library for classification.

## Recognition Process 🧐

1. **Face Detection:** Dlib's HOG facial detector identifies faces within images.
2. **Facial Landmark Detection:** Dlib's 68-point shape predictor enhances recognition accuracy.
3. **Extraction of Facial Embeddings:** The **face_recognition** library extracts unique facial embeddings.
4. **Classification of Unknown Embeddings:** Linear SVM classifies unknown embeddings for recognition.

## Usage 📋

To run the web application locally:

1. Clone this repository.
2. Navigate to the project directory.
3. Install dependencies using `pip install -r requirements.txt`.
4. Configure settings and database using Django commands.
5. Run the development server: `python manage.py runserver`.

## Contributing 🤝

Contributions are welcome! If you have improvements or additional features to suggest, please follow these steps:

1. Fork this repository.
2. Create a new branch for your feature: `git checkout -b feature/new-feature`.
3. Add your changes.
4. Commit your changes: `git commit -m 'Add new feature'`.
5. Push to your branch: `git push origin feature/new-feature`.
6. Submit a pull request.

## License 📄

This project is open-source and available under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code for educational purposes.

With the **Attendance System with Face Recognition** repository, attendance management is reimagined through the power of facial recognition, offering a cutting-edge solution for modern organizations seeking efficient, accurate, and user-friendly attendance tracking.
