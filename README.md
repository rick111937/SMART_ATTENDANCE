# SMART_ATTENDANCE_SYSTEM
SMART ATTENDANCE SYSTEM 🎓
📌 Overview
The Smart Attendance System is a Python-based project that leverages OpenCV for face detection and recognition to automate the process of marking attendance. Instead of manual roll calls or paper registers, this system uses a camera to identify students/employees and record their presence in real-time.

🚀 Features
- Face Detection & Recognition using OpenCV and Haar Cascade/ LBPH algorithm.
- Automated Attendance Logging into CSV/Excel files.
- Real-time Camera Feed for capturing faces.
- Training Module to register new faces.
- Secure & Efficient – reduces proxy attendance and manual errors.

🛠️ Tech Stack
- Programming Language: Python
- Libraries: OpenCV, NumPy, Pandas
- Database/Storage: CSV/Excel files (can be extended to SQL/NoSQL)
- IDE: Visual Studio Code

 Project Structure:-
 SMART_ATTENDANCE_SYSTEM/
│
├── TrainingImage/          # Stores images used for training
├── TrainingImageLabel/     # Stores trained model files
├── Attendance/             # Attendance logs (CSV/Excel)
├── haarcascade_frontalface_default.xml  # Haar Cascade classifier
├── train.py                # Script to train the model
├── recognize.py            # Script to recognize faces & mark attendance
├── dataset.py              # Script to capture images for training
└── README.md               # Project documentation

📊 Output
- Attendance is automatically saved in the Attendance/ folder as CSV/Excel files.
- Each entry includes Name, ID, Date, and Time.

🔮 Future Enhancements
- Integration with SQL/Cloud Database.
- GUI Dashboard for attendance visualization.
- Email/SMS Notifications for absentees.
- Support for multi-camera setups.

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

📜 License
This project is licensed under the MIT License – feel free to use and modify.

👉 Pro tip: Since your repo has a large model file (>100 MB), don’t store it directly in GitHub. Instead, add instructions in the README for users to download the trained model from Google Drive/Dropbox and place it in TrainingImageLabel/.
