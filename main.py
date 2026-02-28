############################################# IMPORTING ################################################
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2, os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

############################################# HELPERS ##################################################

def assure_path_exists(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)

def check_opencv_face_module():
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        mess.showerror(
            "OpenCV Error",
            "cv2.face module not found.\nInstall opencv-contrib-python:\n\npip install opencv-contrib-python"
        )
        return False
    return True

##################################################################################

def check_haarcascadefile():
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if not exists:
        mess.showerror('Missing file', 'haarcascade_frontalface_default.xml not found.\nPlace it beside this script.')
        window.destroy()

###################################################################################

def save_pass():
    assure_path_exists(os.path.join("TrainingImageLabel", "psd.txt"))
    psd_path = os.path.join("TrainingImageLabel", "psd.txt")
    if os.path.isfile(psd_path):
        with open(psd_path, "r") as tf:
            key = tf.read()
    else:
        try:
            master.destroy()
        except Exception:
            pass
        new_pas = tsd.askstring('Password setup', 'Create a new password', show='*')
        if new_pas is None or new_pas.strip() == "":
            mess.showerror('No Password', 'Password not set. Try again.')
        else:
            with open(psd_path, "w") as tf:
                tf.write(new_pas)
            mess.showinfo('Password Registered', 'New password registered successfully!')
        return

    op = old.get()
    newp = new.get()
    nnewp = nnew.get()
    if op == key:
        if newp == nnewp and newp.strip() != "":
            with open(psd_path, "w") as txf:
                txf.write(newp)
            mess.showinfo('Password Changed', 'Password changed successfully!')
            master.destroy()
        else:
            mess.showerror('Error', 'New passwords do not match (or empty).')
    else:
        mess.showerror('Wrong Password', 'Please enter the correct old password.')

###################################################################################

def change_pass():
    global master, old, new, nnew
    master = tk.Toplevel()
    master.geometry("400x160")
    master.resizable(False, False)
    master.title("Change Password")
    master.configure(background="white")

    tk.Label(master, text='    Enter Old Password', bg='white', font=('times', 12, ' bold ')).place(x=10, y=10)
    old = tk.Entry(master, width=25, fg="black", relief='solid', font=('times', 12, ' bold '), show='*')
    old.place(x=180, y=10)

    tk.Label(master, text='   Enter New Password', bg='white', font=('times', 12, ' bold ')).place(x=10, y=45)
    new = tk.Entry(master, width=25, fg="black", relief='solid', font=('times', 12, ' bold '), show='*')
    new.place(x=180, y=45)

    tk.Label(master, text='Confirm New Password', bg='white', font=('times', 12, ' bold ')).place(x=10, y=80)
    nnew = tk.Entry(master, width=25, fg="black", relief='solid', font=('times', 12, ' bold '), show='*')
    nnew.place(x=180, y=80)

    tk.Button(master, text="Save", command=save_pass, fg="black", bg="#3ece48",
              height=1, width=25, activebackground="white", font=('times', 10, ' bold ')).place(x=10, y=120)
    tk.Button(master, text="Cancel", command=master.destroy, fg="black", bg="red",
              height=1, width=25, activebackground="white", font=('times', 10, ' bold ')).place(x=200, y=120)

#####################################################################################

def psw():
    psd_path = os.path.join("TrainingImageLabel", "psd.txt")
    assure_path_exists(psd_path)
    if os.path.isfile(psd_path):
        with open(psd_path, "r") as tf:
            key = tf.read()
    else:
        new_pas = tsd.askstring('Password setup', 'Create a new password', show='*')
        if new_pas is None or new_pas.strip() == "":
            mess.showerror('No Password', 'Password not set. Try again.')
        else:
            with open(psd_path, "w") as tf:
                tf.write(new_pas)
            mess.showinfo('Password Registered', 'New password registered successfully!')
        return

    password = tsd.askstring('Password', 'Enter Password', show='*')
    if password is None:
        return
    if password == key:
        TrainImages()
    else:
        mess.showerror('Wrong Password', 'You have entered a wrong password.')

######################################################################################

def TakeImages():
    if not check_opencv_face_module():
        return
    check_haarcascadefile()

    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    details_csv = os.path.join("StudentDetails", "StudentDetails.csv")
    assure_path_exists(details_csv)
    assure_path_exists(os.path.join("TrainingImage", ""))

    serial = 1
    if os.path.isfile(details_csv):
        try:
            df_ser = pd.read_csv(details_csv)
            if 'SERIAL NO.' in df_ser.columns and not df_ser.empty:
                existing = df_ser['SERIAL NO.'].dropna()
                if not existing.empty:
                    serial = int(existing.astype(int).max()) + 1
        except Exception:
            with open(details_csv, 'r', newline='') as f:
                r = csv.reader(f)
                line_count = sum(1 for _ in r)
                serial = max(1, (line_count // 2))
    else:
        with open(details_csv, 'a', newline='') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
        serial = 1

    Id = entry_id.get().strip()
    name = entry_name.get().strip()

    if not Id or not name:
        mess.showerror("Input Error", "Both ID and Name are required.")
        return

    if not (name.replace(" ", "").isalpha()):
        mess.showerror("Input Error", "Name must contain only letters.")
        return

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        mess.showerror("Camera Error", "Cannot access the camera.")
        return

    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    sampleNum = 0

    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sampleNum += 1
            out_path = os.path.join(
                "TrainingImage",
                f"{name}.{serial}.{Id}.{sampleNum}.jpg"
            )
            cv2.imwrite(out_path, gray[y:y + h, x:x + w])
            cv2.imshow('Taking Images - press q to stop', img)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sampleNum >= 100:
            break

    cam.release()
    cv2.destroyAllWindows()

    with open(details_csv, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([serial, '', Id, '', name])

    status_label.configure(text=f"✅ Images Taken for ID : {Id}")
    try:
        df_count = pd.read_csv(details_csv)
        total = len(df_count['SERIAL NO.'].dropna())
        total_label.configure(text='Total Registrations: ' + str(total))
    except Exception:
        pass

########################################################################################

def getImagesAndLabels(path):
    if not os.path.isdir(path):
        return [], []
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    faces, labels = [], []
    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            parts = os.path.basename(imagePath).split(".")
            serial_label = int(parts[1])
            faces.append(imageNp)
            labels.append(serial_label)
        except Exception:
            continue
    return faces, labels

###########################################################################################

def TrainImages():
    if not check_opencv_face_module():
        return
    check_haarcascadefile()
    assure_path_exists(os.path.join("TrainingImageLabel", "Trainner.yml"))

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = getImagesAndLabels("TrainingImage")

    if not faces or not labels:
        mess.showerror('No Registrations', 'Please register someone first!')
        return

    try:
        recognizer.train(faces, np.array(labels))
    except Exception as e:
        mess.showerror('Training Error', f'Failed to train recognizer:\n{e}')
        return

    recognizer.save(os.path.join("TrainingImageLabel", "Trainner.yml"))
    status_label.configure(text="✅ Profile Saved Successfully")
    total_label.configure(text=f"Total Registrations: {len(set(labels))}")

############################################################################################

def TrackImages():
    if not check_opencv_face_module():
        return
    check_haarcascadefile()
    assure_path_exists(os.path.join("Attendance", ""))

    for k in tv.get_children():
        tv.delete(k)

    model_path = os.path.join("TrainingImageLabel", "Trainner.yml")
    if not os.path.isfile(model_path):
        mess.showerror('Data Missing', 'Please click on Save Profile to create the model first.')
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    details_csv = os.path.join("StudentDetails", "StudentDetails.csv")
    if not os.path.isfile(details_csv):
        mess.showerror('Details Missing', 'Students details are missing, please check!')
        return

    try:
        df = pd.read_csv(details_csv)
    except Exception:
        mess.showerror('CSV Error', 'Could not read StudentDetails.csv')
        return

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        mess.showerror("Camera Error", "Cannot access the camera.")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']
    attendance = None

    while True:
        ret, im = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        bb = "Unknown"

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                try:
                    match = df.loc[df['SERIAL NO.'].astype(int) == int(serial)]
                except Exception:
                    match = df.loc[df['SERIAL NO.'] == serial]

                if not match.empty:
                    name_val = match['NAME'].values
                    id_val = match['ID'].values
                    ID_str = str(id_val)[1:-1]
                    bb = str(name_val)[2:-2]
                    attendance = [str(ID_str), '', bb, '', str(date), '', str(timeStamp)]
                else:
                    bb = "Unknown"
            else:
                bb = "Unknown"

            cv2.putText(im, str(bb), (x, y + h), font, 1, (255, 255, 255), 2)

        cv2.imshow('Taking Attendance - press q to save/exit', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    att_path = os.path.join("Attendance", f"Attendance_{date}.csv")
    file_exists = os.path.isfile(att_path)

    with open(att_path, 'a', newline='') as csvFile1:
        writer = csv.writer(csvFile1)
        if not file_exists:
            writer.writerow(col_names)
        if attendance:
            writer.writerow(attendance)

    try:
        with open(att_path, 'r', newline='') as csvFile1:
            reader1 = csv.reader(csvFile1)
            i = 0
            for lines in reader1:
                i += 1
                if i > 1:
                    try:
                        iidd = str(lines[0]) + '   '
                        tv.insert('', 0, text=iidd, values=(str(lines[2]), str(lines[4]), str(lines[6])))
                    except Exception:
                        continue
    except Exception:
        pass

######################################## GUI FRONT-END ###########################################

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

window = ctk.CTk()
window.title("Face Recognition Attendance System")
window.geometry("1280x720")
window.minsize(1000, 600)

header_frame = ctk.CTkFrame(window, corner_radius=15)
header_frame.pack(fill="x", padx=20, pady=10)

title = ctk.CTkLabel(header_frame, text="📷 Face Recognition Attendance System",
                     font=("Helvetica", 30, "bold"))
title.pack(pady=10)

info_frame = ctk.CTkFrame(window, corner_radius=15)
info_frame.pack(fill="x", padx=20, pady=(0, 10))

now = datetime.datetime.now()
date_label = ctk.CTkLabel(info_frame, text=f"📅 {now.strftime('%d %B %Y')}",
                          font=("Helvetica", 18, "bold"), text_color="orange")
date_label.pack(side="left", padx=20, pady=10)

clock_label = ctk.CTkLabel(info_frame, text="", font=("Helvetica", 18, "bold"), text_color="orange")
clock_label.pack(side="right", padx=20, pady=10)

def update_clock():
    clock_label.configure(text="⏰ " + datetime.datetime.now().strftime("%H:%M:%S"))
    window.after(1000, update_clock)

update_clock()

body_frame = ctk.CTkFrame(window, fg_color="transparent")
body_frame.pack(fill="both", expand=True, padx=20, pady=10)

left_frame = ctk.CTkFrame(body_frame, corner_radius=15)
left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

left_title = ctk.CTkLabel(left_frame, text="📋 Attendance Records",
                          font=("Helvetica", 20, "bold"))
left_title.pack(pady=10)

style = ttk.Style()
style.theme_use("clam")
style.configure("Treeview", background="#1e1e1e", foreground="white",
                rowheight=28, fieldbackground="#1e1e1e", font=("Helvetica", 12))
style.configure("Treeview.Heading", font=("Helvetica", 13, "bold"))

tv = ttk.Treeview(left_frame, height=12, columns=("name", "date", "time"))
tv.heading("#0", text="ID")
tv.heading("name", text="Name")
tv.heading("date", text="Date")
tv.heading("time", text="Time")
tv.column("#0", width=80)
tv.column("name", width=150)
tv.column("date", width=120)
tv.column("time", width=100)
tv.pack(fill="both", expand=True, padx=10, pady=10)

right_frame = ctk.CTkFrame(body_frame, corner_radius=15)
right_frame.pack(side="right", fill="both", expand=True)

right_title = ctk.CTkLabel(right_frame, text="👤 New Registration",
                           font=("Helvetica", 20, "bold"))
right_title.pack(pady=10)

entry_id = ctk.CTkEntry(right_frame, placeholder_text="Enter Student ID")
entry_id.pack(pady=10, padx=20)

entry_name = ctk.CTkEntry(right_frame, placeholder_text="Enter Student Name")
entry_name.pack(pady=10, padx=20)

status_label = ctk.CTkLabel(right_frame, text="1️⃣ Take Images → 2️⃣ Save Profile",
                            font=("Helvetica", 14))
status_label.pack(pady=10)

total_label = ctk.CTkLabel(right_frame, text="Total Registrations: 0",
                           font=("Helvetica", 14, "bold"))
total_label.pack(pady=5)

btn_take_images = ctk.CTkButton(right_frame, text="📸 Take Images", width=200, height=40, command=TakeImages)
btn_take_images.pack(pady=8)

btn_save_profile = ctk.CTkButton(right_frame, text="💾 Save Profile", width=200, height=40, command=psw)
btn_save_profile.pack(pady=8)

btn_attendance = ctk.CTkButton(right_frame, text="✅ Take Attendance", width=200, height=40, command=TrackImages)
btn_attendance.pack(pady=8)

btn_change_pass = ctk.CTkButton(right_frame, text="🔑 Change Password", width=200, height=40, command=change_pass)
btn_change_pass.pack(pady=8)

btn_quit = ctk.CTkButton(right_frame, text="🚪 Quit", width=200, height=40, fg_color="red", command=window.destroy)
btn_quit.pack(pady=8)

window.mainloop()
