import cv2
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from PIL import ImageFont
from threading import Event, Thread
from camera import CameraThread
from FaceDetect import FaceDetector
import face_recognition
import pickle
from fps import FPS

choose_camera = 0 # 选择摄像头，0为内置摄像头，1为外置摄像头
min_matching_degree = 0.6 # 最小匹配度
detection_interval = 1 # 人脸检测间隔

# --------------------输出--------------------

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)): 
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(img)
    
    fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
    
    draw.text(position, text, textColor, font=fontStyle)
    
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# --------------------加载模型，初始化摄像头，初始化窗口--------------------


# 加载opencv Haar Cascade分类器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# GUI窗口
root = tk.Tk()
root.geometry('640x480')
root.title('人脸识别')

# 创建标签用于显示图像
image_label = tk.Label(root)
image_label.pack()

# 创建 PhotoImage 对象
photo = None

# 加载模型
with open('trained_model.pkl', 'rb') as f:
    person_encodings, person_names = pickle.load(f)

# 事件对象用于停止线程
kill_event = Event()

# 启动摄像头
camera_thread = CameraThread(kill_event, src=choose_camera, width=640, height=480)
camera_thread.start()

# 初始化FPS计算
fps_calculator = FPS()

# 初始化人脸检测器
face_detector = FaceDetector(camera_thread, detection_interval)

# --------------------打开摄像头，开始检测--------------------


# 处理捕获的图像
def update_frame():
    global photo
    frame = camera_thread.read()
    frame = cv2.flip(frame, 1)
    
    # 获取人脸位置和编码
    face_locations, face_encodings = face_detector.get_faces()

    # 在图像中框出检测到的人脸
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 检查人脸是否属于模型中的某个人
        matches = face_recognition.compare_faces(person_encodings, face_encoding) # 比较人脸编码
        face_distances = face_recognition.face_distance(person_encodings, face_encoding) # 计算距离
        best_match_index = np.argmin(face_distances) # 找到最小距离的索引
        name = "Unknown" # 默认为未知人脸
        matching_degree = 1 - face_distances[best_match_index] # 计算准确率
        print(f"Matching degree: {matching_degree:.2f}")
             
        if matches[best_match_index] and matching_degree > min_matching_degree:
            name = person_names[best_match_index]

        # 在图像中框出人脸并显示姓名
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
        frame = cv2AddChineseText(frame, name, (left + (right-left)//2 - 10, top - 30), (0, 255, 255), 30)

    # 计算并显示帧率
    fps = fps_calculator.update()
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 将图像转换为PIL Image格式
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    photo = ImageTk.PhotoImage(image)

    # 更新标签
    image_label.configure(image=photo)
    image_label.image = photo

    # 处理GUI事件，避免程序挂起
    root.after(10, update_frame)

# 图像更新循环
update_frame()

# 关闭程序
def on_closing():
    # 停止人脸检测线程
    face_detector.stop()
    # 停止摄像头线程
    kill_event.set()
    camera_thread.join()
    # 释放摄像头并关闭所有窗口
    camera_thread.stream.release()
    cv2.destroyAllWindows()
    # 关闭Tkinter窗口
    root.destroy()
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()