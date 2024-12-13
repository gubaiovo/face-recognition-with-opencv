import threading
import cv2
import face_recognition

class FaceDetector:
    def __init__(self, camera_thread, detection_interval):
        self.camera_thread = camera_thread
        self.detection_interval = detection_interval
        self.frame_count = 0
        self.face_locations = []
        self.face_encodings = []
        self.kill_event = threading.Event()
        self.lock = threading.Lock()  # 添加锁
        self.detection_thread = threading.Thread(target=self.detect_faces)
        self.detection_thread.start()

    def detect_faces(self):
        while not self.kill_event.is_set():
            if self.frame_count % self.detection_interval == 0:
                frame = self.camera_thread.read()
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, self.face_locations)
                
                with self.lock:  # 使用锁保护共享数据
                    self.face_locations = face_locations
                    self.face_encodings = face_encodings
            self.frame_count += 1

    def stop(self):
        self.kill_event.set()
        self.detection_thread.join()

    def get_faces(self):
        with self.lock:  # 使用锁保护共享数据
            return self.face_locations, self.face_encodings