import os
import face_recognition
import pickle

person_encodings = []
person_names = []
for filename in os.listdir('person'):
    if filename.endswith('.jpg'):
        # 编码图像
        image = face_recognition.load_image_file(os.path.join('person', filename))
        encodings = face_recognition.face_encodings(image)
        if encodings:
            encoding = encodings[0]
            person_encodings.append(encoding)
            # 文件名处理
            person_name = ''.join([i for i in os.path.splitext(filename)[0] if not i.isdigit()])
            person_names.append(person_name)

# 输出
with open('trained_model.pkl', 'wb') as f:
    pickle.dump((person_encodings, person_names), f)