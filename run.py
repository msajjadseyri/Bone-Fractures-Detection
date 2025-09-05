import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

st.title("👋 Wellcome to AI Model of Bone Fractures Detection ")
st.badge('develop by @sajad seyri',icon="ℹ️",color='blue')
st.write(" Human Bone Fractures modal, is a collection of medical images (X-ray and MRI) focused on detecting bone fractures in various parts of the human body. It's designed to support research in computer vision and deep learning for medical applications.") 
st.write(" This AI modal develop by yolov8 medium size on 1537 image consist of Elbow, Finger, Forearm, Humerus, Shoulder, Femur, Shinbone, Knee, Hipbone, Wrist, Spinal cord, Some healthy bones and predict 10 class of bone status ")
st.caption('1.Comminuted 2.Greenstick 3.Healthy 4.Linear 5.Oblique Displaced 6.Oblique 7.Segmental 8.Spiral 9.Transverse Displaced 10.Transverse')

#define model
model = YOLO('./best.pt')
#choise image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
else:
    image = Image.open('img.jpg')
#run broken detction model
results = model(image)
img = image.copy()
np_img = np.array(image)
names=['Comminuted', 'Greenstick', 'Healthy', 'Linear', 'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral', 'Transverse Displaced', 'Transverse']
#draw rectangel and type brok class
for result in results:
    boxes = result.boxes.cpu().numpy()
    for i, box in enumerate(boxes):
      if (box.conf[0] > 0.5):
        r = box.xyxy[0].astype(int)
        cv2.rectangle(np_img,(r[0],r[1]),(r[2],r[3]),(255, 0, 0),1)
        txt=names[int(box.cls[0])] + '(' + str(int((box.conf[0])*100))+ "%)"
        cv2.putText(np_img,str(txt),(r[0]-20,r[1]-5), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1, cv2.LINE_AA)

#show result and orginal image
tab1, tab2 ,tab3 = st.tabs(["Detected result","Detected details", "Original image"])
tab1.image(np_img, use_container_width=True)
tab2.text('class of Bone Fractures Detection is ' + names[int(box.cls[0])] + ' and confidens persent of detection is ' + str(float(box.conf[0])*100) + ' ')
tab3.image(image, use_container_width=True)
