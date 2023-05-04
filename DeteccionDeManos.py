import cv2
import mediapipe as mp
import os
os.system("cls")

video = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands # Modulo para los puntitos
mp_drawing = mp.solutions.drawing_utils #Módulo para las conexiones
#Determinamos la precisión de la detección de la mano y el rastreo de la misma
hands = mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5)
def drawHandLandMarks (img,hand_landmarks):
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(img,landmarks,mp_hands.HAND_CONNECTIONS)# Dibuja las conexiones sobre la mano
    
while True:
    success, img = video.read()# Propiedades del video
    img = cv2.flip(img,1) # Toma la imagen y la dirección, para voltear verticalmente la cámara
    result = hands.process(img)
    hand_landmarks = result.multi_hand_landmarks
    drawHandLandMarks(img,hand_landmarks)
    print (hand_landmarks)
    cv2.imshow("Mano",img)
    if cv2.waitKey(4) == 32:
        break
video.release()
