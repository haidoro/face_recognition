# -*- coding: utf-8 -*-
import sys
import face_recognition
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import glob

threshold = 0.5

"""顔情報の初期化"""

face_locations = []
face_encodings = []

"""登録画像の読み込み"""

image_paths = glob.glob('image/*')
image_paths.sort()
known_face_encodings = []
known_face_names = []
checked_face = []

delimiter = "/" # Mac / Linux用

for image_path in image_paths:
    im_name = image_path.split(delimiter)[-1].split('.')[0]
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(im_name)

video_capture = cv2.VideoCapture(0)

def main():

    while True:
        # ビデオの単一フレームを取得
        _, frame = video_capture.read()

        # 顔の位置情報を検索
        face_locations = face_recognition.face_locations(frame)
        # 顔画像の符号化
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            # 顔画像が登録画像と一致しているか検証
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, threshold)
            name = "Unknown"

            # 顔画像と最も近い登録画像を候補とする
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # 位置情報の表示
        for (top, right, bottom, left) in face_locations:
            # 顔領域に枠を描画
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # 顔領域の下に枠を表示
            cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # 結果をビデオに表示
        cv2.imshow('Video', frame)

        # ESCキーで終了
        if cv2.waitKey(1) == 27:
            break

main()

video_capture.release()
cv2.destroyAllWindows()