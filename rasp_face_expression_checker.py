import cv2
import dlib
from imutils import face_utils
# Dlibの顔検出器を初期化する
detector = dlib.get_frontal_face_detector()

# Dlibのランドマーク検出器を初期化する
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# カメラを初期化する
cap = cv2.VideoCapture(0)

# ループを開始する
while True:
    # カメラからフレームを取得する
    ret, frame = cap.read()

    # グレースケールに変換する
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔を検出する
    faces = detector(gray)

    # 各顔に対してループを実行する
    for face in faces:
        # 顔のランドマークを検出する
        landmarks = predictor(gray, face)
        landmark = face_utils.shape_to_np(landmarks)

        # ランドマーク描画
        for (x, y) in landmark:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # 口の角の座標を取得する
        mouth_left_x = landmarks.part(48).x
        mouth_left_y = landmarks.part(48).y
        mouth_right_x = landmarks.part(54).x
        mouth_right_y = landmarks.part(54).y
        mouth_top_x= landmarks.part(51).x
        mouth_top_y = landmarks.part(51).y
        mouth_bottom_x = landmarks.part(57).x
        mouth_bottom_y = landmarks.part(57).y
        mouth_perleche_y =  (mouth_left_y+mouth_right_y)/2
        mouth_perleche_x = (mouth_left_x+ mouth_right_x)/2
        mouth_center_height = (mouth_top_y+mouth_bottom_y)/2
        center_x =  (landmarks.part(62).x+ landmarks.part(66).x)/2.0
        center_y =  (landmarks.part(62).y+ landmarks.part(66).y)/2.0
        eye_right_down = landmarks.part(41).y
        eye_right_up = landmarks.part(37).y
        eye_left_down = landmarks.part(46).y
        eye_left_up = landmarks.part(44).y
        cv2.circle(frame, (landmarks.part(48).x,landmarks.part(48).y), 2, (0, 255, 255), -1)
        cv2.circle(frame, (100, 200), 2, (0, 255, 255), -1)
        cv2.circle(frame, (200, 200), 2, (0, 255, 255), -1)
        cv2.circle(frame, (200, 300), 2, (0, 255, 255), -1)
        cv2.circle(frame, (landmarks.part(54).x,landmarks.part(54).y), 2, (0, 255, 255), -1)
#        cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), -1)
        # 口の角の位置を計算する
        mouth_ratio = mouth_perleche_y - center_y
        #Calculate the degree of eye opening
        eye_ratio_right = eye_right_up-eye_right_down
        eye_ratio_left = eye_left_up-eye_left_down
        eye_ratio = (eye_ratio_right+ eye_ratio_left)/2
        print("eye_ratio:" , eye_ratio)
        print("mouth_ratio", mouth_ratio, " mouth_perleche_y",mouth_perleche_y ,"center_y", center_y, "center_x", center_x)

        # 口の角が下がっている場合には警告を表示する
        if mouth_ratio >  -1.5:
            print("Warning: perleche down  (", mouth_perleche_y,",", center_y, ")")
            cv2.putText(frame, "Warning: Mouth corners down", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            print("OK: Mouth corners up Now(", mouth_perleche_y,",", center_y, ")")
        if eye_ratio >-9.0:
            print("Warning: Don't Open eyes")
            cv2.putText(frame, "Warning: don't open eyes", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # フレームを表示する
    cv2.imshow("frame", frame)

    # キー入力を待つ
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラを解放する
cap.release()

# ウィンドウをすべて閉じる
cv2.destroyAllWindows()
