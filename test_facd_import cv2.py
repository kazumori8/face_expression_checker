import cv2
import dlib

# Dlibの顔検出器を初期化する
detector = dlib.get_frontal_face_detector()

# Dlibのランドマーク検出器を初期化する
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor('C:/dlib/python_examples/shape_predictor_68_face_landmarks.dat')
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

        # 口の角の座標を取得する
        mouth_left = landmarks.part(48).x
        mouth_right = landmarks.part(54).x
        mouth_top = landmarks.part(51).y
        mouth_bottom = landmarks.part(57).y

        # 口の角の位置を計算する
        mouth_width = mouth_right - mouth_left
        mouth_height = mouth_bottom - mouth_top
        mouth_ratio = mouth_width / mouth_height 

        # 口の角が下がっている場合には警告を表示する
        if mouth_ratio < 1.01:
            print("Warning: Mouth corners down")
            cv2.putText(frame, "Warning: Mouth corners down", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            print("OK: Mouth corners up Now")

    # フレームを表示する
    cv2.imshow("frame", frame)

    # キー入力を待つ
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラを解放する
cap.release()

# ウィンドウをすべて閉じる
cv2.destroyAllWindows()
