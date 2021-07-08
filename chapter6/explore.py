import cv2
import tensorflow as tf
import numpy as np
import time
import os


# TF Liteランタイムの初期化
interpreter = tf.lite.Interpreter(model_path='ssd.tflite')
interpreter.allocate_tensors()
# モデル情報の取得
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
target_height = input_details[0]['shape'][1]
target_width = input_details[0]['shape'][2]
# カメラへの接続
camera = cv2.VideoCapture(0)
time.sleep(2)

# クラスデータの用意
f = open('coco_labels.txt', 'r')
lines = f.readlines()
f.close()
classes = {}
for line in lines:
   pair = line.strip().split(maxsplit=1)
   classes[int(pair[0])] = pair[1].strip()


def detect(frame):
    height, width, _ = frame.shape
    resized = cv2.resize(frame, (target_width, target_height))
    input_data = np.expand_dims(resized, axis=0)
    # 画像データの入力
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # 推論実行
    interpreter.invoke()
    # 結果の取得
    detected_boxes = interpreter.get_tensor(output_details[0]['index'])
    detected_classes = interpreter.get_tensor(output_details[1]['index'])
    detected_scores = interpreter.get_tensor(output_details[2]['index'])
    num_boxes = interpreter.get_tensor(output_details[3]['index'])
    # バウンディングボックスの描画処理
    for i in range(int(num_boxes)):
        # 物体の位置を取得
        top, left, bottom, right = detected_boxes[0][i]
        # 物体のクラスを取得
        class_id = int(detected_classes[0][i])
        # 物体のスコア（尤度）を取得
        score = detected_scores[0][i]
        # スコアが50%を超えない物体は無視する
        if score < 0.5:
            continue
        # if class_id != 0:
            # continue
        # 位置座標を元の画像と同じ尺に戻す
        xmin = int(left * width)
        ymin = int(top * height)
        xmax = int(right * width)
        ymax = int(bottom * height)
        # クラスとスコアのタグを用意
        tag = '{}: {:.2f}%'.format("person", score * 100)
        # 画像に物体の位置を矩形として描画
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        # 矩形の近くにタグを描画
        cv2.putText(frame, tag, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
    return frame


while True:
    # フレームの読み込み
    ret, frame = camera.read()
    if not ret:
        continue
    # フレームに写った人の判別
    frame = detect(frame)
    # 人の位置を描画したフレームを画面に表示
    cv2.imshow("frame", frame)
    # 0キーが押されたら、繰り返しを終了する
    if cv2.waitKey(1)==0x30:
        break
    
camera.stop()
cv2.destroyAllWindows()