import requests
import cv2
import tflite_runtime.interpreter as tflite
import numpy as np

import time
import pickle
import os
from datetime import datetime, date, timedelta


### プログラム実行中に変化しない値をまとめて定義
DATA_PATH = 'data.pickle' # データの保存先
CHECK_SECONDS = 60 # 検知を行う間隔（秒）
TARGET_HOUR=7 # 集計を行う時
TARGET_MINUTE=30 # 集計を行う分
LINE_USER_ID='USER_ID' # ユーザーID
LINE_ACCESS_TOKEN='ACCESS_TOKEN' # チャネルアクセストークン

### 集計を行う時刻を設定する
# 現在時刻の取得
current_time = datetime.now()
# 現在時刻を元に、集計を行う時刻を設定する
target_time = current_time.replace(
    hour=TARGET_HOUR,
    minute=TARGET_MINUTE
)
# 集計時刻がすでに過ぎている場合に実行
if target_time < current_time:
    # 集計時刻を1日後の同じ時間に設定する
    target_time = target_time + timedelta(days=1)
print('target time:', target_time.isoformat())

### データの保存先を用意する
# まだ保存先のファイルが無い場合のみ実行する
if not os.path.exists(DATA_PATH):
    # 新しいファイルを作成して開く
    with open(DATA_PATH, 'wb') as f:
        # 空のリストをファイルに書き込み
        pickle.dump([], f)


# TF Liteランタイムの初期化
interpreter = tflite.Interpreter(model_path='ssd.tflite')
interpreter.allocate_tensors()
# モデル情報の取得
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
target_height = input_details[0]['shape'][1]
target_width = input_details[0]['shape'][2]
# カメラへの接続
camera = cv2.VideoCapture(0)
time.sleep(2)


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
    
    # 1枚のフレーム内で検知した人の数
    num_persons = 0  
    # 物体の描画処理
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
        if class_id != 0:
            continue
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
        cv2.putText(frame, tag, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # 検知した人の数を1増やす
        num_persons += 1

    return frame, num_persons

# 検知した人数と時刻をファイルに保存する関数
def save(num_persons):
    items = []
    # ファイルに保存されている検知結果の一覧を読み込む
    with open(DATA_PATH, 'rb') as f:
        items = pickle.load(f)
    # 新しい検知結果を一覧の先頭に加える
    items.insert(0, {
        "count": num_persons,
        "created_at": datetime.now().isoformat()
    })
    # 更新された一覧をファイルに保存する
    with open(DATA_PATH, 'wb') as f:
        pickle.dump(items, f)
        
# 滞在時間を秒数で集計する関数
def get_stay_seconds(start_time, end_time):
    items = []
    total_seconds = 0
    # ファイルに保存されている検知結果の一覧を読み込む
    with open(DATA_PATH, 'rb') as f:
        items = pickle.load(f)
    # 検知結果を１つずつ処理する
    for item in items:
        # 検知結果を記録した時刻を取得
        created_at = datetime.fromisoformat(item['created_at'])
        # 検知結果が開始時刻以前、または終了時刻以降なら繰り返しを終了する
        if created_at < start_time or created_at > end_time:
            break
        # 人数が1人以上なら滞在時間を増やす
        if item['count'] > 0:
            total_seconds += CHECK_SECONDS
    # 滞在時間の合計を返す
    return total_seconds

# 滞在時間をLINEに通知する関数
def send_message(seconds):
    hour = seconds // 3600 # 時を算出する
    minute = (seconds % 3600) // 60 # 分を算出する
    # LINEに通知するメッセージを作成
    message = 'stay time is {}hour {}min'.format(hour, minute)
    
    try:
        # HTTPリクエストの送信
        requests.post(
            url="https://api.line.me/v2/bot/message/push",
            headers={'Authorization': 'Bearer {}'.format(LINE_ACCESS_TOKEN)},
            json={
                "to": LINE_USER_ID,
                "messages": [{
                    "type": "text",
                    "text": message
                }]
            }
        )
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)

while True:
    # 次の検知まで待機する
    time.sleep(CHECK_SECONDS)
    
    # 集計を行う時刻を経過したら実行する
    if target_time < datetime.now():
        # 集計の開始時刻
        start_time = target_time - timedelta(days=1)
        # 集計の終了時刻
        end_time = target_time
        # 開始時刻から終了時刻までの滞在時間を集計する
        seconds = get_stay_seconds(start_time, end_time)
        print('total:', str(seconds)+'s')
        # LINEに滞在時間を通知する
        send_message(seconds)
        # 集計時刻を1日後に更新する
        target_time = target_time + timedelta(days=1)
        
    # フレームの読み込み
    ret, frame = camera.read()
    if not ret:
        continue
    # フレームに写った人の判別
    frame, num_persons = detect(frame)
    # 検知した人数を記録する
    save(num_persons)
    # 人の位置を描画したフレームを画面に表示
    cv2.imshow("frame", frame)
    # 0キーが押されたら、繰り返しを終了する
    if cv2.waitKey(1)==0x30:
        break
    
camera.stop()
cv2.destroyAllWindows()
