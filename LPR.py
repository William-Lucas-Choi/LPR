import cv2
import pytesseract
import numpy as np
import os

# 画像ファイルがあるディレクトリパス
image_directory = './img/'

# 結果を保存するディレクトリパス
output_directory = './detected/'

# 出力ディレクトリが存在しない場合は作成
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# サポートされている画像ファイル拡張子リスト
supported_extensions = ('.jpeg', '.jpg', '.png')

# ナンバープレート認識のための関数
def recognize_plate(image_path, output_path):
    # 画像の読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"画像を読み込めません: {image_path}")
        return None

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 画像を2次元配列に変換
    pixel_values = img_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # K-meansクラスタリングの適用
    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # クラスタ中心値を整数型に変換
    centers = np.uint8(centers)

    # 各ピクセルにクラスタ中心値を割り当て
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img_rgb.shape)

    # 最も大きな領域（ナンバープレートと推定される領域）を選択
    gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
    _, binary_segmented = cv2.threshold(gray_segmented, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 輪郭を探す
    contours, _ = cv2.findContours(binary_segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        if len(approx) >= 4:
            plate = approx
            break

    if plate is None:
        print("ナンバープレートを見つけることができません。")
        return None

    # ナンバープレート領域に緑色の線で四角形を描画
    cv2.drawContours(image, [plate], -1, (0, 255, 0), 3)
    
    # ナンバープレート領域を切り取り
    x, y, w, h = cv2.boundingRect(plate)
    cropped = gray_segmented[y:y + h, x:x + w]

    # ナンバープレート画像の前処理
    _, binary = cv2.threshold(cropped, 150, 255, cv2.THRESH_BINARY)
    resized = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    
    # 上段と下段を分割
    upper_half = resized[0:int(resized.shape[0]/2), :]
    lower_half = resized[int(resized.shape[0]*3/10):, :]

    # 上部からボルトとボルトの間の領域を抽出
    uh, uw = upper_half.shape
    center_upper_half = upper_half[:, int(uw*0.2):int(uw*0.8)]  # 左右20％ずつカット
    
    # 下段左右分割
    h, w = lower_half.shape
    left_part = lower_half[:int(h*0.9), int(w*0.05):int(w*0.2)]
    right_part = lower_half[:, int(w*0.2):]

    # Tesseract OCRでテキスト認識
    config = '--psm 7 --oem 3'
    upper_text = pytesseract.image_to_string(center_upper_half, config=config, lang='jpn').strip()
    left_text = pytesseract.image_to_string(left_part, config=config, lang='jpn').strip()
    right_text = pytesseract.image_to_string(right_part, config=config, lang='jpn').strip()

    # ナンバープレート領域が含まれた画像を保存
    output_image_path = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)

    # 認識結果をテキストファイルに保存（拡張子なし）
    output_text_path = os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0])
    with open(output_text_path, 'w', encoding='utf-8') as f:
        f.write(f'{upper_text} {left_text} {right_text}')

    return f'{upper_text} {left_text} {right_text}'

# 指定されたディレクトリのすべての画像ファイルに対してナンバープレート認識を実行
for filename in os.listdir(image_directory):
    if filename.lower().endswith(supported_extensions):
        image_path = os.path.join(image_directory, filename)
        result = recognize_plate(image_path, output_directory)
        print(f'File: {filename} Result: {result}')
