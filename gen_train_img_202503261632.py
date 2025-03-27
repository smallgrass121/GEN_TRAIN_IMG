# === UI 擾動擴充工具（含標註複製）===
# 功能：
# - 將應用程式截圖進行各種擾動處理（11 種）以擴充 YOLO 訓練資料
# - 自動對應原始 .txt 標註（YOLO 格式），生成對應的 label
# - 結果存於 augmented_dataset/train/images 與 train/labels 中

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import shutil

# === 資料夾設定 ===
INPUT_FOLDER = r"C:\Users\waynelin\VSC\GEN_TRAIN_IMG\input_img"  # 原始圖片資料夾
LABEL_FOLDER = r"C:\Users\waynelin\VSC\GEN_TRAIN_IMG\input_label"  # 原始標註 .txt 位置
OUTPUT_FOLDER = r"C:\Users\waynelin\VSC\GEN_TRAIN_IMG\output_img"  # 擾動後圖片輸出位置
LABEL_OUT_FOLDER = r"C:\Users\waynelin\VSC\GEN_TRAIN_IMG\output_label"  # 擾動後標註輸出位置

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
Path(LABEL_OUT_FOLDER).mkdir(parents=True, exist_ok=True)

# === 各種擾動方法（包含參數與建議值域）===

def adjust_brightness(image, factor=1.2):
    """
    亮度調整
    參數：factor（float）>1.0 增加亮度，<1.0 降低亮度；建議值域 0.5 ~ 2.0
    """
    return ImageEnhance.Brightness(image).enhance(factor)

def adjust_contrast(image, factor=1.5):
    """
    對比度調整
    參數：factor（float）>1.0 增強對比，<1.0 降低對比；建議值域 0.5 ~ 2.0
    """
    return ImageEnhance.Contrast(image).enhance(factor)

def apply_blur(image, radius=2):
    """
    高斯模糊處理
    參數：radius（int or float）模糊半徑；建議值域 1 ~ 5
    """
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def add_noise(image, std=25):
    """
    加入隨機高斯雜訊
    參數：std（int）雜訊標準差，值越高雜訊越明顯；建議值域 10 ~ 50
    """
    arr = np.array(image)
    noise = np.random.normal(0, std, arr.shape).astype(np.uint8)
    noisy_img = cv2.add(arr, noise)
    return Image.fromarray(noisy_img)

def add_occlusion(image, area_ratio=0.05, color=(200, 200, 200)):
    """
    加入遮擋矩形塊（面積比）
    參數：
    - area_ratio（float）：遮擋區塊佔整體畫面面積的比例，建議值域 0.01 ~ 0.2
    - color（tuple）：遮擋顏色 (R,G,B)
    """
    arr = np.array(image)
    h, w = arr.shape[:2]
    total_area = h * w
    target_area = int(total_area * area_ratio)

    # 嘗試維持接近長寬比 2:1 的遮擋框
    box_w = int((target_area * 2) ** 0.5)
    box_h = int(box_w / 2)

    # 隨機選擇落點（確保遮擋框不超出邊界）
    x = np.random.randint(0, max(1, w - box_w))
    y = np.random.randint(0, max(1, h - box_h))

    cv2.rectangle(arr, (x, y), (x + box_w, y + box_h), color, -1)
    return Image.fromarray(arr)

def barrel_distortion(image, distortion=(0.1, 0.1)):
    """
    廣角鏡頭變形模擬
    參數：distortion（tuple）失真係數 (k1, k2)；建議值域 0.0001 ~ 0.001
    變形基準點改為隨機位置以模擬不同拍攝角度
    """
    arr = np.array(image)
    h, w = arr.shape[:2]
    cx = np.random.randint(int(w * 0.3), int(w * 0.7))
    cy = np.random.randint(int(h * 0.3), int(h * 0.7))
    K = np.array([[w, 0, cx], [0, h, cy], [0, 0, 1]])
    D = np.array([distortion[0], distortion[1], 0, 0])
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
    distorted = cv2.remap(arr, map1, map2, interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(distorted)

def perspective_warp(image, offset=30, direction=5):
    """
    視角仿射扭曲 (Perspective Warp)
    參數:
      image:  PIL Image 或可轉為 NumPy 陣列的影像
      offset: 仿射扭曲強度 (10 ~ 50)
      direction: 九宮格方向 (1~9)，5 表示無扭曲
    """
    arr = np.array(image)
    h, w = arr.shape[:2]

    # 原始四角 (左上、右上、左下、右下)
    src = np.float32([[0,0],[w,0],[0,h],[w,h]])
    dx = dy = offset

    # 順序均維持 [左上, 右上, 左下, 右下]
    dst_map = {
        1: np.float32([[dx, dy],   [w,    0],    [0,    h],    [w,    h]]),
        2: np.float32([[dx, dy],   [w-dx, dy],   [0,    h],    [w,    h]]),
        3: np.float32([[0,  0],    [w-dx, dy],   [0,    h],    [w,    h]]),
        4: np.float32([[dx, dy],   [w,    0],    [dx, h-dy],   [w,    h]]),
        5: src,
        6: np.float32([[0,  0],    [w-dx, dy],   [0,    h],    [w-dx, h-dy]]),
        7: np.float32([[0,  0],    [w,    0],    [dx, h-dy],   [w,    h]]),
        8: np.float32([[0,  0],    [w,    0],    [dx, h-dy],   [w-dx, h-dy]]),
        9: np.float32([[0,  0],    [w,    0],    [0,    h],    [w-dx, h-dy]])
    }

    matrix = cv2.getPerspectiveTransform(src, dst_map.get(direction, src))
    warped = cv2.warpPerspective(arr, matrix, (w, h))
    return Image.fromarray(warped)



def resolution_scale(image, scale=0.6):
    """
    解析度縮放模擬低畫質
    參數：scale（float）縮放比例；值 < 1 表示先縮小後再放大，建議值域 0.4 ~ 0.9
    """
    w, h = image.size
    img_small = image.resize((int(w * scale), int(h * scale)), resample=Image.BILINEAR)
    return img_small.resize((w, h), resample=Image.BILINEAR)

def hue_shift(image, shift_val=25):
    """
    色相偏移（HSV 色調調整）
    參數：shift_val（int）色相變化量；建議值域 -45 ~ 45
    """
    img_hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    img_hsv[..., 0] = (img_hsv[..., 0] + shift_val) % 180
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(img_rgb)

def white_balance_shift(image, r_shift=1.05, g_shift=0.95, b_shift=0.9):
    """
    白平衡偏移（RGB 通道比例調整）
    參數：r_shift, g_shift, b_shift（float）各通道的增益係數；建議值域 0.8 ~ 1.2
    """
    arr = np.array(image).astype(np.float32)
    arr[..., 0] *= r_shift
    arr[..., 1] *= g_shift
    arr[..., 2] *= b_shift
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def replace_background(image, threshold=30):
    """
    自動偵測背景色並隨機替換
    參數：threshold（int）像素與背景色差異的容許範圍；建議值域 10 ~ 60
    背景色由邊緣 8 方位 + 靠邊隨機 7 點，執行 9 次後取眾數最多的顏色
    """
    arr = np.array(image)
    h, w = arr.shape[:2]

    def sample_bg_color():
        edge_points = [
            arr[0, 0], arr[0, w // 2], arr[0, -1],
            arr[h // 2, 0], arr[h // 2, -1],
            arr[-1, 0], arr[-1, w // 2], arr[-1, -1]
        ]
        margin_x = max(1, int(w * 0.01))
        margin_y = max(1, int(h * 0.01))
        random_points = [
            arr[np.random.randint(0, margin_y), np.random.randint(0, w)],
            arr[np.random.randint(h - margin_y, h), np.random.randint(0, w)],
            arr[np.random.randint(0, h), np.random.randint(0, margin_x)],
            arr[np.random.randint(0, h), np.random.randint(w - margin_x, w)],
            arr[np.random.randint(0, margin_y), np.random.randint(0, margin_x)],
            arr[np.random.randint(0, margin_y), np.random.randint(w - margin_x, w)],
            arr[np.random.randint(h - margin_y, h), np.random.randint(w - margin_x, w)]
        ]
        all_samples = np.array(edge_points + random_points)
        return tuple(np.round(np.mean(all_samples, axis=0)).astype(np.uint8))

    # 執行 9 次採樣，統計眾數
    from collections import Counter
    results = [sample_bg_color() for _ in range(9)]
    bg_color = Counter(results).most_common(1)[0][0]

    diff = np.abs(arr - bg_color)
    mask = (diff.sum(axis=2) < threshold).astype(np.uint8)

    new_color = np.random.randint(0, 256, size=3)
    arr[mask == 1] = new_color
    return Image.fromarray(arr)

def shrink_and_embed(image, min_ratio=0.15):
    """
    將原圖縮小後嵌入到隨機尺寸背景中
    - min_ratio：縮小後圖片的面積比例需大於原圖總面積的指定比例，建議值域 0.1 ~ 0.3
    """
    orig_w, orig_h = image.size
    orig_area = orig_w * orig_h

    # 隨機選擇背景長寬比
    aspect_ratios = [(4,3), (16,9), (3,2), (1,1), (9,16)]
    ar = aspect_ratios[np.random.randint(0, len(aspect_ratios))]
    bg_w = max(orig_w, orig_h) * 2
    bg_h = int(bg_w * ar[1] / ar[0])

    # 隨機縮放比例，使縮圖面積大於 min_ratio * 原圖面積
    while True:
        scale = np.random.uniform(0.3, 0.7)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        if new_w * new_h >= orig_area * min_ratio:
            break

    # 產生背景並將原圖貼上去
    bg = Image.new('RGB', (bg_w, bg_h), tuple(np.random.randint(0, 255, 3)))
    resized = image.resize((new_w, new_h), resample=Image.BILINEAR)
    paste_x = np.random.randint(0, bg_w - new_w)
    paste_y = np.random.randint(0, bg_h - new_h)
    bg.paste(resized, (paste_x, paste_y))

    return bg

def random_rotate(image, degree=15):
    """
    隨機旋轉圖片
    參數：degree（int）最大旋轉角度（雙向 ±degree 範圍）；建議值域 5 ~ 30
    """
    angle = np.random.uniform(-degree, degree)
    return image.rotate(angle, resample=Image.BILINEAR, expand=True)

def overlay_ghost(image, alpha=0.4, offset=(10, 10)):
    """
    疊加半透明的圖片或拖影
    參數：
    - alpha（float）：透明度，0 為全透明，1 為不透明；建議值域 0.2 ~ 0.6
    - offset（tuple）：拖影的偏移像素 (x, y)
    """
    ghost = image.copy().convert("RGBA")
    ghost_arr = np.array(ghost)
    ghost_arr[..., 3] = int(alpha * 255)
    ghost = Image.fromarray(ghost_arr)

    base = image.convert("RGBA")
    w, h = base.size
    base.paste(ghost, offset, mask=ghost)
    return base.convert("RGB")

# === 建立擾動版本 ===
def generate_versions(image_path):
    base_name = Path(image_path).stem
    img = Image.open(image_path).convert('RGB')

    # 各種擾動樣態與強度設計（低、中、高），使用統一前綴編號 + 程度後綴數值
    versions = {
        # 亮度
        f"{base_name}_01_bright_11.jpg": adjust_brightness(img, factor=1.2),
        f"{base_name}_01_bright_14.jpg": adjust_brightness(img, factor=1.6),
        f"{base_name}_01_bright_18.jpg": adjust_brightness(img, factor=1.9),

        # 暗度
        f"{base_name}_04_dark_09.jpg": adjust_brightness(img, factor=0.9),
        f"{base_name}_04_dark_06.jpg": adjust_brightness(img, factor=0.65),
        f"{base_name}_04_dark_03.jpg": adjust_brightness(img, factor=0.4),

        # 對比
        f"{base_name}_07_contrast_12.jpg": adjust_contrast(img, factor=1.4),
        f"{base_name}_07_contrast_15.jpg": adjust_contrast(img, factor=1.75),
        f"{base_name}_07_contrast_18.jpg": adjust_contrast(img, factor=2.0),

        # 模糊
        f"{base_name}_10_blur_1.jpg": apply_blur(img, radius=1.6),
        f"{base_name}_10_blur_2.jpg": apply_blur(img, radius=2.4),
        f"{base_name}_10_blur_35.jpg": apply_blur(img, radius=3.5),

        # 雜訊
        f"{base_name}_13_noise_10.jpg": add_noise(img, std=5),
        f"{base_name}_13_noise_25.jpg": add_noise(img, std=25),
        f"{base_name}_13_noise_45.jpg": add_noise(img, std=50),

        # 遮擋（隨機顏色）
        f"{base_name}_16_occl_03.jpg": add_occlusion(img, area_ratio=0.08, color=tuple(int(v) for v in np.random.randint(0, 255, 3))),
        f"{base_name}_16_occl_08.jpg": add_occlusion(img, area_ratio=0.15, color=tuple(int(v) for v in np.random.randint(0, 255, 3))),
        f"{base_name}_16_occl_15.jpg": add_occlusion(img, area_ratio=0.2, color=tuple(int(v) for v in np.random.randint(0, 255, 3))),

        # 廣角
        f"{base_name}_19_barrel_01.jpg": barrel_distortion(img, distortion=(0.1, 0.1)),
        f"{base_name}_19_barrel_1.jpg": barrel_distortion(img, distortion=(0.3, 0.3)),
        f"{base_name}_19_barrel_8.jpg": barrel_distortion(img, distortion=(0.9, 0.9)),

        # 仿射視角（九方向）
        f"{base_name}_22_warp_dir_1.jpg": perspective_warp(img, offset=30, direction=1),
        f"{base_name}_22_warp_dir_2.jpg": perspective_warp(img, offset=30, direction=2),
        f"{base_name}_22_warp_dir_3.jpg": perspective_warp(img, offset=30, direction=3),
        f"{base_name}_22_warp_dir_4.jpg": perspective_warp(img, offset=30, direction=4),
        f"{base_name}_22_warp_dir_5.jpg": perspective_warp(img, offset=0, direction=5),  # 無變形
        f"{base_name}_22_warp_dir_6.jpg": perspective_warp(img, offset=30, direction=6),
        f"{base_name}_22_warp_dir_7.jpg": perspective_warp(img, offset=30, direction=7),
        f"{base_name}_22_warp_dir_8.jpg": perspective_warp(img, offset=30, direction=8),
        f"{base_name}_22_warp_dir_9.jpg": perspective_warp(img, offset=30, direction=9),

        # 解析度縮放
        f"{base_name}_25_scale_90.jpg": resolution_scale(img, scale=0.9),
        f"{base_name}_25_scale_60.jpg": resolution_scale(img, scale=0.6),
        f"{base_name}_25_scale_40.jpg": resolution_scale(img, scale=0.2),

        # 色相偏移
        f"{base_name}_28_hue_10.jpg": hue_shift(img, shift_val=10),
        f"{base_name}_28_hue_25.jpg": hue_shift(img, shift_val=25),
        f"{base_name}_28_hue_45.jpg": hue_shift(img, shift_val=45),

        # 白平衡
        f"{base_name}_31_wb_102_098_098.jpg": white_balance_shift(img, r_shift=1.02, g_shift=0.98, b_shift=0.95),
        f"{base_name}_31_wb_105_095_090.jpg": white_balance_shift(img, r_shift=1.05, g_shift=0.95, b_shift=0.85),
        f"{base_name}_31_wb_110_085_080.jpg": white_balance_shift(img, r_shift=1.1, g_shift=0.85, b_shift=0.7),

        # 背景替換（隨機顏色）
        f"{base_name}_34_bgreplace_30.jpg": replace_background(img, threshold=30),
        f"{base_name}_34_bgreplace_40.jpg": replace_background(img, threshold=40),
        f"{base_name}_34_bgreplace_50.jpg": replace_background(img, threshold=50),
        f"{base_name}_34_bgreplace_60.jpg": replace_background(img, threshold=60),
        f"{base_name}_34_bgreplace_80.jpg": replace_background(img, threshold=80),

        # 原圖縮小嵌入尺寸背景
        f"{base_name}_37_shrink_1.jpg": shrink_and_embed(img, min_ratio=0.15),
        f"{base_name}_37_shrink_2.jpg": shrink_and_embed(img, min_ratio=0.18),
        f"{base_name}_37_shrink_3.jpg": shrink_and_embed(img, min_ratio=0.2),

        # 旋轉擾動
        f"{base_name}_40_rotate_5.jpg": random_rotate(img, degree=60),
        f"{base_name}_40_rotate_10.jpg": random_rotate(img, degree=120),
        f"{base_name}_40_rotate_15.jpg": random_rotate(img, degree=195),
        f"{base_name}_40_rotate_20.jpg": random_rotate(img, degree=280),
        f"{base_name}_40_rotate_30.jpg": random_rotate(img, degree=340),

        # 拖影重疊
        f"{base_name}_43_ghost_10.jpg": overlay_ghost(img, alpha=0.3, offset=(10, 10)),
        f"{base_name}_43_ghost_20.jpg": overlay_ghost(img, alpha=0.4, offset=(20, 20)),
        f"{base_name}_43_ghost_30.jpg": overlay_ghost(img, alpha=0.5, offset=(-30, -15)),
        f"{base_name}_43_ghost_40.jpg": overlay_ghost(img, alpha=0.6, offset=(25, -30)),
        f"{base_name}_43_ghost_50.jpg": overlay_ghost(img, alpha=0.5, offset=(-40, 40)),
    }

    for filename, new_img in versions.items():
        new_img.save(os.path.join(OUTPUT_FOLDER, filename))
        label_name = base_name + '.txt'
        if os.path.exists(os.path.join(LABEL_FOLDER, label_name)):
            shutil.copy(
                os.path.join(LABEL_FOLDER, label_name),
                os.path.join(LABEL_OUT_FOLDER, Path(filename).with_suffix('.txt').name)
            )

# === 主程序：處理資料夾所有圖片 ===
def process_folder(folder):
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.png')):
            generate_versions(os.path.join(folder, file))

# 執行
process_folder(INPUT_FOLDER)
print("✅ 擾動圖片與標註完成，已儲存至 augmented_dataset/train")
