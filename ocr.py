import easyocr
import cv2
import numpy as np

# 初始化 OCR 閱讀器
reader = easyocr.Reader(['en'])

# 讀取驗證碼圖片
captcha_image = cv2.imread('captcha.png')

# 將圖片轉換為灰度圖像
gray = cv2.cvtColor(captcha_image, cv2.COLOR_BGR2GRAY)

# 應用中值模糊去噪
blurred = cv2.medianBlur(gray, 3)

# 應用自適應閾值處理
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# 應用形態學操作去噪
kernel = np.ones((2, 2), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 進一步增強對比度和亮度
alpha = 2.0  # 對比度控制 (1.0-3.0)
beta = 50    # 亮度控制 (0-100)
enhanced = cv2.convertScaleAbs(morph, alpha=alpha, beta=beta)

# 放大圖像
large_image = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

# 保存處理後的圖片以供檢查
cv2.imwrite('processed_captcha.png', large_image)

# 使用 EasyOCR 讀取驗證碼
result = reader.readtext(large_image)

# 提取識別出的文字
captcha_text = ''.join([res[1] for res in result])

# 打印識別結果
print(f"識別的驗證碼是：{captcha_text}")
