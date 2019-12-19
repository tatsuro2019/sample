#coding:utf-8

import numpy as np
import cv2
import random


class mouseParam:
    def __init__(self, input_img_name):
        # マウス入力用のパラメータ
        self.mouseEvent = {"x": None, "y": None, "event": None, "flags": None}
        # マウス入力の設定
        cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)

    # コールバック関数
    def __CallBackFunc(self, eventType, x, y, flags, userdata):
        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType
        self.mouseEvent["flags"] = flags

        # マウス入力用のパラメータを返すための関数

    def getData(self):
        return self.mouseEvent

    # マウスイベントを返す関数
    def getEvent(self):
        return self.mouseEvent["event"]

        # マウスフラグを返す関数

    def getFlags(self):
        return self.mouseEvent["flags"]

        # xの座標を返す関数

    def getX(self):
        return self.mouseEvent["x"]

        # yの座標を返す関数

    def getY(self):
        return self.mouseEvent["y"]

        # xとyの座標を返す関数

    def getPos(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])


if __name__ == "__main__":
    # 入力画像
    read = cv2.imread("input.bmp")

    # 表示するWindow名
    window_name = "input window"

    # 画像の表示
    cv2.imshow(window_name, read)

    # コールバックの設定
    mouseData = mouseParam(window_name)

    while 1:
        cv2.waitKey(20)
        # 左クリックがあったら表示
        if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN:
            print(mouseData.getPos())
            file = open('point_x.txt', 'w')
            file.write(str(mouseData.getX()))
            file.close()
            file = open('point_y.txt', 'w')
            file.write(str(mouseData.getY()))
            file.close()
        # Mクリックがあったら終了
        elif mouseData.getEvent() == cv2.EVENT_MBUTTONDOWN:
            break

    cv2.destroyAllWindows()

# 画像の読み込み
test = cv2.imread("input.bmp", cv2.IMREAD_COLOR)#BGRなので気をつける
gray_test = cv2.imread("input.bmp",cv2.IMREAD_GRAYSCALE)
width = test.shape[0]
height = test.shape[1]
frag = np.zeros(gray_test.shape)#領域分割フラグ

# 画像の書き出し
cv2.imwrite('test.bmp', test)
cv2.imwrite('gray_test.bmp',gray_test)

# 初期地点
file_data = open('point_x.txt', 'r')
lines_x = file_data.readline()
file_data.close()
cv_x = int(lines_x)

file_data = open('point_y.txt', 'r')
lines_y = file_data.readline()
file_data.close()
cv_y = int(lines_y)



color = 255
i = 0

stack = [cv_x,cv_y]
while len(stack) != 0:
    #xyが逆　例:(27,26)→(y,x)
    pyy = stack.pop()
    pxx = stack.pop()
    if (gray_test[pyy][pxx] == 255):
        frag[pyy][pxx] = 255
        gray_test[pyy][pxx] = 0
        if ((pyy+1 < height) & (gray_test[pyy+1][pxx] == color)):
            stack.append(pxx)
            stack.append(pyy+1)
        if ((pxx+1 < width) & (gray_test[pyy][pxx+1] == color)):
            stack.append(pxx+1)
            stack.append(pyy)
        if (pyy-1 >= 0) & (gray_test[pyy-1][pxx] == color):
            stack.append(pxx)
            stack.append(pyy-1)
        if (pxx-1 >= 0) & (gray_test[pyy][pxx-1] == color):
            stack.append(pxx-1)
            stack.append(pyy)
        i += 1
print(i)
print("Finished")
cv2.imwrite('result.jpg', frag)

# カラー画像への処理
if __name__ == "__main__":
    # 入力画像
    read = cv2.imread("input_c.bmp")

    # 表示するWindow名
    window_name = "input_c window"

    # 画像の表示
    cv2.imshow(window_name, read)

    # コールバックの設定
    mouseData = mouseParam(window_name)

    print("色を二色選んでください\n色A:左クリック\n色B:右クリック\n終了:ホイール押し込み")
    magnification = 1
    while 1:
        cv2.waitKey(20)
        # 左クリックで座標表示＆色取得
        if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN:
            if magnification == 1:
                print(mouseData.getPos())
                file = open('color_a_x.txt', 'w')
                file.write(str(mouseData.getX()))
                file.close()
                file = open('color_a_y.txt', 'w')
                file.write(str(mouseData.getY()))
                file.close()
            else:
                print("(", mouseData.getX()//magnification, ",", mouseData.getY()//magnification, ")")
                file = open('color_a_x.txt', 'w')
                file.write(str(mouseData.getX()//magnification))
                file.close()
                file = open('color_a_y.txt', 'w')
                file.write(str(mouseData.getY()//magnification))
                file.close()
        # 右クリックで座標表示＆色取得
        if mouseData.getEvent() == cv2.EVENT_RBUTTONDOWN:
            if magnification == 1:
                print(mouseData.getPos())
                file = open('color_b_x.txt', 'w')
                file.write(str(mouseData.getX()))
                file.close()
                file = open('color_b_y.txt', 'w')
                file.write(str(mouseData.getY()))
                file.close()
            else:
                print("(", mouseData.getX()//magnification, ",", mouseData.getY()//magnification, ")")
                file = open('color_b_x.txt', 'w')
                file.write(str(mouseData.getX()//magnification))
                file.close()
                file = open('color_b_y.txt', 'w')
                file.write(str(mouseData.getY()//magnification))
                file.close()
        # 20Fescキー長押しで画像の2倍
        if cv2.waitKey(20) & 0xFF == 27:
                zoomed_image = read.repeat(magnification*2, axis=0).repeat(magnification*2, axis=1)
                cv2.imshow(window_name, zoomed_image)
                magnification = magnification*2
                cv2.waitKey(80)
        # Mボタンクリックがあったら終了
        elif mouseData.getEvent() == cv2.EVENT_MBUTTONDOWN:
            break

    cv2.destroyAllWindows()

# 座標の書き出し
# 色A
file_data = open('color_a_x.txt', 'r')
lines_x = file_data.readline()
file_data.close()
ca_x = int(lines_x)

file_data = open('color_a_y.txt', 'r')
lines_y = file_data.readline()
file_data.close()
ca_y = int(lines_y)

# 色B
file_data = open('color_b_x.txt', 'r')
lines_x = file_data.readline()
file_data.close()
cb_x = int(lines_x)

file_data = open('color_b_y.txt', 'r')
lines_y = file_data.readline()
file_data.close()
cb_y = int(lines_y)

# 画像の読み込み
test_c = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)#BGRなので気をつける
gray_test2 = cv2.imread("input.bmp", cv2.IMREAD_GRAYSCALE)
frag2 = np.zeros(test_c.shape)#領域分割フラグ

# 画像の書き出し
cv2.imwrite('test_c.bmp', test_c)

stack = [cv_x, cv_y]
while len(stack) != 0:
    #xyが逆　例:(27,26)→(y,x)
    pyy = stack.pop()
    pxx = stack.pop()
    if (gray_test2[pyy][pxx] == 255):
        gray_test2[pyy][pxx] = 0
        if (set(test_c[pyy][pxx]) == set(test_c[ca_y][ca_x])):
            frag2[pyy][pxx] = [255, 0, 255]
            if ((pyy+1 < height) & (gray_test2[pyy+1][pxx] == color)):
                stack.append(pxx)
                stack.append(pyy+1)
            if ((pxx+1 < width) & (gray_test2[pyy][pxx+1] == color)):
                stack.append(pxx+1)
                stack.append(pyy)
            if (pyy-1 >= 0) & (gray_test2[pyy-1][pxx] == color):
                stack.append(pxx)
                stack.append(pyy-1)
            if (pxx-1 >= 0) & (gray_test2[pyy][pxx-1] == color):
                stack.append(pxx-1)
                stack.append(pyy)
        if (set(test_c[pyy][pxx]) == set(test_c[cb_y][cb_x])):
            frag2[pyy][pxx] = [255, 255, 0]
            if ((pyy+1 < height) & (gray_test2[pyy+1][pxx] == color)):
                stack.append(pxx)
                stack.append(pyy+1)
            if ((pxx+1 < width) & (gray_test2[pyy][pxx+1] == color)):
                stack.append(pxx+1)
                stack.append(pyy)
            if (pyy-1 >= 0) & (gray_test2[pyy-1][pxx] == color):
                stack.append(pxx)
                stack.append(pyy-1)
            if (pxx-1 >= 0) & (gray_test2[pyy][pxx-1] == color):
                stack.append(pxx-1)
                stack.append(pyy)

print("Finished")
cv2.imwrite('result2.png', frag2)

# 色領域A
# 画像の読み込み
color_1 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)#BGRなので気をつける
gray_test3 = cv2.imread("input.bmp", cv2.IMREAD_GRAYSCALE)
frag3 = np.zeros(color_1.shape)#領域分割フラグ

stack = [ca_x, ca_y]
while len(stack) != 0:
    #xyが逆　例:(27,26)→(y,x)
    pyy = stack.pop()
    pxx = stack.pop()
    if (gray_test3[pyy][pxx] == 255):
        gray_test3[pyy][pxx] = 0
        if (set(color_1[pyy][pxx]) == set(color_1[ca_y][ca_x])):
            frag3[pyy][pxx] = [255, 255, 255]
            if ((pyy+1 < height) & (gray_test3[pyy+1][pxx] == color)):
                stack.append(pxx)
                stack.append(pyy+1)
            if ((pxx+1 < width) & (gray_test3[pyy][pxx+1] == color)):
                stack.append(pxx+1)
                stack.append(pyy)
            if (pyy-1 >= 0) & (gray_test3[pyy-1][pxx] == color):
                stack.append(pxx)
                stack.append(pyy-1)
            if (pxx-1 >= 0) & (gray_test3[pyy][pxx-1] == color):
                stack.append(pxx-1)
                stack.append(pyy)

print("色A領域")
cv2.imwrite('color_1.png', frag3)

# 膨張
di_1 = cv2.imread('color_1.png', 0)
kernel = np.ones((3, 3), np.uint8)
cv2.imwrite('di_1.png', cv2.dilate(di_1, kernel, iterations = 1))

# 色A領域(元画像)
di_c1 = cv2.imread('color_1.png', 0)
# 膨張画像
di_a1 = cv2.imread('di_1.png', 0)
# 結果反映用画像
di_r1 = cv2.imread('di_1.png', 0)
for i in range(height):
    for j in range(width):
        if (di_c1[i][j] == di_a1[i][j]):
            di_r1[i][j] = 0

# 膨張画像di_1と色領域Aの差分
cv2.imwrite('gradient_a.png', di_r1)


# 色領域B
# 画像の読み込み
color_2 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)#BGRなので気をつける
gray_test4 = cv2.imread("input.bmp", cv2.IMREAD_GRAYSCALE)
frag4 = np.zeros(color_2.shape)#領域分割フラグ

stack = [cb_x, cb_y]
while len(stack) != 0:
    #xyが逆　例:(27,26)→(y,x)
    pyy = stack.pop()
    pxx = stack.pop()
    if (gray_test4[pyy][pxx] == 255):
        gray_test4[pyy][pxx] = 0
        if (set(color_2[pyy][pxx]) == set(color_1[cb_y][cb_x])):
            frag4[pyy][pxx] = [255, 255, 255]
            if ((pyy+1 < height) & (gray_test4[pyy+1][pxx] == color)):
                stack.append(pxx)
                stack.append(pyy+1)
            if ((pxx+1 < width) & (gray_test4[pyy][pxx+1] == color)):
                stack.append(pxx+1)
                stack.append(pyy)
            if (pyy-1 >= 0) & (gray_test4[pyy-1][pxx] == color):
                stack.append(pxx)
                stack.append(pyy-1)
            if (pxx-1 >= 0) & (gray_test4[pyy][pxx-1] == color):
                stack.append(pxx-1)
                stack.append(pyy)

print("色B領域")
cv2.imwrite('color_2.png', frag4)

# 膨張
di_2 = cv2.imread('color_2.png', 0)
cv2.imwrite('di_2.png', cv2.dilate(di_2, kernel, iterations = 1))

# 色B領域(元画像)
di_c2 = cv2.imread('color_2.png', 0)
# 膨張画像
di_a2 = cv2.imread('di_2.png', 0)
# 結果反映用画像
di_r2 = cv2.imread('di_2.png', 0)
for i in range(height):
    for j in range(width):
        if (di_c2[i][j] == di_a2[i][j]):
            di_r2[i][j] = 0

# 膨張画像di_2と色領域Bの差分
cv2.imwrite('gradient_b.png', di_r2)

# 膨張Aと色B領域の重なり部分の処理
# 膨張差分A
gr_a = cv2.imread('gradient_a.png', 0)
# 色B領域
cregion_b = cv2.imread('color_2.png', 0)
# 反映先画像
tile_a = np.zeros(color_2.shape)
for i in range(height):
    for j in range(width):
        if ((gr_a[i][j] == 255) & (cregion_b[i][j] == 255)):
            tile_a[i][j] = 255

# タイルパターン領域aの生成
cv2.imwrite('tile_a.png', tile_a)



# 膨張Bと色A領域の重なり部分の処理
# 膨張差分B
gr_b = cv2.imread('gradient_b.png', 0)
# 色B領域
cregion_a = cv2.imread('color_1.png', 0)
# 反映先画像
tile_b = np.zeros(color_2.shape)
for i in range(height):
    for j in range(width):
        if ((gr_b[i][j] == 255) & (cregion_a[i][j] == 255)):
            tile_b[i][j] = 255

# タイルパターン領域bの生成
cv2.imwrite('tile_b.png', tile_b)



# タイルパターン領域aとbの合成
# 膨張差分B
tile_a2 = cv2.imread('tile_a.png', 0)
# 色B領域
tile_b2 = cv2.imread('tile_b.png', 0)
# 反映先画像
tile_ab = np.zeros(color_2.shape)
for i in range(height):
    for j in range(width):
        if (tile_a2[i][j] != tile_b2[i][j]):
            tile_ab[i][j] = 255

# タイルパターン領域abの生成
cv2.imwrite('tile_ab.png', tile_ab)

# 真ん中領域の膨張
# 膨張
center_b = cv2.imread('tile_ab.png', 0)
cv2.imwrite('center_b.png', cv2.dilate(center_b, kernel, iterations = 1))
center_bb = cv2.imread('center_b.png', 0)
# 結果反映用画像
center_ll = cv2.imread('center_b.png', 0)
# 膨張画像との差分
for i in range(height):
    for j in range(width):
        if (center_bb[i][j] == center_b[i][j]):
            center_ll[i][j] = 0

cv2.imwrite('center_ll.png', center_ll)


# 左領域(A)(di_c1)との共通部分
# 色A領域(元画像) di_c1 = cv2.imread('color_1.png', 0)
# 結果反映用画像
tile_left = np.zeros(color_2.shape)
for i in range(height):
    for j in range(width):
        if ((center_ll[i][j] == 255) & (di_c1[i][j] == 255)):
            tile_left[i][j] = 255

# 膨張画像center_llと色領域A(di_c1=color_1.png)の重なり部分
cv2.imwrite('tile_left.png', tile_left)

# 右領域(B)(di_c2)との共通部分
# 色B領域(元画像) di_c2 = cv2.imread('color_2.png', 0)
# 結果反映用画像
tile_right = np.zeros(color_2.shape)
for i in range(height):
    for j in range(width):
        if ((center_ll[i][j] == 255) & (di_c2[i][j] == 255)):
            tile_right[i][j] = 255

# 膨張画像center_llと色領域B(di_c2=color_2.png)の重なり部分
cv2.imwrite('tile_right.png', tile_right)


# タイルパターン領域abとright,leftの合成
# 真ん中center_b = cv2.imread('tile_ab.png', 0)
# 右領域
tile_br = cv2.imread('tile_right.png', 0)
# 左領域
tile_bl = cv2.imread('tile_left.png', 0)
# 反映先画像
tile_rabl = np.zeros(color_2.shape)
for i in range(height):
    for j in range(width):
        if (tile_br[i][j] != center_b[i][j]):
            tile_rabl[i][j] = 255
        if (tile_bl[i][j] != center_b[i][j]):
            tile_rabl[i][j] = 255

# タイルパターン領域abの生成
cv2.imwrite('tile_rabl.png', tile_rabl)


# 局所探索領域の作成
# abとright,leftの合成画像
tile_rabl2 = cv2.imread('tile_rabl.png', 0)
# 縦方向走査
check_p = 0
count_s = 0
count_t = 0
for i in range(width):
    for j in range(height):
        if (tile_rabl2[j][i] == 255):
            if check_p == 0:
                left_x = i
                check_p = 1
                break
            else:
                right_x = i
                count_s += 1
                break
    if count_s != 0:
        count_t += 1
        if count_s != count_t:
            break

print(left_x, right_x)
# 横方向走査
check_p = 0
count_s = 0
count_t = 0
for i in range(height):
    for j in range(width):
        if (tile_rabl2[i][j] == 255):
            if check_p == 0:
                up_y = i
                check_p = 1
                break
            else:
                under_y = i
                count_s += 1
                break
    if count_s != 0:
        count_t += 1
        if count_s != count_t:
            break

print(up_y, under_y)

# 反映先画像
lsr = np.zeros(color_2.shape)
for i in range(up_y, under_y+1):
    for j in range(left_x, right_x+1):
        lsr[i][j] = 255

# タイルパターンを施す長方形画像の生成
cv2.imwrite('lsr.png', lsr)

# ここからタイルパターン設定

# 50%パターン
# 結果反映用画像lsr2
tile_50 = cv2.imread('lsr.png', 0)
if ((up_y + left_x) % 2) == 0:
    upper_left_c = 0
else:
    upper_left_c = 1
for i in range(up_y, under_y+1):
    for j in range(left_x, right_x+1):
        if upper_left_c == 0:
            if ((i + j) % 2) == 1:
                tile_50[i][j] = 0
        else:
            if ((i + j) % 2) == 0:
                tile_50[i][j] = 0

# 50%確認用
cv2.imwrite('tile_50.png', tile_50)

# 50%タイル実装
# 画像の読み込み　color_2 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)#BGRなので気をつける
# 真ん中(50%)のTRP center_b = cv2.imread('tile_ab.png', 0)
# 色A座標 ca_x = int(lines_x),ca_y = int(lines_y)
# 色B座標 cb_x = int(lines_x),cb_y = int(lines_y)
# 結果反映用画像finish_50
finish_50 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
for i in range(up_y, under_y+1):
    for j in range(left_x, right_x+1):
        if (tile_50[i][j] == 255) & (center_b[i][j] == 255):
            finish_50[i][j] = color_2[ca_y][ca_x]
        if (tile_50[i][j] == 0) & (center_b[i][j] == 255):
            finish_50[i][j] = color_2[cb_y][cb_x]

# 50%確認用
cv2.imwrite('finish_50.png', finish_50)


# 25%パターン
# 結果反映用画像tile_25
tile_25 = cv2.imread('lsr.png', 0)
Nol = 0
for i in range(up_y, under_y+1):
    if (Nol % 2) != 0:
        for j in range(left_x, right_x+1, 2):
                tile_25[i][j] = 0
    Nol += 1
# 25%確認用
cv2.imwrite('tile_25.png', tile_25)

# 50%タイル画像に25%タイル実装
# 画像の読み込み　color_2 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)#BGRなので気をつける
# 左領域(25%)tile_bl = cv2.imread('tile_left.png', 0)
# 色A座標 ca_x = int(lines_x),ca_y = int(lines_y)
# 色B座標 cb_x = int(lines_x),cb_y = int(lines_y)
for i in range(up_y, under_y+1):
    for j in range(left_x, right_x+1):
        if (tile_25[i][j] == 255) & (tile_bl[i][j] == 255):
            finish_50[i][j] = color_2[ca_y][ca_x]
        if (tile_25[i][j] == 0) & (tile_bl[i][j] == 255):
            finish_50[i][j] = color_2[cb_y][cb_x]

# 50%+25%適用確認用
cv2.imwrite('finish_50+25.png', finish_50)


# 75%パターン
# 結果反映用画像lsr2
tile_75 = cv2.imread('lsr.png', 0)
Nol =0
for i in range(up_y, under_y+1):
    if (Nol % 2) == 0:
        for j in range(left_x, right_x+1):
            if upper_left_c == 0:
                if ((i + j) % 2) == 1:
                    tile_75[i][j] = 0
            else:
                if ((i + j) % 2) == 0:
                    tile_75[i][j] = 0
    else:
        for j in range(left_x, right_x + 1):
            tile_75[i][j] = 0
    Nol += 1
# 75%確認用
cv2.imwrite('tile_75.png', tile_75)

# 50+25%タイル画像に75%タイル実装
# 画像の読み込み　color_2 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)#BGRなので気をつける
# 右領域(75%)tile_br = cv2.imread('tile_right.png', 0)
# 色A座標 ca_x = int(lines_x),ca_y = int(lines_y)
# 色B座標 cb_x = int(lines_x),cb_y = int(lines_y)
for i in range(up_y, under_y+1):
    for j in range(left_x, right_x+1):
        if (tile_75[i][j] == 255) & (tile_br[i][j] == 255):
            finish_50[i][j] = color_2[ca_y][ca_x]
        if (tile_75[i][j] == 0) & (tile_br[i][j] == 255):
            finish_50[i][j] = color_2[cb_y][cb_x]

# 50%+25%+75%適用確認用
cv2.imwrite('finish_50+25+75.png', finish_50)

# 12.5%パターン
# 結果反映用画像tile_12
tile_12 = cv2.imread('lsr.png', 0)
Nol = 0
check_tile_12 = 0
for i in range(up_y, under_y+1):
    if (Nol % 2) != 0:
        if (check_tile_12 % 2) == 0:
            for j in range(left_x, right_x+1, 4):
                tile_12[i][j] = 0
        else:
            for j in range(left_x + 2, right_x + 1, 4):
                tile_12[i][j] = 0
        check_tile_12 += 1
    Nol += 1
# 12.5%確認用
cv2.imwrite('tile_12.5.png', tile_12)

# 37.5%パターン
# 結果反映用画像tile_37
tile_37 = cv2.imread('lsr.png', 0)
Nol = 0
for i in range(up_y, under_y+1):
    if Nol != 0:
        if (Nol % 2) == 0:
            if (Nol % 4) == 0:
                for j in range(left_x + 3, right_x + 1, 4):
                    tile_37[i][j] = 0
            else:
                for j in range(left_x + 1, right_x + 1, 4):
                    tile_37[i][j] = 0
        else:
            if upper_left_c == 0:
                for j in range(left_x, right_x + 1):
                    if ((i + j) % 2) == 1:
                        tile_37[i][j] = 0
            else:
                for j in range(left_x, right_x + 1):
                    if ((i + j) % 2) == 0:
                        tile_37[i][j] = 0
    Nol += 1
# 37.5%確認用
cv2.imwrite('tile_37.5.png', tile_37)

# 62.5%パターン
# 結果反映用画像tile_62
tile_62 = cv2.imread('lsr.png', 0)
Nol = 0
check_tile_62 = 1
for i in range(up_y, under_y+1):
    if (Nol % 2) != 0:
        if (Nol % 4) == 3:
            check_tile_62 = 3
            for j in range(left_x, right_x + 1):
                if (check_tile_62 % 4) != 0:
                    tile_62[i][j] = 0
                    check_tile_62 += 1
                else:
                    check_tile_62 += 1
            check_tile_62 = 1
        else:
            for j in range(left_x, right_x + 1):
                if (check_tile_62 % 4) != 0:
                    tile_62[i][j] = 0
                    check_tile_62 += 1
                else:
                    check_tile_62 += 1
            check_tile_62 = 1
    else:
        if upper_left_c == 0:
            for j in range(left_x, right_x + 1):
                if ((i + j) % 2) == 1:
                    tile_62[i][j] = 0
        else:
            for j in range(left_x, right_x + 1):
                if ((i + j) % 2) == 0:
                    tile_62[i][j] = 0
    Nol += 1
# 62.5%確認用
cv2.imwrite('tile_62.5.png', tile_62)

# 87.5%パターン
# 結果反映用画像tile_87
tile_87 = cv2.imread('lsr.png', 0)
Nol = 0
check_tile_87 = 1
for i in range(up_y, under_y+1):
    if (Nol % 2) != 0:
        if (Nol % 4) == 3:
            for j in range(left_x, right_x + 1):
                if (check_tile_87 % 4) != 0:
                    tile_87[i][j] = 0
                    check_tile_87 += 1
                else:
                    check_tile_87 += 1
            check_tile_87 = 1
        else:
            check_tile_87 = 3
            for j in range(left_x, right_x + 1):
                if (check_tile_87 % 4) != 0:
                    tile_87[i][j] = 0
                    check_tile_87 += 1
                else:
                    check_tile_87 += 1
            check_tile_87 = 1
    else:
        for j in range(left_x, right_x + 1):
            tile_87[i][j] = 0
    Nol += 1
# 87.5%確認用
cv2.imwrite('tile_87.5.png', tile_87)

# 100%パターン
# 結果反映用画像tile_100
tile_100 = cv2.imread('lsr.png', 0)
for i in range(up_y, under_y+1):
    for j in range(left_x, right_x + 1):
        tile_100[i][j] = 0
# 100%確認用
cv2.imwrite('tile_100.png', tile_100)

# 0%パターン
# 結果反映用画像tile_0
tile_0 = cv2.imread('lsr.png', 0)
# 0%確認用
cv2.imwrite('tile_0.png', tile_0)

# =========================ここまでタイル================================

# R, G, Bの値を取得して0～1の範囲内にする
[blue, green, red] = color_2[ca_y][ca_x]/255.0
# R, G, Bの値から最大値を計算
mx_v1 = max(red, green, blue)

# 同文b
[blue, green, red] = color_2[cb_y][cb_x]/255.0
mx_v2 = max(red, green, blue)

if mx_v2 > mx_v1:
    region_weight_a = 2
    region_weight_b = 0
    region_weight_ab = 1
else:
    region_weight_a = 0
    region_weight_b = 2
    region_weight_ab = 1

# 領域の重み確認用
print("領域a,b,abの重み")
print(region_weight_a, region_weight_b, region_weight_ab)

# 領域反映ファンクション
def tile_set_ab(x, y, z):
    for i in range(up_y, under_y + 1):
        for j in range(left_x, right_x + 1):
            if (x[i][j] == 255) & (y[i][j] == 255):
                z[i][j] = color_2[ca_y][ca_x]
            if (x[i][j] == 0) & (y[i][j] == 255):
                z[i][j] = color_2[cb_y][cb_x]

# 作業領域の選定
def tile_set_check_lr(x, y, z):
    if y == 100:
        tile_set_ab(x, tile_br, z)
    elif y == 0:
        tile_set_ab(x, tile_bl, z)
    else:
        tile_set_ab(x, center_b, z)

# 使用タイルの選別
def tile_set_check_no(x, y, z):
    if x == 1:
        tile_set_check_lr(tile_12, y, z)
    elif x == 2:
        tile_set_check_lr(tile_25, y, z)
    elif x == 3:
        tile_set_check_lr(tile_37, y, z)
    elif x == 4:
        tile_set_check_lr(tile_50, y, z)
    elif x == 5:
        tile_set_check_lr(tile_62, y, z)
    elif x == 6:
        tile_set_check_lr(tile_75, y, z)
    elif x == 7:
        tile_set_check_lr(tile_87, y, z)
    elif x == 8:
        tile_set_check_lr(tile_100, y, z)
    else:
        tile_set_check_lr(tile_0, y, z)

# 領域の重み判別
def tile_relay(x, y, z, op):
    if region_weight_a == 0:
        tile_set_check_no(x, 0, op)
        tile_set_check_no(y, 100, op)
        tile_set_check_no(z, 50, op)
    else:
        tile_set_check_no(x, 100, op)
        tile_set_check_no(y, 0, op)
        tile_set_check_no(z, 50, op)

# 乱数の生成
def rand_set(x):
    # 0~99の整数を1個作成
    check_ctile_no = random.randint(0, 99)
    if check_ctile_no <= 4:
        ctile_no = 1
    elif (check_ctile_no >= 5) & (check_ctile_no <= 14):
        ctile_no = 2
    elif (check_ctile_no >= 15) & (check_ctile_no <= 34):
        ctile_no = 3
    elif (check_ctile_no >= 35) & (check_ctile_no <= 64):
        ctile_no = 4
    elif (check_ctile_no >= 65) & (check_ctile_no <= 84):
        ctile_no = 5
    elif (check_ctile_no >= 85) & (check_ctile_no <= 94):
        ctile_no = 6
    else:
        ctile_no = 7

    if region_weight_a == 0:
        ltile_no = random.randint(0, ctile_no - 1)
        rtile_no = random.randint(ctile_no + 1, 8)
        tile_relay(ltile_no, rtile_no, ctile_no, x)
    else:
        rtile_no = random.randint(0, ctile_no - 1)
        ltile_no = random.randint(ctile_no + 1, 8)
        tile_relay(ltile_no, rtile_no, ctile_no, x)

    print("各タイルNo:", ltile_no, rtile_no, ctile_no)
    return ltile_no, rtile_no, ctile_no

# 初期解の生成及び引数の格納
finish_1 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
lt_no, rt_no, ct_no = rand_set(finish_1)
tile_resource = [[lt_no, rt_no, ct_no]]
cv2.imwrite('finish_1.png', finish_1)

finish_2 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
lt_no2, rt_no2, ct_no2 = rand_set(finish_2)
tile_resource.append([lt_no2, rt_no2, ct_no2])
cv2.imwrite('finish_2.png', finish_2)

finish_3 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
lt_no3, rt_no3, ct_no3 = rand_set(finish_3)
tile_resource.append([lt_no3, rt_no3, ct_no3])
cv2.imwrite('finish_3.png', finish_3)

finish_4 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
lt_no4, rt_no4, ct_no4 = rand_set(finish_4)
tile_resource.append([lt_no4, rt_no4, ct_no4])
cv2.imwrite('finish_4.png', finish_4)

finish_5 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
lt_no5, rt_no5, ct_no5 = rand_set(finish_5)
tile_resource.append([lt_no5, rt_no5, ct_no5])
cv2.imwrite('finish_5.png', finish_5)

finish_6 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
lt_no6, rt_no6, ct_no6 = rand_set(finish_6)
tile_resource.append([lt_no6, rt_no6, ct_no6])
cv2.imwrite('finish_6.png', finish_6)

finish_7 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
lt_no7, rt_no7, ct_no7 = rand_set(finish_7)
tile_resource.append([lt_no7, rt_no7, ct_no7])
cv2.imwrite('finish_7.png', finish_7)

finish_8 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
lt_no8, rt_no8, ct_no8 = rand_set(finish_8)
tile_resource.append([lt_no8, rt_no8, ct_no8])
cv2.imwrite('finish_8.png', finish_8)

# print(tile_resource[1][1])

# ===================================初期解の表示および選択===================================
print("いずれかの画像をアクティブにしescキーを押すと次に進みます")
while 1:
    # 入力画像
    suggestion_0_1 = cv2.imread("finish_1.png")
    # 表示するWindow名
    window_name_0_1 = "suggestion_0_1"
    # 画像の表示
    cv2.imshow(window_name_0_1, suggestion_0_1)
    # Window位置の変更　第1引数：Windowの名前　第2引数：x 第3引数：y
    cv2.moveWindow('suggestion_0_1', 100, 200)

    suggestion_0_2 = cv2.imread("finish_2.png")
    window_name_0_2 = "suggestion_0_2"
    cv2.imshow(window_name_0_2, suggestion_0_2)
    cv2.moveWindow('suggestion_0_2', 250, 200)

    suggestion_0_3 = cv2.imread("finish_3.png")
    window_name_0_3 = "suggestion_0_3"
    cv2.imshow(window_name_0_3, suggestion_0_3)
    cv2.moveWindow('suggestion_0_3', 400, 200)

    suggestion_0_4 = cv2.imread("finish_4.png")
    window_name_0_4 = "suggestion_0_4"
    cv2.imshow(window_name_0_4, suggestion_0_4)
    cv2.moveWindow('suggestion_0_4', 550, 200)

    suggestion_0_5 = cv2.imread("finish_5.png")
    window_name_0_5 = "suggestion_0_5"
    cv2.imshow(window_name_0_5, suggestion_0_5)
    cv2.moveWindow('suggestion_0_5', 100, 350)

    suggestion_0_6 = cv2.imread("finish_6.png")
    window_name_0_6 = "suggestion_0_6"
    cv2.imshow(window_name_0_6, suggestion_0_6)
    cv2.moveWindow('suggestion_0_6', 250, 350)

    suggestion_0_7 = cv2.imread("finish_7.png")
    window_name_0_7 = "suggestion_0_7"
    cv2.imshow(window_name_0_7, suggestion_0_7)
    cv2.moveWindow('suggestion_0_7', 400, 350)

    suggestion_0_8 = cv2.imread("finish_8.png")
    window_name_0_8 = "suggestion_0_8"
    cv2.imshow(window_name_0_8, suggestion_0_8)
    cv2.moveWindow('suggestion_0_8', 550, 350)

    if cv2.waitKey(20) & 0xFF == 27:
        print("好きな画像を1~8で二つ選んでください")
        pick_1 = input("一つ目の好みの画像の番号:")
        pick_1 = int(pick_1)
        pick_2 = input("二つ目の好みの画像の番号:")
        pick_2 = int(pick_2)
        if (pick_1 >= 1) & (pick_1 <= 8):
            if (pick_2 >= 1) & (pick_2 <= 8):
                if pick_1 == pick_2:
                    print("同じ数字が入力されています")
                else:
                    break
        print("もう一度選択しなおしてください\nいずれかの画像をアクティブにしescキーで次に進みます")
        continue

cv2.destroyAllWindows()
print(pick_1, pick_2)
for i in range(3):
    print("ピックされた画像1のタイルNo:", tile_resource[pick_1 - 1][i])
for i in range(3):
    print("ピックされた画像2のタイルNo:", tile_resource[pick_2 - 1][i])

def set_picture_1_2(x, y, z):
    # 選択した解をfinish_1,2に入れなおす
    finish_1 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
    tile_relay(x[y][0], x[y][1], x[y][2], finish_1)
    cv2.imwrite('finish_1.png', finish_1)

    finish_2 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
    tile_relay(x[z][0], x[z][1], x[z][2], finish_2)
    cv2.imwrite('finish_2.png', finish_2)

set_picture_1_2(tile_resource, pick_1 - 1, pick_2 - 1)
cv2.imwrite('finish_1_f.png', finish_1)
cv2.imwrite('finish_2_f.png', finish_2)

# ================================遺伝的アルゴリズム===============================
# 交叉
cross_rand = random.randint(0, 98)
if cross_rand % 3 == 0:
    cross_son = [[tile_resource[pick_2 - 1][0], tile_resource[pick_1 - 1][1], tile_resource[pick_1 - 1][2]],
                 [tile_resource[pick_1 - 1][0], tile_resource[pick_2 - 1][1], tile_resource[pick_2 - 1][2]]]
elif cross_rand % 3 == 1:
    cross_son = [[tile_resource[pick_1 - 1][0], tile_resource[pick_2 - 1][1], tile_resource[pick_1 - 1][2]],
                 [tile_resource[pick_2 - 1][0], tile_resource[pick_1 - 1][1], tile_resource[pick_2 - 1][2]]]
else:
    cross_son = [[tile_resource[pick_1 - 1][0], tile_resource[pick_1 - 1][1], tile_resource[pick_2 - 1][2]],
                 [tile_resource[pick_2 - 1][0], tile_resource[pick_2 - 1][1], tile_resource[pick_1 - 1][2]]]

def cross_rand_made(a, b, c, d):
    cross_rand = random.randint(0, 98)
    if len(a) != 0:
        a.pop()
        a.pop()
    if cross_rand % 3 == 0:
        a.append([b[d][0], b[c][1], b[c][2]])
        a.append([b[c][0], b[d][1], b[d][2]])
    elif cross_rand % 3 == 1:
        a.append([b[c][0], b[d][1], b[c][2]])
        a.append([b[d][0], b[c][1], b[d][2]])
    else:
        a.append([b[c][0], b[c][1], b[d][2]])
        a.append([b[d][0], b[d][1], b[c][2]])

# 交叉確認用
print("交叉")
for i in range(2):
    print(cross_son[i][0], cross_son[i][1], cross_son[i][2])

def mutation_add_sort():
    # 突然変異
    # 行う個所数
    select_rand = [0, 0, 0]
    timese_rand = (random.randint(0, 98) % 3) + 1

    mutation_rand = random.randint(0, 99) % 2
    for i in range(3):
        select_rand[i] = random.randint(0, 98) % 3
    while (select_rand[0] == select_rand[1]):
        select_rand[1] = random.randint(0, 98) % 3
    while (select_rand[2] == select_rand[1]) | (select_rand[2] == select_rand[0]):
        select_rand[2] = random.randint(0, 98) % 3
    # time_rand確認用
    print("timese_rand:", timese_rand)
    # select_rand確認用
    print("select_rand:", select_rand[0], select_rand[1], select_rand[2])

    for i in range(timese_rand):
        plus_or_minus = random.randint(0, 99) % 2
        if cross_son[mutation_rand][select_rand[i]] == 8:
            cross_son[mutation_rand][select_rand[i]] = cross_son[mutation_rand][select_rand[i]] - 1
        elif cross_son[mutation_rand][select_rand[i]] == 0:
            cross_son[mutation_rand][select_rand[i]] = cross_son[mutation_rand][select_rand[i]] + 1
        elif plus_or_minus == 0:
            cross_son[mutation_rand][select_rand[i]] = cross_son[mutation_rand][select_rand[i]] - 1
        else:
            cross_son[mutation_rand][select_rand[i]] = cross_son[mutation_rand][select_rand[i]] + 1

    # 突然変異確認用
    print("突然変異後")
    for i in range(2):
        print(cross_son[i][0], cross_son[i][1], cross_son[i][2])

    # ソート
    # 最小,中間,最大でソート
    for i in range(2):
        for j in range(2):
            for k in range(j + 1, 3):
                if cross_son[i][j] > cross_son[i][k]:
                    cross_change = cross_son[i][j]
                    cross_son[i][j] = cross_son[i][k]
                    cross_son[i][k] = cross_change
    # 最小,最大,中間でソート
    for i in range(2):
        cross_change = cross_son[i][1]
        cross_son[i][1] = cross_son[i][2]
        cross_son[i][2] = cross_change

    # ソート確認用
    print("ソート")
    for i in range(2):
        print(cross_son[i][0], cross_son[i][1], cross_son[i][2])

mutation_add_sort()

def picture_store():
    # 子をfinish_3,4に格納
    finish_3 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
    tile_relay(cross_son[0][0], cross_son[0][1], cross_son[0][2], finish_3)
    cv2.imwrite('finish_3.png', finish_3)

    finish_4 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
    tile_relay(cross_son[1][0], cross_son[1][1], cross_son[1][2], finish_4)
    cv2.imwrite('finish_4.png', finish_4)

    print("いずれかの画像をアクティブにしescキーを押すと次に進みます")
    while 1:
        # 入力画像
        suggestion_0_1 = cv2.imread("finish_1.png")
        # 表示するWindow名
        window_name_0_1 = "suggestion_0_1"
        # 画像の表示
        cv2.imshow(window_name_0_1, suggestion_0_1)
        # Window位置の変更　第1引数：Windowの名前　第2引数：x 第3引数：y
        cv2.moveWindow('suggestion_0_1', 100, 200)

        suggestion_0_2 = cv2.imread("finish_2.png")
        window_name_0_2 = "suggestion_0_2"
        cv2.imshow(window_name_0_2, suggestion_0_2)
        cv2.moveWindow('suggestion_0_2', 250, 200)

        suggestion_0_3 = cv2.imread("finish_3.png")
        window_name_0_3 = "suggestion_0_3"
        cv2.imshow(window_name_0_3, suggestion_0_3)
        cv2.moveWindow('suggestion_0_3', 400, 200)

        suggestion_0_4 = cv2.imread("finish_4.png")
        window_name_0_4 = "suggestion_0_4"
        cv2.imshow(window_name_0_4, suggestion_0_4)
        cv2.moveWindow('suggestion_0_4', 550, 200)

        if cv2.waitKey(20) & 0xFF == 27:
            global pick_2_1
            global pick_2_2
            pick_2_1 = input("一番好みの画像の番号:")
            pick_2_1 = int(pick_2_1)
            pick_2_2 = input("二番目に好みの画像の番号:")
            pick_2_2 = int(pick_2_2)
            if (pick_2_1 >= 1) & (pick_2_1 <= 4):
                if (pick_2_2 >= 1) & (pick_2_2 <= 4):
                    if pick_2_1 == pick_2_2:
                        print("同じ数字が入力されています")
                    else:
                        break
            print("もう一度選択しなおしてください\nいずれかの画像をアクティブにしescキーで次に進みます")
            continue

    cv2.destroyAllWindows()
    print(pick_2_1, pick_2_2)

# pick_2_1,2ソート(小さい順)
#if pick_2_1 > pick_2_2:
#    pick_change = pick_2_1
#    pick_2_1 = pick_2_2
#    pick_2_2 = pick_change

picture_store()

# 適応度
fitness = 89

# 子が共に優秀であるとき
if ((pick_2_1 == 3) & (pick_2_2 == 4)) | ((pick_2_1 == 4) & (pick_2_2 == 3)):
    fitness_1 = ((tile_resource[pick_1 - 1][0]*tile_resource[pick_1 - 1][0])
                 + (tile_resource[pick_1 - 1][1] * tile_resource[pick_1 - 1][1])
                 + (tile_resource[pick_1 - 1][2] * tile_resource[pick_1 - 1][2])) / 3
    fitness_2 = ((tile_resource[pick_2 - 1][0] * tile_resource[pick_2 - 1][0])
                 + (tile_resource[pick_2 - 1][1] * tile_resource[pick_2 - 1][1])
                 + (tile_resource[pick_2 - 1][2] * tile_resource[pick_2 - 1][2])) / 3
    fitness_89_1_abs = abs(89 - fitness_1)
    fitness_89_2_abs = abs(89 - fitness_2)
    if fitness_89_1_abs <= fitness_89_2_abs:
        local_group = [[tile_resource[pick_1 - 1][0], tile_resource[pick_1 - 1][1], tile_resource[pick_1 - 1][2]],
                       [cross_son[0][0], cross_son[0][1], cross_son[0][2]],
                       [cross_son[1][0], cross_son[1][1], cross_son[1][2]]]

    else:
        local_group = [[tile_resource[pick_2 - 1][0], tile_resource[pick_2 - 1][1], tile_resource[pick_2 - 1][2]],
                       [cross_son[0][0], cross_son[0][1], cross_son[0][2]],
                       [cross_son[1][0], cross_son[1][1], cross_son[1][2]]]

# 親が共に優秀であるとき
if ((pick_2_1 == 1) & (pick_2_2 == 2)) | ((pick_2_1 == 2) & (pick_2_2 == 1)):
    fitness_1 = ((tile_resource[pick_1 - 1][0]*tile_resource[pick_1 - 1][0])
                 + (tile_resource[pick_1 - 1][1] * tile_resource[pick_1 - 1][1])
                 + (tile_resource[pick_1 - 1][2] * tile_resource[pick_1 - 1][2])) / 3
    fitness_2 = ((tile_resource[pick_2 - 1][0] * tile_resource[pick_2 - 1][0])
                 + (tile_resource[pick_2 - 1][1] * tile_resource[pick_2 - 1][1])
                 + (tile_resource[pick_2 - 1][2] * tile_resource[pick_2 - 1][2])) / 3
    fitness_89_1_abs = abs(89 - fitness_1)
    fitness_89_2_abs = abs(89 - fitness_2)
    # 解候補を新しく設定
    finish_1 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
    lt_no9, rt_no9, ct_no9 = rand_set(finish_1)
    if fitness_89_1_abs <= fitness_89_2_abs:
        local_group = [[tile_resource[pick_1 - 1][0], tile_resource[pick_1 - 1][1], tile_resource[pick_1 - 1][2]],
                       [lt_no9, rt_no9, ct_no9]]
    else:
        local_group = [[tile_resource[pick_2 - 1][0], tile_resource[pick_2 - 1][1], tile_resource[pick_2 - 1][2]],
                       [lt_no9, rt_no9, ct_no9]]

# どちらかの親一個が子より優秀(子1子1)親a>子a>(親bor子b)
if ((pick_2_1 == 1) & (pick_2_2 != 2)) | ((pick_2_1 == 2) & (pick_2_2 != 1)):
    fitness_1 = ((tile_resource[pick_1 - 1][0]*tile_resource[pick_1 - 1][0])
                 + (tile_resource[pick_1 - 1][1] * tile_resource[pick_1 - 1][1])
                 + (tile_resource[pick_1 - 1][2] * tile_resource[pick_1 - 1][2])) / 3
    fitness_2 = ((tile_resource[pick_2 - 1][0] * tile_resource[pick_2 - 1][0])
                 + (tile_resource[pick_2 - 1][1] * tile_resource[pick_2 - 1][1])
                 + (tile_resource[pick_2 - 1][2] * tile_resource[pick_2 - 1][2])) / 3
    fitness_89_1_abs = abs(89 - fitness_1)
    fitness_89_2_abs = abs(89 - fitness_2)
    if fitness_89_1_abs <= fitness_89_2_abs:
        local_group = [[tile_resource[pick_1 - 1][0], tile_resource[pick_1 - 1][1], tile_resource[pick_1 - 1][2]],
                       [cross_son[pick_2_2 - 3][0], cross_son[pick_2_2 - 3][1], cross_son[pick_2_2 - 3][2]]]

    else:
        local_group = [[tile_resource[pick_2 - 1][0], tile_resource[pick_2 - 1][1], tile_resource[pick_2 - 1][2]],
                       [cross_son[pick_2_2 - 3][0], cross_son[pick_2_2 - 3][1], cross_son[pick_2_2 - 3][2]]]

# どちらかの子一個が親より優秀(子1)子a>(親a or 親b)>子b
if ((pick_2_1 == 3) & (pick_2_2 != 4)) | ((pick_2_1 == 4) & (pick_2_2 != 3)):
    # 解候補を新しく設定
    finish_1 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
    lt_no9, rt_no9, ct_no9 = rand_set(finish_1)
    local_group = [[lt_no9, rt_no9, ct_no9],
                   [cross_son[pick_2_1 - 3][0], cross_son[pick_2_1 - 3][1], cross_son[pick_2_1 - 3][2]]]

def conditional_branch(x, y):
    relay_local_group = [[local_group[x][0], local_group[x][1], local_group[x][2]],
                         [local_group[y][0], local_group[y][1], local_group[y][2]]]
    local_group.pop(y)
    local_group.pop(x)
    fitness_1 = ((relay_local_group[0][0] * relay_local_group[0][0])
                 + (relay_local_group[0][1] * relay_local_group[0][1])
                 + (relay_local_group[0][2] * relay_local_group[0][2])) / 3
    fitness_2 = ((relay_local_group[1][0] * relay_local_group[1][0])
                 + (relay_local_group[1][1] * relay_local_group[1][1])
                 + (relay_local_group[1][2] * relay_local_group[1][2])) / 3
    fitness_89_1_abs = abs(89 - fitness_1)
    fitness_89_2_abs = abs(89 - fitness_2)
    # 子が共に優秀であるとき
    if ((pick_2_1 == 3) & (pick_2_2 == 4)) | ((pick_2_1 == 4) & (pick_2_2 == 3)):
        if fitness_89_1_abs <= fitness_89_2_abs:
            local_group.append([relay_local_group[0][0], relay_local_group[0][1], relay_local_group[0][2]])
            local_group.append([cross_son[0][0], cross_son[0][1], cross_son[0][2]])
            local_group.append([cross_son[1][0], cross_son[1][1], cross_son[1][2]])
        else:
            local_group.append([relay_local_group[1][0], relay_local_group[1][1], relay_local_group[1][2]])
            local_group.append([cross_son[0][0], cross_son[0][1], cross_son[0][2]])
            local_group.append([cross_son[1][0], cross_son[1][1], cross_son[1][2]])

    # 親が共に優秀であるとき
    if ((pick_2_1 == 1) & (pick_2_2 == 2)) | ((pick_2_1 == 2) & (pick_2_2 == 1)):
        # 解候補を新しく設定
        finish_1 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
        lt_no9, rt_no9, ct_no9 = rand_set(finish_1)
        if fitness_89_1_abs <= fitness_89_2_abs:
            local_group.append([relay_local_group[0][0], relay_local_group[0][1], relay_local_group[0][2]])
            local_group.append([lt_no9, rt_no9, ct_no9])
        else:
            local_group.append([relay_local_group[1][0], relay_local_group[1][1], relay_local_group[1][2]])
            local_group.append([lt_no9, rt_no9, ct_no9])

    # どちらかの親一個が子より優秀(子1子1)親a>子a>(親bor子b)
    if ((pick_2_1 == 1) & (pick_2_2 != 2)) | ((pick_2_1 == 2) & (pick_2_2 != 1)):
        if fitness_89_1_abs <= fitness_89_2_abs:
            local_group.append([relay_local_group[0][0], relay_local_group[0][1], relay_local_group[0][2]])
            local_group.append([cross_son[pick_2_2 - 3][0], cross_son[pick_2_2 - 3][1], cross_son[pick_2_2 - 3][2]])
        else:
            local_group.append([relay_local_group[1][0], relay_local_group[1][1], relay_local_group[1][2]])
            local_group.append([cross_son[pick_2_2 - 3][0], cross_son[pick_2_2 - 3][1], cross_son[pick_2_2 - 3][2]])

    # どちらかの子一個が親より優秀(子1)子a>(親a or 親b)>子b
    if ((pick_2_1 == 3) & (pick_2_2 != 4)) | ((pick_2_1 == 4) & (pick_2_2 != 3)):
        # 解候補を新しく設定
        finish_1 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
        lt_no9, rt_no9, ct_no9 = rand_set(finish_1)
        local_group.append([lt_no9, rt_no9, ct_no9])
        local_group.append([cross_son[pick_2_1 - 3][0], cross_son[pick_2_1 - 3][1], cross_son[pick_2_1 - 3][2]])

# 確認用
# print(local_group[0][0], local_group[0][1], local_group[0][2],)
# print(local_group[1][0], local_group[1][1], local_group[1][2],)
# print(local_group[2][0], local_group[2][1], local_group[2][2],)

for j in range(4):
    # 個体番号の選別
    local_group_rand_1 = random.randint(0, len(local_group) - 1) % len(local_group)
    local_group_rand_2 = random.randint(0, len(local_group) - 1) % len(local_group)
    while local_group_rand_1 == local_group_rand_2:
        local_group_rand_2 = random.randint(0, len(local_group) - 1) % len(local_group)

    # 小さい順ソート
    if local_group_rand_1 > local_group_rand_2:
        local_group_rand_change = local_group_rand_1
        local_group_rand_1 = local_group_rand_2
        local_group_rand_2 = local_group_rand_change

    # 交叉
    cross_rand_made(cross_son, local_group, local_group_rand_1, local_group_rand_2)
    print("親A")
    print(local_group[local_group_rand_1][0], local_group[local_group_rand_1][1], local_group[local_group_rand_1][2])
    print("親B")
    print(local_group[local_group_rand_2][0], local_group[local_group_rand_2][1], local_group[local_group_rand_2][2])
    # 交叉確認用
    print("交叉", j + 2, "回目")
    for i in range(2):
        print(cross_son[i][0], cross_son[i][1], cross_son[i][2])
    # 突然変異
    mutation_add_sort()
    # 親世代画像
    set_picture_1_2(local_group, local_group_rand_1, local_group_rand_2)
    # 子世代画像
    picture_store()
    if j == 3:
        break
    else:
        conditional_branch(local_group_rand_1, local_group_rand_2)

# 最終世代
finish_9 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
if pick_2_1 == 1:
    tile_relay(local_group[local_group_rand_1][0], local_group[local_group_rand_1][1],
               local_group[local_group_rand_1][2], finish_9)
elif pick_2_1 == 2:
    tile_relay(local_group[local_group_rand_2][0], local_group[local_group_rand_2][1],
               local_group[local_group_rand_2][2], finish_9)
elif pick_2_1 == 3:
    tile_relay(cross_son[0][0], cross_son[0][1], cross_son[0][2], finish_9)
else:
    tile_relay(cross_son[1][0], cross_son[1][1], cross_son[1][2], finish_9)
cv2.imwrite('finish_9.png', finish_9)

# 既存手法画像の生成
finish_0 = cv2.imread("input_c.bmp", cv2.IMREAD_COLOR)
tile_relay(2, 6, 4, finish_0)
cv2.imwrite('finish_0.png', finish_0)

# 既存と提案手法の比較
print("いずれかの画像をアクティブにしescキーを押すと次に進みます")
while 1:
    # 入力画像
    suggestion_0_1 = cv2.imread("finish_0.png")
    # 表示するWindow名
    window_name_0_1 = "suggestion_0_1"
    # 画像の表示
    cv2.imshow(window_name_0_1, suggestion_0_1)
    # Window位置の変更　第1引数：Windowの名前　第2引数：x 第3引数：y
    cv2.moveWindow('suggestion_0_1', 100, 200)

    suggestion_0_2 = cv2.imread("finish_9.png")
    window_name_0_2 = "suggestion_0_2"
    cv2.imshow(window_name_0_2, suggestion_0_2)
    cv2.moveWindow('suggestion_0_2', 250, 200)

    suggestion_0_3 = cv2.imread("finish_1_f.png")
    window_name_0_3 = "suggestion_0_3"
    cv2.imshow(window_name_0_3, suggestion_0_3)
    cv2.moveWindow('suggestion_0_3', 400, 200)

    suggestion_0_4 = cv2.imread("finish_2_f.png")
    window_name_0_4 = "suggestion_0_4"
    cv2.imshow(window_name_0_4, suggestion_0_4)
    cv2.moveWindow('suggestion_0_4', 550, 200)

    if cv2.waitKey(20) & 0xFF == 27:
        pick_2_1 = input("一番好みの画像の番号:")
        pick_2_1 = int(pick_2_1)
        if (pick_2_1 >= 1) & (pick_2_1 <= 4):
            break
        print("もう一度選択しなおしてください\nいずれかの画像をアクティブにしescキーで次に進みます")
        continue

cv2.destroyAllWindows()

if pick_2_1 == 1:
    print("従来")
elif pick_2_1 == 2:
    print("提案手法")
else:
    print("初期解")