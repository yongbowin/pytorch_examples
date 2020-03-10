# -*- coding:utf-8 -*-
"""
@Time  : 3/10/20 10:53 AM
@Author: Yongbo Wang
@Email : yongbowin@outlook.com
@File  : PIL.py
@Desc  : reference from https://www.liaoxuefeng.com/wiki/897692888725344/966759628285152
"""

"""
PIL：Python Imaging Library，已经是Python平台事实上的图像处理标准库了。PIL功能非常强大，但API却非常简单易用。
Install:
    sudo dnf install python-imaging
"""

from PIL import Image


"""
# 打开一个jpg图像文件，注意是当前路径:
im = Image.open('test.jpg')
# 获得图像尺寸:
w, h = im.size
print('Original image size: %sx%s' % (w, h))
# 缩放到50%:
im.thumbnail((w//2, h//2))
print('Resize image to: %sx%s' % (w//2, h//2))
# 把缩放后的图像用jpeg格式保存:
im.save('thumbnail.jpg', 'jpeg')
"""


def imgthumbnail():
    """
    图像缩放操作
    :return:
    """
    # 打开一个图像文件
    im = Image.open('img/cat.png')
    print(im)  # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=241x295 at 0x7FB7FB844E10>
    # 获得图像尺寸
    w, h = im.size
    print('Original image size: %sx%s' % (w, h))
    # 缩放到50%
    im.thumbnail((w//2, h//2))
    print('Resize image to: %sx%s' % (w // 2, h // 2))
    # 把缩放后的图像用jpeg格式保存
    im.save('img/cat_thumbnail.jpeg')


imgthumbnail()
