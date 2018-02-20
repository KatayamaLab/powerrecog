
# -*- coding: utf-8 -*-
"""
matplotlibでリアルタイムプロットする例

無限にsin関数をplotし続ける
"""
from __future__ import unicode_literals, print_function

import numpy as np
import matplotlib.pyplot as plt


def pause_plot():

    x1 = np.arange(-np.pi, np.pi, 0.1)
    x2 = np.arange(-np.pi, np.pi, 0.1)
    y1 = np.sin(x1)
    y2 = np.cos(x2)
    # 初期化的に一度plotしなければならない
    # そのときplotしたオブジェクトを受け取る受け取る必要がある．
    # listが返ってくるので，注意
    plt.plot(x1, y1)
    plt.plot(x2, y2)


    # ここから無限にplotする
    while True:
        # plotデータの更新
        x1 += 0.1
        x2 += 0.1
        y1 = np.sin(x1)
        y2 = np.cos(x2)

        # 描画データを更新するときにplot関数を使うと
        # lineオブジェクトが都度増えてしまうので，注意．
        #
        # 一番楽なのは上記で受け取ったlinesに対して
        # set_data()メソッドで描画データを更新する方法．
        plt.plot(x1, y1)
        plt.plot(x2, y2)





        # set_data()を使うと軸とかは自動設定されないっぽいので，
        # 今回の例だとあっという間にsinカーブが描画範囲からいなくなる．
        # そのためx軸の範囲は適宜修正してやる必要がある．
        

        # 一番のポイント
        # - plt.show() ブロッキングされてリアルタイムに描写できない
        # - plt.ion() + plt.draw() グラフウインドウが固まってプログラムが止まるから使えない
        # ----> plt.pause(interval) これを使う!!! 引数はsleep時間
        plt.pause(.01)

if __name__ == "__main__":
    pause_plot()
