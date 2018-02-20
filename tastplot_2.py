
# -*- coding: utf-8 -*-
"""
matplotlibでリアルタイムプロットする例

無限にsin関数をplotし続ける
"""
from __future__ import unicode_literals, print_function

import numpy as np
import matplotlib.pyplot as plt


def pause_plot():

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax2 = fig.add_subplot(111)


    x1 = np.arange(-np.pi, np.pi, 0.1)
    x2 = np.arange(-np.pi, np.pi, 0.1)
    y1 = np.sin(x1)
    y2 = np.cos(x2)
    # 初期化的に一度plotしなければならない
    # そのときplotしたオブジェクトを受け取る受け取る必要がある．
    # listが返ってくるので，注意
    lines1, = ax1.plot(x1, y1)
    lines2, = ax2.plot(x2, y2)


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
        lines1.set_data(x1, y1)
        lines2.set_data(x2, y2)##個々の部分頑張れば行けそう





        # set_data()を使うと軸とかは自動設定されないっぽいので，
        # 今回の例だとあっという間にsinカーブが描画範囲からいなくなる．
        # そのためx軸の範囲は適宜修正してやる必要がある．
        ax1.set_xlim((x1.min(), x1.max()))
        ax2.set_xlim((x2.min(), x2.max()))

        # 一番のポイント
        # - plt.show() ブロッキングされてリアルタイムに描写できない
        # - plt.ion() + plt.draw() グラフウインドウが固まってプログラムが止まるから使えない
        # ----> plt.pause(interval) これを使う!!! 引数はsleep時間
        plt.pause(.01)

if __name__ == "__main__":
    pause_plot()
