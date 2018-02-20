
# -*- coding: utf-8 -*-
"""
matplotlibでリアルタイムプロットする例

無限にsin関数をplotし続ける
"""
from __future__ import unicode_literals, print_function

import numpy as np
import matplotlib.pyplot as plt


def pause_plot():
    fig, ax = plt.subplots(1, 1)
    x = np.arange(-np.pi, np.pi, 0.1)
    y = np.sin(x)
    # 初期化的に一度plotしなければならない
    # そのときplotしたオブジェクトを受け取る受け取る必要がある．
    # listが返ってくるので，注意
    lines, = ax.plot(x, y)


    while 1:
        while True:
            data1, data2 = readAdc()
            if data1 == 0:
                break
        while True:
            data1, data2 = readAdc()
            if data1 >= 5:
                prevtime = time.perf_counter()
                data1, data2 = readAdc()
                break

        testdataset = np.empty((0,INPUT_NUM), np.float32)

        data = []

        for i in range(SAMPLES*REC_NUM):
            while prevtime + interval > time.perf_counter():
                pass
            prevtime += interval
            data1,data2= readAdc()
            data.append(data2)

        dataarray = np.array(data, dtype=np.float32) - (2**11)
        t = np.abs(np.fft.fft( dataarray[0:SAMPLES*REC_NUM].reshape([REC_NUM,SAMPLES]) )[:,0:INPUT_NUM]).astype(np.float32)
        testdataset = np.append(testdataset, t, axis = 0)



        result = sess.run(y, feed_dict={x: testdataset, keep_prob: 1.0})
        print("%s" % LABELS[np.argmax(result[0,:])], end=' ')
        for V,r in zip(LABELS, result[0,:]):
          print("%s: %5.3f" % (V, r), end=' ')
        print('')

    # ここから無限にplotする
    while True:
        # plotデータの更新
        x += 0.1
        y = np.sin(x)

        # 描画データを更新するときにplot関数を使うと
        # lineオブジェクトが都度増えてしまうので，注意．
        #
        # 一番楽なのは上記で受け取ったlinesに対して
        # set_data()メソッドで描画データを更新する方法．
        lines.set_data(x, y)

        # set_data()を使うと軸とかは自動設定されないっぽいので，
        # 今回の例だとあっという間にsinカーブが描画範囲からいなくなる．
        # そのためx軸の範囲は適宜修正してやる必要がある．
        ax.set_xlim((x.min(), x.max()))

        # 一番のポイント
        # - plt.show() ブロッキングされてリアルタイムに描写できない
        # - plt.ion() + plt.draw() グラフウインドウが固まってプログラムが止まるから使えない
        # ----> plt.pause(interval) これを使う!!! 引数はsleep時間
        plt.pause(.01)

if __name__ == "__main__":
    pause_plot()
