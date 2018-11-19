import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import sys

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)


def animate(i):
    f =  open("error_log", "r")
    f.readline()
    pullData = f.read()

    dataArray = pullData.split('\n')
    xar = []
    yar = []
    for eachLine in dataArray:
        if len(eachLine) > 1:
            x, y = eachLine.split(',')
            xar.append((x))
            yar.append((float(y)))
    ax1.clear()
    ax1.plot(xar, yar)


def _run():
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()
# run()
