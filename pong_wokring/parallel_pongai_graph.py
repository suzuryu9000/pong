# import threading
# from threading import Thread

# import time

import pg_pong
import live_graph


from multiprocessing import Process
import sys

rocket = 0

def func1():
    live_graph._run()

def func2():
    pg_pong._run()


if __name__=='__main__':
    p1 = Process(target = func1)
    p1.start()
    p2 = Process(target = func2)
    p2.start()
