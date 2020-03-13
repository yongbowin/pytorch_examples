# -*- coding:utf-8 -*-
"""
@Time  : 3/13/20 3:30 PM
@Author: Yongbo Wang
@Email : yongbowin@outlook.com
@File  : multi_processing.py
@Desc  : reference from https://www.cnblogs.com/kangoroo/p/7628092.html
"""

import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Executor


# 求最大公约数（计算密集型任务）
def gcd(pair):
    a, b = pair
    low = min(a, b)
    for i in range(low, 0, -1):
        if a % i == 0 and b % i == 0:
            return i


numbers = [
    (1963309, 2265973), (1879675, 2493670), (2030677, 3814172),
    (1551645, 2229620), (1988912, 4736670), (2198964, 7876293)
]


# --------------------------- 1.不使用多线程/多进程 ---------------------------
def no_multi():
    start = time.time()
    results = list(map(gcd, numbers))
    print(results)
    end = time.time()
    print('No Multi, Took %.3f seconds.' % (end - start))  # Took 0.619 seconds.


# --------------------------- 2.使用多线程ThreadPoolExecutor -----------------
def multi_thread():
    """
    使用多线程来处理计算密集型任务.

    一个计算密集型函数，因为GIL的原因，多线程是无法提升效率的。同时，线程启动的时候，有一定的开销，与线程池进行通信，
    也会有开销，所以这个程序使用了多线程反而更慢了。
    """
    start = time.time()
    pool = ThreadPoolExecutor(max_workers=10)
    results = list(pool.map(gcd, numbers))
    print(results)
    end = time.time()
    print('Multi-Thread, Took %.3f seconds.' % (end - start))  # Took 0.626 seconds.


# --------------------------- 3.使用多进程ProcessPoolExecutor ----------------
def multi_process():
    """
    使用多进程来处理计算密集型任务.
    """
    start = time.time()
    pool = ProcessPoolExecutor(max_workers=2)
    results = list(pool.map(gcd, numbers))
    print(results)
    end = time.time()
    print('Multi-Process, Took %.3f seconds.' % (end - start))


"""
在两个CPU核心的机器上运行多进程程序，比其他两个版本都快。这是因为，ProcessPoolExecutor类会利用multiprocessing模块所提供的底层机制，完成下列操作：
    1）把numbers列表中的每一项输入数据都传给map。
    2）用pickle模块对数据进行序列化，将其变成二进制形式。
    3）通过本地套接字，将序列化之后的数据从煮解释器所在的进程，发送到子解释器所在的进程。
    4）在子进程中，用pickle对二进制数据进行反序列化，将其还原成python对象。
    5）引入包含gcd函数的python模块。
    6）各个子进程并行的对各自的输入数据进行计算。
    7）对运行的结果进行序列化操作，将其转变成字节。
    8）将这些字节通过socket复制到主进程之中。
    9）主进程对这些字节执行反序列化操作，将其还原成python对象。
    10）最后，把每个子进程所求出的计算结果合并到一份列表之中，并返回给调用者。
multiprocessing开销比较大，原因就在于：主进程和子进程之间通信，必须进行序列化和反序列化的操作。
"""


if __name__ == '__main__':
    # 1.不使用多线程/多进程
    no_multi()  # Took 0.619 seconds.

    # 2.使用多线程
    multi_thread()  # Took 0.626 seconds.

    # 3.使用多进程
    multi_process()  # Multi-Process, Took 0.362 seconds.

