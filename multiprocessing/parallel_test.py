# -*- coding:utf-8 -*-
"""
@Time  : 3/13/20 4:06 PM
@Author: Yongbo Wang
@Email : yongbowin@outlook.com
@File  : parallel_test.py
@Desc  : reference from https://www.cnblogs.com/linxiyue/p/10224004.html
"""

import requests
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Executor, \
    as_completed, wait, ALL_COMPLETED, FIRST_COMPLETED
import time

"""
concurrent.futures进行并发编程
"""

# -------------------------- 1.Executor与Future --------------------------
"""
concurrent.futures供了ThreadPoolExecutor和ProcessPoolExecutor两个类，都继承自Executor，
分别被用来创建线程池和进程池，接受max_workers参数，代表创建的线程数或者进程数。
ProcessPoolExecutor的max_workers参数可以为空，程序会自动创建基于电脑cpu数目的进程数。
"""


def load_url(url):
    return requests.get(url)


url = 'https://www.baidu.com'
executor = ThreadPoolExecutor(max_workers=1)
future = executor.submit(load_url, url)

"""
Executor中定义了submit()方法，这个方法的作用是提交一个可执行的回调task,并返回一个future实例。
future能够使用done()方法判断该任务是否结束，
done()方法是不阻塞的，
使用result()方法可以获取任务的返回值，这个方法是阻塞的。

这里说的‘不阻塞’是指会同时进行。
"""

print(future.done())  # False， 不阻塞，因为现在还没完成，所以返回False
print(future.result().status_code)  # 200
print(future.done())  # True, 不阻塞，现在请求已经完成，返回True

"""
Future类似于js中的Promise，可以添加回调函数：
>>> future.add_done_callback(fn)
回调函数fn在future取消或者完成后运行，参数是future本身。
"""

# -------------------------- 2.map()函数 ----------------------------------
"""
注意： submit()方法只能进行单个任务，用并发多个任务，需要使用map与as_completed。
"""
URLS = ['https://www.baidu.com', 'https://www.google.com',
        'https://github.com', 'https://fanyi.baidu.com',
        'https://www.jianshu.com']


def load_url(url):
    return requests.get(url)


with ThreadPoolExecutor(max_workers=3) as executor:
    for url, data in zip(URLS, executor.map(load_url, URLS)):
        print('%r page status_code %s' % (url, data.status_code))

"""
'https://www.baidu.com' page status_code 200
'https://www.google.com' page status_code 200
'https://github.com' page status_code 200
'https://fanyi.baidu.com' page status_code 200
'https://www.jianshu.com' page status_code 403

map方法接收两个参数，第一个为要执行的函数，第二个为一个序列，会对序列中的每个元素都执行这个函数，返回值为执行结果组成的生成器。
由上面可以看出返回结果与序列结果的顺序是一致的
"""

# -------------------------- 3.as_completed()函数 ------------------------- 多线程模拟并发操作
"""
as_completed()方法返回一个Future组成的生成器，在没有任务完成的时候，会阻塞，在有某个任务完成的时候，会yield这个任务，直到所有的任务结束。
"""


def load_url(url):
    return url, requests.get(url).status_code


start = time.time()
with ThreadPoolExecutor(max_workers=3) as executor:
    print("------> as_completed() ...")
    tasks = [executor.submit(load_url, url) for url in URLS]
    for future in as_completed(tasks):
        print(future.result())
end = time.time()
print('as_completed(), Took %.3f seconds.' % (end - start))

"""
('https://www.baidu.com', 200)
('https://fanyi.baidu.com', 200)
('https://www.jianshu.com', 403)
('https://www.google.com', 200)
('https://github.com', 200)

可以看出，结果与序列顺序不一致，先完成的任务会先通知主线程。
"""

# -------------------------- 4.wait()函数 ---------------------------------
"""
wait方法可以让主线程阻塞，直到满足设定的要求。有三种条件ALL_COMPLETED, FIRST_COMPLETED，FIRST_EXCEPTION。
"""


def load_url(url):
    requests.get(url)
    print(url)


start = time.time()
with ThreadPoolExecutor(max_workers=3) as executor:
    print("------> wait() ...")
    tasks = [executor.submit(load_url, url) for url in URLS]
    wait(tasks, return_when=ALL_COMPLETED)
    print('all_cone')
end = time.time()
print('wait(), Took %.3f seconds.' % (end - start))

"""
https://www.baidu.com
https://fanyi.baidu.com
https://www.jianshu.com
https://www.google.com
https://github.com
all_cone

可以看出阻塞到任务全部完成。
"""

# -------------------------- 5.多进程ProcessPoolExecutor -------------------
"""
使用ProcessPoolExecutor与ThreadPoolExecutor方法基本一致，注意文档中有一句：

The __main__ module must be importable by worker subprocesses. This means that ProcessPoolExecutor 
will not work in the interactive interpreter.

需要__main__模块
"""


def process_test():
    with ProcessPoolExecutor() as executor:
        print("------> ProcessPoolExecutor() ...")
        tasks = [executor.submit(load_url, url) for url in URLS]
        for f in as_completed(tasks):
            # print(f.result().status_code, f.done())
            ret = f.done()
            if ret:
                # print(f.result().status_code)
                print(f.result())


if __name__ == '__main__':
    process_test()
