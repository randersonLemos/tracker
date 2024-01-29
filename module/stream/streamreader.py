import os
import cv2
import time
import queue
from threading import Thread
from argparse import Namespace



try:
    from module.mylog import MyLog
except ModuleNotFoundError:
    from backend_parent.module.mylog import MyLog


class StreamReader(MyLog, Thread):
    def __init__(self, _id, addr, name='', mat=None):
        super().__init__()

        if mat is not None:
            self._name = 'Dev'
            self._id = 1
            self.addr = 'Dev'

            self._mat = mat
            self.cap = Namespace()
            self.cap.grab = self.fake_grab
            self.cap.retrieve = self.fake_retrieve
        else:
            self._name = str(name)
            self._id = int(_id)
            self.addr = addr

            self.cap = cv2.VideoCapture(self.addr)

        self.q = queue.Queue()
        self._run = True
        self._read = False


    def fake_grab(self):
        time.sleep(0.1)
        return True


    def fake_retrieve(self):
        return True, self._mat


    def run(self): # Read frames as soon as they are available, keeping only most recent one
        self.myprint('Running! pid: {}'.format(os.getpid()))

        while self._run:
            ret = self.cap.grab()

            if not ret:  # lost connection, must retry
                self.myprint('Failed to read frame from {} ({}):'.format(self._id, self.addr))
                self.myprint('Reloading camera')
                self.cap.release()
                time.sleep(3)

                if self.addr == '0':
                    self.addr = 0

                self.cap = cv2.VideoCapture(self.addr)

                self.myprint('Reloading camera done')
                continue

            if self._read:
                try:
                    self.q.get_nowait()  # Discard previous (unprocessed) frame
                except queue.Empty:
                    pass
                ret, frame = self.cap.retrieve()
                if ret:
                    self.q.put(frame)
                    self._read = False

        self.myprint('Done')


    def read(self):
        self._read = True
        return self._name, self._id, self.addr, self.q.get()


    def stop(self):
        self._run = False
