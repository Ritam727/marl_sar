from cv2 import imshow, waitKey, destroyAllWindows, line, EVENT_LBUTTONDOWN, EVENT_MOUSEMOVE, EVENT_LBUTTONUP, setMouseCallback, namedWindow, resize, imwrite, INTER_AREA, resizeWindow, WINDOW_NORMAL
from numpy import array
from numpy.random import random

class MapCreator:
    def __init__(self, file, height = None, width = None):
        self.array = array([[[1.0] * 3] * 30] * 30)
        self.m_x = -1
        self.m_y = -1
        self.drawing = False
        self.changes = []
        self.file = file
        self.tgt = False
        namedWindow("Map Editor", WINDOW_NORMAL)
        setMouseCallback("Map Editor", self.draw)

    def render(self):
        while True:
            self.update()
            resizeWindow("Map Editor", 400, 400)
            imshow("Map Editor", self.array)
            k = waitKey(1) & 0xFF
            if k == ord("t"):
                self.tgt = not self.tgt
            if k == 27:
                break
        self.array *= 255
        imwrite(self.file, self.array)
        destroyAllWindows()

    def update(self):
        for change in self.changes:
            if not self.tgt:
                line(self.array, change[0], change[1], (0, 0, 0), 1)
            else:
                line(self.array, change[0], change[1], (1, 0, 0), 1)
        while len(self.changes) > 0:
            self.changes.pop()

    def draw(self, event, x, y, flags, param):
        if event == EVENT_LBUTTONDOWN:
            self.drawing = True
            self.m_x, self.m_y = x, y
        elif event == EVENT_MOUSEMOVE:
            if self.drawing:
                self.changes.append(((self.m_x, self.m_y), (x, y)))
                self.m_x, self.m_y = x, y
        elif event == EVENT_LBUTTONUP:
            self.drawing = False
            self.changes.append(((self.m_x, self.m_y), (x, y)))
