from matplotlib import pyplot as plt
import numpy as np


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def make_lines(xx, yy, slope):
    lines = []
    for i in range(len(xx)):
        lines.append(MyLine(xx[i], yy[i], slope[i]))
    return lines


class MyLine:
    def __init__(self, x, y, slope):
        self.x = x
        self.y = y
        self.slope = slope

    def plotLine(self, color):
        plt.axline((self.x, self.y), color=color, slope=self.slope)

    def plotPoint(self, color, marker):
        plt.plot(self.x, self.y, color=color, marker=marker)

    def getYatX(self, x):
        return self.y + self.slope * (x - self.x)


def lines_are_touching(left, right):
    if left.x_start == right.x_end and left.y_start == right.y_end:
        return True
    elif left.x_end == right.x_start and left.y_end == right.y_start:
        return True
    else:
        return False


class LineSegment:
    def __init__(self, x_start, x_end, y_start, y_end):
        self.x_start = x_start
        self.x_end = x_end

        self.y_start = y_start
        self.y_end = y_end

        self.length = -1
        self.slope = 0
        self.calcParams()

    def calcParams(self):
        self.length = dist(self.x_start, self.y_start, self.x_end, self.y_end)
        if self.x_end != self.x_start:
            self.slope = (self.y_end - self.y_start) / (self.x_end - self.x_start)
        else:
            self.slope = np.inf

    def dist2point(self, x, y):
        return min(dist(self.x_start, self.y_start, x, y), dist(self.x_end, self.y_end, x, y))

    def addPoint(self, x, y):
        self.x_end = x
        self.y_end = y
        self.calcParams()

    def plotSegment(self, color, linewidth):
        plt.plot([self.x_start, self.x_end], [self.y_start, self.y_end], color=color, linewidth=linewidth)
