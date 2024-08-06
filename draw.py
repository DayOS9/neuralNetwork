import tkinter as tk

import numpy as np
from PIL import Image, ImageGrab


class Draw:
    def __init__(self) -> None:
        self.results = None
        self.root = tk.Tk()
        self.root.title("Draw a number")
        self.root.geometry("300x300")

        topFrame = tk.Frame(self.root)
        topFrame.pack(side="top")

        self.saveButton = tk.Button(topFrame, text="Test", command=self.test)
        self.clearButton = tk.Button(topFrame, text="Clear", command=self.clear)
        self.nextButton = tk.Button(topFrame, text="Next", command=self.nextEpoch)
        self.saveButton.pack(side="right")
        self.clearButton.pack(side="right")
        self.nextButton.pack(side="left")

        self.canvas = tk.Canvas(self.root, width=300, height=280, background="black")
        self.canvas.bind("<Button-1>", self.savePos)
        self.canvas.bind("<B1-Motion>", self.addLine)
        self.canvas.pack(anchor="nw", fill="both", expand=1)

    def clear(self):
        self.canvas.delete("all")

    def test(self):
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        img = ImageGrab.grab((x + 3, y + 3, x1 - 3, y1 - 3)).convert("L")
        bw = img.point(lambda p: p > 128 and 255)

        bbox = bw.getbbox()
        drawing = bw.crop(bbox)
        drawing.thumbnail((20, 20))

        final_img = Image.new("L", (28, 28), 0)
        draw_w, draw_h = drawing.size
        upper_left_x = (28 - draw_w) // 2
        upper_left_y = (28 - draw_h) // 2
        final_img.paste(drawing, (upper_left_x, upper_left_y))

        img_data = np.array(final_img) / 255.0
        img_datas = img_data.reshape((784, 1))
        self.results = img_datas

    def nextEpoch(self):
        pass

    def savePos(self, event):
        global lastx, lasty
        lastx, lasty = event.x, event.y

    def addLine(self, event):
        self.canvas.create_line(
            (lastx, lasty, event.x, event.y),
            width=15,
            capstyle=tk.ROUND,
            joinstyle=tk.BEVEL,
            fill="white",
        )
        self.savePos(event)

    def start(self):
        self.root.mainloop()
        return self.results
