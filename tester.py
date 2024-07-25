import tkinter as tk

import numpy as np
from PIL import Image, ImageGrab, ImageTk


def savePosn(event):
    global lastx, lasty
    lastx, lasty = event.x, event.y


def addLine(event):
    canvas.create_line(
        (lastx, lasty, event.x, event.y), width=5, capstyle=tk.ROUND, joinstyle=tk.BEVEL
    )
    savePosn(event)


def saveImg():
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + 300
    y1 = y + 280
    img = ImageGrab.grab().crop((x, y, x1, y1)).convert("L")
    img = img.resize((28, 28))
    img_data = np.array(img) / 255.0
    print(img_data)


root = tk.Tk()
root.title("Draw a number!")
root.resizable(False, False)
root.geometry("300x300")

# creating button
button = tk.Button(root, text="Save", command=saveImg)
button.pack()

canvas = tk.Canvas(root, width=300, height=280)
canvas.pack(anchor="nw", fill="both", expand=1)
canvas.bind("<Button-1>", savePosn)
canvas.bind("<B1-Motion>", addLine)

root.mainloop()
