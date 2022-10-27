#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import load_model
from tkinter import *
import tkinter as tk
import objc
import pygetwindow
import pyautogui
from PIL import Image, ImageOps, ImageGrab
import numpy as np
from appscript import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = load_model('mnistADAM.h5')
path = '/Users/vishnukunnummal/Desktop/result.png'
def predict_digit(img):
    #resize image to 28×28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = ImageOps.invert(img)
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Draw a digit", font=("Helvetica", 38))
        self.classify_btn = tk.Button(self, text = "Recognize", command = self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
    def clear_all(self):
        self.canvas.delete("all")
    def classify_handwriting(self):
        titles = pygetwindow.getAllTitles()
        #print(titles)
        x1, y1, width, height = pygetwindow.getWindowGeometry('python tk')
        #print(titles[len(titles) - 1])
        pyautogui.screenshot(path)
        im = Image.open(path)
        x2 = x1+1.25*width
        y2 = y1+2*height
        im = im.crop((x1, y1, x2, y2))
        im.save(path)
        digit, acc = predict_digit(im)
        self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')
#             a = self.canvas.winfo_rootx()
#             b = self.canvas.winfo_rooty()
#             c = (a * 75) + self.canvas.winfo_width()
#             d = (b * 7.5) + self.canvas.winfo_height()
#             rect = (a, b, c - 4, d - 4)
#             print(rect)
#             im = ImageGrab.grab(rect)
#             im.show()
#             digit, acc = predict_digit(im)
#             self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
app = App()
mainloop()


# In[ ]:




