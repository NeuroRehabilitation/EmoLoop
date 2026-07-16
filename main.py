# UI/__init__.py or main app file
from sensors.ECG import ECG
from UI.pages.dashboard import Dashboard
import tkinter as tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.ecg = ECG()  # instantiate once
        self.dashboard = Dashboard(self, self.ecg)

if __name__ == '__main__':
    app = App()
    app.mainloop()