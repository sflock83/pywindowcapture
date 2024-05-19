# Section imports
from time import time

import cv2 as cv
import numpy as np
import win32api
import win32con
import win32gui
import win32ui


# Section class definition
class WindowCapture:
    # Section class variables
    window_name = None
    w = None
    h = None
    hwnd = None

    # Section __init__ method
    def __init__(self, window_name=None, width=None, height=None):
        if window_name is not None and (width is not None or height is not None):
            raise ValueError(
                "Either window_name or width and height should be provided, not both."
            )
        self.window_name = window_name
        self.w = width
        self.h = height
        self.hwnd = None
        if self.window_name is None:
            self.get_screen_size()
        else:
            self.get_window_handle()
            self.get_window_size()

    def get_screen_size(self):
        if self.window_name is None:
            if self.w is None:
                self.w = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            if self.h is None:
                self.h = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
            self.hwnd = None
        else:
            raise ValueError("Window name is provided. get_window_size should be used.")

    def get_window_handle(self):
        if self.window_name is not None:
            self.hwnd = win32gui.FindWindow(None, self.window_name)
        else:
            raise ValueError(
                "Window name is not provided. get_screen_size should be used."
            )

    def get_window_size(self):
        self.get_window_handle()
        if self.hwnd is None:
            raise ValueError(f"Window with name {self.window_name} not found.")
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        self.w = right - left
        self.h = bottom - top

    def get_window_dc(self):
        hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        self.save_dc = mfc_dc.CreateCompatibleDC()

    # Section get_screenshot method
    def get_screenshot(self):
        capture_time = time()

        # Get window device context

        # Create bitmap object
        save_bitmap = win32ui.CreateBitmap()
        # Create bitmap object with window size
        save_bitmap.CreateCompatibleBitmap(self.mfc_dc, self.width, self.height)
        # Select bitmap object to device context
        self.save_dc.SelectObject(save_bitmap)
        # Copy window image to bitmap object
        save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)
        # Get bitmap object data
        bmp_str = save_bitmap.GetBitmapBits(True)
        # Convert bitmap object data to numpy array
        img = np.frombuffer(bmp_str, dtype="uint8")
        # Reshape numpy array to image shape
        img.shape = (height, width, 4)
        # Release resources
        mfc_dc.DeleteDC()
        save_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        # make imageC_CONTIGUOUS
        img = np.ascontiguousarray(img) if not img.flags["C_CONTIGUOUS"] else img

        capture_fps = 1 / (time() - capture_time)
        # Convert image to RGB format and return
        return cv.cvtColor(img, cv.COLOR_RGBA2RGB), capture_fps

    # Section get_screen_position method
    # Section get_window_rect method
    # Section list_window_names method
    # Section list_window_rects method
    # Section list_window_names_and_rects method
    # Section __del__ method
