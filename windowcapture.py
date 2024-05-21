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
    """
    A class for capturing screenshots of a window.

    Args:
        window_name (str, optional): The name of the window to capture. If not provided, the entire screen will be captured.
        width (int, optional): The width of the window to capture. Only applicable if `window_name` is not provided.
        height (int, optional): The height of the window to capture. Only applicable if `window_name` is not provided.

    Attributes:
        window_name (str): The name of the window to capture.
        w (int): The width of the window.
        h (int): The height of the window.
        hwnd (int): The handle of the window.
        hwnd_dc (int): The device context (DC) for the window.
        save_dc (int): The device context (DC) for saving the captured screenshot.

    Raises:
        ValueError: If both `window_name` and `width` or `height` are provided.

    """

    # Section class variables
    window_name = None
    w = None
    h = None
    hwnd = None
    hwnd_dc = None
    save_dc = None


    # Section __init__ method
    class WindowCapture:
        def __init__(self, window_name=None, width=None, height=None):
            """
            Initializes a WindowCapture object.

            Args:
                window_name (str): The name of the window to capture. If None, the entire screen will be captured.
                width (int): The width of the window to capture. Only applicable if window_name is None.
                height (int): The height of the window to capture. Only applicable if window_name is None.

            Raises:
                ValueError: If both window_name and width/height are provided.

            """
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
        """
        Retrieves the screen size of the system.

        If the window name is not provided, it uses the win32api.GetSystemMetrics
        function to get the screen width and height. If the window name is provided,
        it raises a ValueError indicating that the `get_window_size` method should
        be used instead.

        Returns:
            tuple: A tuple containing the screen width and height.

        Raises:
            ValueError: If the window name is provided.
        """
        if self.window_name is None:
            if self.w is None:
                self.w = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            if self.h is None:
                self.h = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
            self.hwnd = None
        else:
            raise ValueError("Window name is provided. get_window_size should be used.")

    def get_window_handle(self):
        """
        Retrieves the handle of the window with the specified name.

        Returns:
            int: The handle of the window.

        Raises:
            ValueError: If the window name is not provided.
        """
        if self.window_name is not None:
            self.hwnd = win32gui.FindWindow(None, self.window_name)
        else:
            raise ValueError(
                "Window name is not provided. get_screen_size should be used."
            )

    def get_window_size(self):
        """
        Retrieves the size of the window.

        Raises:
            ValueError: If the window with the specified name is not found.

        Returns:
            Tuple[int, int]: The width and height of the window.
        """
        self.get_window_handle()
        if self.hwnd is None:
            raise ValueError(f"Window with name {self.window_name} not found.")
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        self.w = right - left
        self.h = bottom - top

    def get_window_dc(self):
        """
        Retrieves the device context (DC) for the window.

        Returns:
            The device context (DC) for the window.
        """
        self.hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        self.mfc_dc = win32ui.CreateDCFromHandle(self.hwnd_dc)
        self.save_dc = self.mfc_dc.CreateCompatibleDC()

    # Section get_screenshot method
    def get_screenshot(self):
        """
        Capture a screenshot of the window and return it along with the capture FPS.

        Returns:
            tuple: A tuple containing the captured screenshot in RGB format and the capture FPS.
        """
        capture_time = time()

        # Get window device context
        self.get_window_dc()

        # Create bitmap object
        save_bitmap = win32ui.CreateBitmap()
        # Create bitmap object with window size
        save_bitmap.CreateCompatibleBitmap(self.mfc_dc, self.w, self.h)
        # Select bitmap object to device context
        self.save_dc.SelectObject(save_bitmap)
        # Copy window image to bitmap object
        self.save_dc.BitBlt((0, 0), (self.w, self.h), self.mfc_dc, (0, 0), win32con.SRCCOPY)
        # Get bitmap object data
        bmp_str = save_bitmap.GetBitmapBits(True)
        # Convert bitmap object data to numpy array
        img = np.frombuffer(bmp_str, dtype="uint8")
        # Reshape numpy array to image shape
        img.shape = (self.h, self.w, 4)
        # Release resources
        self.mfc_dc.DeleteDC()
        self.save_dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, self.hwnd_dc)

        # make imageC_CONTIGUOUS
        img = np.ascontiguousarray(img) if not img.flags["C_CONTIGUOUS"] else img

        capture_fps = 1 / (time() - capture_time)
        # Convert image to RGB format and return
        return cv.cvtColor(img, cv.COLOR_RGBA2RGB), capture_fps
