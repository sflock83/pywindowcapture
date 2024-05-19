# import os
from time import time

import cv2 as cv
import numpy as np
import win32con
import win32gui
import win32ui


def get_window_name():
    while True:
        # Get window name from list of windows
        window_list = []

        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) != "":
                window_list.append(win32gui.GetWindowText(hwnd))

        win32gui.EnumWindows(winEnumHandler, None)
        # Append None to list. This will allow user to capture full screen
        window_list.append("None. Capture full screen.")
        # Append Exit option to list. This will allow user to exit program
        window_list.append("Choose this to exit program.")
        # Let user choose window from list. print list of windows one by one
        print("List of windows")
        for i, window in enumerate(window_list):
            print(f"{i}: {window}")
        selection = input("Select window: ")
        try:
            window_name = window_list[int(selection)]
            if window_name == "None. Capture full screen.":
                window_name = None
            elif window_name == "Choose this to exit program.":
                print("Exit. User chose to exit program.")
                exit(0)
            return window_name
        except IndexError:
            print("Invalid input. Please enter a number from list.")
        except Exception as e:
            print(f"Error: {e}")


def window_capture(window_name=None):
    capture_time = time()
    # Get window handle
    if window_name is not None:
        hwnd = win32gui.FindWindow(None, window_name)
    else:
        hwnd = None
    if hwnd is None:
        hwnd = None
        left, top, right, bottom = 0, 0, 1920, 1080
    else:
        # Get window position and size
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    # Get window device context
    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    # Create bitmap object
    save_bitmap = win32ui.CreateBitmap()
    # Create bitmap object with window size
    save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
    # Select bitmap object to device context
    save_dc.SelectObject(save_bitmap)
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


if __name__ == "__main__":
    window_name = get_window_name()
    while True:
        screenshot_RGB2BGR, capture_fps = window_capture(window_name)

        cv.imshow("CV: Enshrouded", screenshot_RGB2BGR)

        print("FPS {}".format(capture_fps))

        if cv.waitKey(1) == ord("q"):
            cv.destroyAllWindows()
            break

    print("Done")
    exit(0)
