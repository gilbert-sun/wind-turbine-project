#! python3
# import pyautogui, sys
import time

import pyautogui


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.



if __name__ == '__main__':
    print_hi('Auto Cursor Report!')
    print('======================== > Press Ctrl-C to quit. < ===========================')

    try:
        while True:
                currentMouseX, currentMouseY = pyautogui.position()
                print(currentMouseX, currentMouseY)
                time.sleep(1)
    except KeyboardInterrupt:
        print('\n')
