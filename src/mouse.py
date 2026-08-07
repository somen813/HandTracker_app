import pyautogui
x, y = pyautogui.position()
print(x,y)

pyautogui.moveTo(500,300, duration=1)
pyautogui.move(100,200, duration=2)
pyautogui.rightClick()