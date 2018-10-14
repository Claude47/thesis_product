"""
    Claude Betz (BTZCLA001)

    main.py
    Launch MOT system
"""

import sys
from PyQt5.QtWidgets import QApplication
from front_end.gui import GUI

def main():
    # motion tracking application
    App = QApplication(sys.argv)
    Gui = GUI()
    sys.exit(App.exec_())

if __name__ == '__main__':
    main()
