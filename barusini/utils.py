###################################################################
# Copyright (C) Martin Barus <martin.barus@gmail.com>
#
# This file is part of barusini.
#
# barusini can not be copied and/or distributed without the express
# permission of Martin Barus or Miroslav Barus
####################################################################
import os


def get_terminal_size():
    try:
        _, size = os.popen("stty size", "r").read().split()
        return int(size)
    except ValueError:  # Running from Pycharm causes ValueError
        return 101
