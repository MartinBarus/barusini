###################################################################
# Copyright (C) Martin Barus <martin.barus@gmail.com>
#
# This file is part of barusini.
#
# barusini can not be copied and/or distributed without the express
# permission of Martin Barus or Miroslav Barus
####################################################################
import os
import pickle


def get_terminal_size():
    try:
        _, size = os.popen("stty size", "r").read().split()
        return int(size)
    except ValueError:  # Running from Pycharm causes ValueError
        return 101


def save_object(o, path):
    with open(path, "wb") as file:
        pickle.dump(o, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(path):
    with open(path, "rb") as file:
        return pickle.load(file)
