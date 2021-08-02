import os
import pathlib


def version() -> str:
    current_file = pathlib.Path(__file__)
    version_file = os.path.join(current_file.parent.parent.parent, "VERSION")
    with open(version_file) as file:
        return file.readline()
