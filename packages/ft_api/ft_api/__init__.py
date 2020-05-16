from settings import PROJECTSPATH

with open(f"{PROJECTSPATH}/VERSION") as version_file:
    __version__ = version_file.read().strip()
