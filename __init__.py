import sys
import os

"""Top-level package for raylight."""


this_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.join(this_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


from raylight.wanvideo.nodes_temp import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

__author__ = """Micko Lesmana"""
__email__ = "mickolesmana@gmail.com"
__version__ = "0.0.1"
