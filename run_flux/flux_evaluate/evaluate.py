#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.image_eval import main


if __name__ == "__main__":
    main()
