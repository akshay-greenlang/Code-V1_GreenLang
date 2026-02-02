# -*- coding: utf-8 -*-
"""
Standards Generators
GL-VCCI Scope 3 Platform
"""

from .esrs_e1 import ESRSE1Generator
from .cdp import CDPGenerator
from .ifrs_s2 import IFRSS2Generator
from .iso_14083 import ISO14083Generator

__all__ = ["ESRSE1Generator", "CDPGenerator", "IFRSS2Generator", "ISO14083Generator"]
