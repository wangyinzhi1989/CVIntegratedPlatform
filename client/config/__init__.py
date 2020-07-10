from .config import Conf
import os

path = os.path.abspath(".") + "\\res\\config.cfg"
CONF = Conf(path)
