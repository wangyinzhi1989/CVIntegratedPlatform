import sys
import os
import paramiko as po
from log import LOG

class GetTrainingInfo:
    '''
    line format: label x1 y1 x2 y2
    '''
    def __init__(self, remote=None):
        self.remote = remote

    def get_info(self):
        pass
