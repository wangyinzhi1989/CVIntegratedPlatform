import logging
import logging.handlers

class MyLogger():
    def __init__(self):
        self.logger = logging.getLogger("MyLogger")

    def setting(self, file, level, format, backupCount, interval):
        self.logger.setLevel(level)
        handler = logging.handlers.TimedRotatingFileHandler(filename=file, when='D', 
        interval=interval, backupCount=backupCount, encoding='utf-8')
        handler.setFormatter(logging.Formatter(format, '%Y-%m-%d %H:%M:%S', '%'))
        self.logger.addHandler(handler)

        # 控制台日志
        chlr = logging.StreamHandler()
        chlr.setFormatter(logging.Formatter(format, '%Y-%m-%d %H:%M:%S', '%'))
        self.logger.addHandler(chlr)

    def get_logger(self):
        return self.logger

