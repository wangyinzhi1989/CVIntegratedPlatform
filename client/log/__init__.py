from .loger import MyLogger

# 全局log记录器
my_logger = MyLogger()
def log_setting(file, level, format, backupCount, interval):
    my_logger.setting(file, level, format, backupCount, interval)

# log记录对象
LOG = my_logger.get_logger()
