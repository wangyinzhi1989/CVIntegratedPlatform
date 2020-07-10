import json

class Conf():
    def __init__(self, file):
        self.file = file
        self.parse()

    def parse(self):
        with open(self.file, "rb") as fstr:
            self.fjson = json.load(fstr)
            self.service = self.fjson["service"]
            self.logger = self.fjson["logger"]

            # 标注配置
            self.anno_conf = self.fjson["anno"]
            self.train_conf = self.fjson["train"]

    def save(self):
        with open(self.file, "w") as fstr:
            fstr.write(json.dumps(self.fjson))
