# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/4 17:27
@Auth ： KECILIMU
@File ：log.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
# Name: log
# Time: 2024/9/21
# Auther: "Kecilimu"
import File_log.file_create as fc
import File_log.IOstream as ios


class log:
    def __init__(self):
        self.time = fc.init_time()
        self.author = "Kecilimu"
        self.part_name = "log"
        self.file_name = self.time
        self.file_path = fc.create_log_path(self.file_name)
        self.model_path = fc.create_model_path(self.file_name)
        self.result_path = fc.create_result_path(self.file_name)

    def log_write(self, content):
        time = fc.init_time()
        content = f"# Name: {self.part_name}\n# Time: {time}\n# Auther: \"{self.author}\"\n\n" + content
        return fc.note(self.file_path, content)

    def model_write(self, model):
        content = f"Model is saved successfully."
        self.log_write(content)
        return ios.model_output(self.model_path, model)

    def save(self, obj, path):
        content = f"\tSave {path}"
        self.log_write(content)
        return ios.save(obj, path)

    def load(self, path):
        content = f"\tLoad {path}"
        self.log_write(content)
        return ios.load(path)

    def get_result_path(self):
        return self.result_path

    def img_write(self, fig, path):
        return ios.img_save(fig, path)
