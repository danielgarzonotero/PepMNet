import torch

class device_info:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.device_str = "\n ///////// Running on the GPU /////////"
        else:
            self.device = torch.device("cpu")
            self.device_str = "\n //////// Running on the CPU ////////"

    def __str__(self):
        return self.device_str

    def print_device_info(self):
        print("Device: {}".format(self.device))


