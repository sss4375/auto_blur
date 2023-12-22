import time
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import ToTensor


class Detection:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        print(self.device)
        self.model.to(self.device)
        self.model.eval()
        # self.labels = None
        # with open("coco_labels.txt", "r") as f:
        #     self.labels = [line.strip() for line in f.readlines()]

    def detection(self, input_image):
        # 이미지 전처리
        transform = ToTensor()
        image = transform(input_image)

        # GPU 메모리에 데이터를 올리기
        image = image.to(self.device)

        # 모델로부터 객체 탐지 수행
        start_time = time.time()
        output = self.model([image])
        end_time = time.time()
        print(f"pytorch : {end_time - start_time:.5f} sec")

        return output
