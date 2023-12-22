import io

import colorsys, onnxruntime, time, logging
from PIL import Image
import numpy as np


def calculate_distance(box):
    x_center = (box[0] + box[2]) / 2  # x 좌표 중앙값 계산
    y_center = (box[1] + box[3]) / 2  # y 좌표 중앙값 계산
    distance = np.sqrt((x_center - 960) ** 2 + (y_center - 540) ** 2)  # 중앙 좌표 (960, 540)과의 거리 계산
    return distance


class DetectionOnnx(object):
    def __init__(self, path_model, path_classes, image_shape):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # providers = ['CPUExecutionProvider']
        # session_options = onnxruntime.SessionOptions()
        # session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        # self.session = onnxruntime.InferenceSession(path_model, providers=providers, session_options=session_options)
        self.session = onnxruntime.InferenceSession(path_model, providers=providers)
        print(self.session.get_providers())
        self.class_labels, self.num_names = self.get_classes(path_classes)
        self.image_shape = image_shape
        # self.font = ImageFont.truetype('font.otf', 10)
        self.colors()

    def colors(self):
        hsv_tuples = [(x / len(self.class_labels), 1., 1.) for x in range(len(self.class_labels))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        class_colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        np.random.seed(43)
        np.random.shuffle(colors)
        np.random.seed(None)
        self.class_colors = np.tile(class_colors, (16, 1))

    def cvtColor(self, image):
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image
        else:
            image = image.convert('RGB')
            return image

    def get_classes(self, classes_path):
        with open(classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names, len(class_names)

    def preprocess_input(self, image):
        image /= 255.0
        image -= np.array([0.485, 0.456, 0.406])
        image /= np.array([0.229, 0.224, 0.225])
        return image

    def resize_image(self, image, size):
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image

    def inference(self, image, size):
        ort_inputs = {self.session.get_inputs()[0].name: image, self.session.get_inputs()[1].name: size}
        box_out, scores_out, classes_out = self.session.run(None, ort_inputs)
        return box_out, scores_out, classes_out

    # def draw_detection(self, image, boxes_out, scores_out, classes_out):
    #     image_pred = image.copy()
    #     thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.image_shape), 1))
    #     for i, c in reversed(list(enumerate(classes_out))):
    #         draw = ImageDraw.Draw(image_pred)
    #         predicted_class = self.class_labels[c]
    #         box = boxes_out[i]
    #         score = scores_out[i]
    #         label = '{}:{:.2f}%'.format(predicted_class, score * 100)
    #         label_size = draw.textsize(label, self.font)
    #         top, left, bottom, right = box
    #         top = max(0, np.floor(top + 0.5).astype('int32'))
    #         left = max(0, np.floor(left + 0.5).astype('int32'))
    #         bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    #         right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    #         if top - label_size[1] >= 0:
    #             text_origin = np.array([left, top - label_size[1]])
    #         else:
    #             text_origin = np.array([left, top + 1])
    #         draw.rectangle([left, top, right, bottom], outline=tuple(self.class_colors[c]), width=2)
    #         draw.text(text_origin, label, fill=(255, 255, 0), font=self.font)
    #         del draw
    #     return np.array(image_pred)

    def predict_image(self, image):
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        image = self.cvtColor(image)
        image_data = self.resize_image(image, (self.image_shape[1], self.image_shape[0]))
        image_data = np.expand_dims(self.preprocess_input(np.array(image_data, dtype='float32')), 0)
        start_time = time.time()
        box_out, scores_out, classes_out = self.inference(image_data, input_image_shape)
        end_time = time.time()
        print(f"   ONNX : {end_time - start_time:.5f} sec")
        # if __name__ == "__main__":
        #     image_pred = self.draw_detection(image, box_out, scores_out, classes_out)
        #     return np.array(image_pred)
        return box_out, scores_out, classes_out

    # def detection_video(self, video_path: str, output_path: str, fps=25):
    #     cap = cv2.VideoCapture(video_path)
    #     ret, frame = cap.read()
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     frame_height, frame_width, _ = frame.shape
    #     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret: break
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         frame = Image.fromarray(np.uint8(frame))
    #         output = self.predict_image(frame, True)
    #         output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    #         out.write(output)
    #     out.release()


def main():
    from sunapi import Cam
    host = "http://192.168.123.233/"
    username = "admin"
    password = "samsungg2b!"

    x55cam = Cam(host, 80, username, password)

    logging.basicConfig(filename=f'../log/app.log', filemode='a', format='%(asctime)s - %(message)s',
                        level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')
    # path_model = '../weights/model.onnx'
    path_model = '../weights/recognition-s.onnx'
    path_output = 'output.jpg'

    args = {"path_model": path_model, "path_classes": '../labels/license_classes.txt',
            "image_shape": (640, 640)}
    cls = DetectionOnnx(**args)

    start_time = time.time()
    logging.info(f'load model {path_model}')

    end_time = time.time() - start_time
    end_time = end_time / 60
    while True:
        start_time2 = time.time()
        resp = x55cam.snap_shot().content
        data_io = io.BytesIO(bytearray(resp))
        input_image = Image.open(data_io)
        output_image = cls.predict_image(input_image)
        end_time2 = time.time() - start_time2
        print(end_time2)

        if 0 in output_image[2]:
            distances = [calculate_distance(box) for box in output_image[0]]

            closest_box_index = np.argmin(distances)

            lp_y1, lp_x1, lp_y2, lp_x2 = output_image[0][closest_box_index]
            print("번호판 발견", lp_x1, lp_y1, lp_x2, lp_y2)
            # x55cam.area_zoom(int((lp_x1 + lp_x2) / 2), int((lp_y1 + lp_y2) / 2),
            #                  int((lp_x1 + lp_x2) / 2), int((lp_y1 + lp_y2) / 2),
            #                  1920, 1080)
            time.sleep(2)
        else:
            print("번호판 인식 불가")
            time.sleep(2)

    # output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(path_output, output_image)
    logging.info(f'[INFO]: save out {path_output}')


if __name__ == "__main__":
    main()
