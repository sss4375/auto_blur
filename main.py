import time

from detection.pt_object_detection import Detection
from detection.onnx_license_detection import DetectionOnnx
from utils.mariaDBConnector import MariaDBConnector
from utils.ftpConnector import FtpConnector
from PIL import Image, ImageDraw

import json
import os

# 사람 객체 인식
car_detection = Detection()

# 자동차 번호판 객체 인식
path_model = 'weights/recognition-s.onnx'

args = {"path_model": path_model, "path_classes": 'labels/license_classes.txt',
        "image_shape": (640, 640)}

onnx_detection = DetectionOnnx(**args)

# DB 연결 (MariaDB)
dbConn = MariaDBConnector('resource/DBResources.json')


# FTP 연결
ftpConn = FtpConnector('resource/ftp.json')


while True:
    dbConn.connection()
    ftpConn.ftp_connection()
    # mask_file == null 인 data select
    testList = dbConn.select_list("SELECT image_id, crackdown_id, image_level, crackdown_level, origin_file "
                                  "  FROM " + dbConn.table_name +
                                  # " WHERE crackdown_id = '280024'")
                                  " WHERE mask_file IS NULL LIMIT 1000")
    # json data
    data = {}

    for image_id, crackdown_id, image_level, crackdown_level, origin_file in testList:
        # if image_level == 0 and crackdown_level == 1:
        #     # 번호 재인식 개발부
        #     print(image_id, crackdown_id, image_level, crackdown_level, origin_file)
        try:
            filename = origin_file.split('/')
            downloaded_file_path = "temp/" + filename[len(filename) - 1]
            ftpConn.ftp_download(origin_file, downloaded_file_path)
            img = Image.open(downloaded_file_path)
        except Exception as ex:
            print(ex)
            update_query = "UPDATE " + dbConn.table_name + " SET mask_file = 'None' WHERE image_id = " + str(image_id)
            dbConn.update_list(update_query)
            continue

        output = car_detection.detection(img)

        result_boxes = []

        data['Person'] = []
        data['CarPlates'] = []

        for i in range(len(output[0]['labels'])):
            bbox = output[0]['boxes'][i].detach().cpu().numpy()
            conf = output[0]['scores'][i].detach().cpu().numpy()
            if conf > 0.85 and output[0]['labels'][i] == 1:
                data['Person'].append({'EndX': int(bbox[2]), 'EndY': int(bbox[3]), 'Score': float(conf), 'StartX': int(bbox[0]), 'StartY': int(bbox[1])})

        license_bbox, license_score, license_classes = onnx_detection.predict_image(img)

        for i in range(len(license_bbox)):
            if license_score[i] > 0.75:
                lp_y1, lp_x1, lp_y2, lp_x2 = license_bbox[i]
                data['CarPlates'].append({'EndX': int(lp_x2), 'EndY': int(lp_y2), 'Score': float(license_score[i]), 'StartX': int(lp_x1), 'StartY': int(lp_y1)})

        file_name = os.path.basename(downloaded_file_path)
        file_base_name = os.path.splitext(file_name)[0]

        json_name = file_base_name + '.json'
        json_file_path = "json_temp/" + json_name
        upload_file_path = 'test/' + json_name
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        with open(json_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False)

        ftpConn.ftp_upload('./json_temp/' + json_name, upload_file_path)

        update_query = "UPDATE " + dbConn.table_name + " SET mask_file = '" + upload_file_path + "' WHERE image_id = " + str(image_id)
        dbConn.update_list(update_query)

        os.remove(downloaded_file_path)
        os.remove(json_file_path)

    dbConn.disconnection()
    ftpConn.ftp.quit()
    time.sleep(1)
    print("1사이클 완료")
