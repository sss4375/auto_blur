import json
from utils.mariaDBConnector import MariaDBConnector
from utils.ftpConnector import FtpConnector

dbConn = MariaDBConnector('resource/DBResources.json')

ftpConn = FtpConnector('resource/ftp.json')

dbConn.connection()

ftpConn.ftp_connection()

testList = dbConn.select_list("SELECT image_id, crackdown_id, image_level, crackdown_level, origin_file FROM crackdown_image_list_b WHERE origin_file LIKE '/%' LIMIT 10")

for image_id, crackdown_id, image_level, crackdown_level, origin_file in testList:
    if image_level == 0 and crackdown_level == 1:
        # 번호인식
        print(image_id, crackdown_id, image_level, crackdown_level, origin_file)
    ftpConn.ftp_download(origin_file)


dbConn.disconnection()
ftpConn.ftp.quit()

# print(testList)



#
# data = {}
#
# data['test'] = []
#
# data['test'].append({'asdf': 1234})
#
# with open("test.json", 'w', encoding='utf-8') as outfile:
#     json.dump(data, outfile, ensure_ascii=False)
#
#
