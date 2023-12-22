import ftplib
import json


class FtpConnector:
    def __init__(self, db_path):
        with open(db_path, 'r', encoding='utf-8') as file:
            ftp_data = json.load(file)
            self.host = ftp_data.get('Ip')
            self.port = int(ftp_data.get('Port'))
            self.user = ftp_data.get('UserId')
            self.passwd = ftp_data.get('UserPassword')

        self.ftp = ftplib.FTP()

    def ftp_connection(self):
        self.ftp.connect(self.host, self.port)

        self.ftp.login(self.user, self.passwd)

        self.ftp.cwd("")

    def ftp_download(self, origin_file, downloaded_file_path):
        with open(downloaded_file_path, 'wb') as local_file:
            self.ftp.retrbinary(f"RETR {origin_file}", local_file.write)

    def ftp_upload(self, local_file_path, remote_file_path):
        with open(local_file_path, 'rb') as local_file:
            self.ftp.storbinary(f"STOR {remote_file_path}", local_file)


def main():
    ftpConn = FtpConnector('../resource/ftp.json')

    ftpConn.ftp_connection()

    # ftpConn.ftp_download("/uploads/rexgen/consolidate/Seq3_20231001_071308_STEP011.jpg")

    ftpConn.ftp_upload('C:/Users/jskim/Documents/GitHub/auto_blur/json_temp/Seq1_20231001_070433_STEP011.json', 'Seq1_20231001_070433_STEP011.json')

    ftpConn.ftp.quit()


if __name__ == "__main__":
    main()
