import socket
import io
import PIL.Image as Image

class SimulatedCamera:
    def __init__(self, host="localhost", port=8088):
        self.soc = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        self.soc.connect((host, port))
    def get_frame(self):
        self.soc.sendall(b"data;")
        size_str = ""
        while True:
            c_data = self.soc.recv(1)
            c = c_data.decode("utf-8")[0]
            if c == ";":
                break
            size_str += c
        size = int(size_str)
        frame_data = self.soc.recv(size)
        return frame_data
    def move_to(self, location):
        s = "move;%s;"%location
        self.soc.sendall(s.encode("utf-8"))
    def close(self):
        self.soc.sendall(b"exit;")
        self.soc.close()

if __name__=="__main__":
    sc = SimulatedCamera()

    frame = sc.get_frame()
    image = Image.open(io.BytesIO(frame))
    image.save("./img.png")

    sc.move_to(2)
    frame = sc.get_frame()
    image = Image.open(io.BytesIO(frame))
    image.save("./img2.png")

    sc.close()