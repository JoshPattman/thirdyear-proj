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

    def get_num_locations(self):
        self.soc.sendall(b"num;")
        num_str = ""
        while True:
            c_data = self.soc.recv(1)
            c = c_data.decode("utf-8")[0]
            if c == ";":
                break
            num_str += c
        return int(num_str)

    def close(self):
        self.soc.sendall(b"exit;")
        self.soc.close()


if __name__=="__main__":
    sc = SimulatedCamera()

    num_locations = sc.get_num_locations()
    for i in range(num_locations):
        sc.move_to(i)
        frame = sc.get_frame()
        image = Image.open(io.BytesIO(frame))
        image.save("./imgs/img%s.png"%i)

    sc.close()