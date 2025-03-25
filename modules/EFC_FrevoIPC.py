import socket
import struct
from google.protobuf.internal import encoder
from google.protobuf.internal import decoder

class FrevoInterface:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # enable address reuse
        # self.port = port
        try:
            self.sock.connect(('localhost', port))
        except socket.error as e:
            raise ConnectionError(f"Failed to connect to server on port {port}: {e}")
    
    # def reopen_channel(self):
    #     try:
    #         self.sock.connect(('localhost', self.port))
    #     except socket.error as e:
    #         raise ConnectionError(f"Failed to connect to server on port {self.port}: {e}")
    
    def close_channel(self):
        self.sock.close()
    
    def send_over_socket(self, message):
        delimiter = encoder._VarintBytes(len(message))
        message = delimiter + message
        msg_len = len(message)
        total_sent = 0
        while total_sent < msg_len:
            sent = self.sock.send(message[total_sent:])
            if sent == 0:
                raise RuntimeError('Socket connection broken')
            total_sent = total_sent + sent
    
    def receive_over_socket(self):
        varint_buff = []
        while True:
            byte = self.sock.recv(1)
            if byte:
                varint_buff.append(byte)
                # Check if we've received the full varint for length
                if not (ord(byte) & 0x80):  # MSB not set means end of varint
                    break
            else:
                raise RuntimeError('Socket connection broken')

        # Step 2: Decode the delimiter to get the message length
        msg_len, _ = decoder._DecodeVarint32(b''.join(varint_buff), 0)
        
        # Step 3: Read the full message based on the length
        message_data = b''
        while len(message_data) < msg_len:
            chunk = self.sock.recv(min(msg_len - len(message_data), 2048))
            if chunk == b'':
                raise RuntimeError('Socket connection broken')
            message_data += chunk

        return message_data
