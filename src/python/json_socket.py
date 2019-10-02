"""
/*
    Copyright 2018 0kims association.

    This file is part of cusnarks.

    cusnarks is a free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your option)
    any later version.

    cusnarks is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
    more details.

    You should have received a copy of the GNU General Public License along with
    cusnarks. If not, see <https://www.gnu.org/licenses/>.
*/

#  NOTES:
//
//
// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : jsons_socket.py
//
// Date       : 01/08/2019
//
// ------------------------------------------------------------------
//
// NOTES:

# 
# Simple TCP server that sends json data
# 

// Description:
//    
//    
// ------------------------------------------------------------------

"""
import socket
import json
import struct

class jsonSocket(object):
   def __init__(self, host=None, port=None):
     if host is not None:
        self.host = host
     else :
        self.host = 'localhost'

     if port is not None:
       self.port = port
     else :
       self.port = 8192

     self.buffer_size = 1024

   def sever_socket(self):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((self.host,self.port))
    s.listen(1)
    while True: # Accept connections from multiple clients
         conn, addr = s.accept()
         msg = self.receive_message(conn)
         if len(msg):
           # Call some action and return results
           new_msg = {'status' : 'received'}
           self.send_message(new_msg, conn)
           conn.close()
              

   def receive_message(self,conn):
      new_msg = True
      msglen = 0
      msg = ""
      pending_bytes = 0
      while True: # Accept multiple messages from each client
         if new_msg:
            buffer = conn.recv(4)
            if buffer == b'':
               break
            msglen = struct.unpack("I",buffer)[0]
            pending_bytes = msglen
            new_msg = False
         else:
            bytes_to_read = min(self.buffer_size, pending_bytes)
            buffer = conn.recv(bytes_to_read)
            if buffer.decode():
               msg += buffer.decode()
               pending_bytes-= bytes_to_read
               if pending_bytes == 0:
                 break
      return msg


   def send_message(self,data, conn=None):
      json_data = json.dumps(data, sort_keys=False, indent=2)
      header = struct.pack("I",len(json_data))
      data=[]
      if conn is None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
          s.connect((self.host, self.port))
          s.sendall(header)
          s.sendall(json_data.encode())
          data = self.receive_message(s)
  
      else:
          conn.sendall(header)
          conn.sendall(json_data.encode())

      return data

   def json_message(self,direction):
      local_ip = socket.gethostbyname(socket.gethostname())
      data = {
          'sender': local_ip,
          'instruction': direction
      }
  
      self.send_message(data )

def is_open(host, port):
       s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
       try:
          s.connect((host, int(port)))
          s.shutdown(2)
          return True
       except:
          return False

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0 


if __name__ == '__main__':
   server_socket()
