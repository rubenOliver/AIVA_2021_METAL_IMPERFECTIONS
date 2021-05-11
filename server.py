# This code has been carried out for the Applications subject of the
# Master's Degree in Computer Vision at the Rey Juan Carlos University
# of Madrid.
# Date: April 2021
# Authors: Rubén Oliver, Ismael Linares and Juan Luis Carrillo

from http.server import BaseHTTPRequestHandler
import socketserver
import base64
import json
import cv2
import cgi
import numpy as np
from SafetyCheck import RecognizerMetalImperfections


class Servidor(BaseHTTPRequestHandler):
    '''
    This class acts like a HTTP server, receiving images from client and sending detection to client
    '''

    def __init__(self, *args, **kwargs):
        '''
        Inits the server
        :param args: Optional args
        :param kwargs: Other optional args
        '''
        self.recognizer = RecognizerMetalImperfections()
        BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

    def _set_headers(self):
        '''
        This method writes the header of the response
        :return:
        '''
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_POST(self):
        '''
        This method processes POST requests from the client
        :return:
        '''

        message = self._extract_msg()

        # First, extract the Base64 formatted image from JSON
        base64_img = message['img']
        jpg_img = base64.b64decode(base64_img)
        img = cv2.imdecode(np.frombuffer(jpg_img,dtype=np.int8),1)

        # Then, call the recognize method of the AIVA library
        label, bounding_boxes = self.recognizer.recognize(img)

        # Write the result to messeage
        message_out = {}
        message_out['label'] = label
        message_out['bounding_boxes'] = bounding_boxes
        bytes_message = bytes(json.dumps(message_out),encoding='UTF-8')

        # Finally, send back the message with result to the client
        self._set_headers()
        self.wfile.write(bytes_message)

    def _extract_msg(self):
        '''
        Extract content-type field from header of message
        :return: Message without header
        '''

        header = self.headers.get('content-type')
        ctype, pdict = cgi.parse_header(header)

        # Check the JSON content
        if ctype != 'application/json':
            self.send_response(400)
            self.end_headers()
            raise Exception()

        # Read the JSON content and introduce it on message
        length = int(self.headers.get('content-length'))
        message = json.loads(self.rfile.read(length))
        return message

# Start server
server_address = ('', 9000)
httpd = socketserver.TCPServer(server_address, Servidor)
print('Starting httpd on port 9000')
httpd.serve_forever()
