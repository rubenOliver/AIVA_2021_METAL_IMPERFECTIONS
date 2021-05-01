from http.server import BaseHTTPRequestHandler
import socketserver
import base64
import json
import cv2
import cgi
import numpy as np
from SafetyCheck import RecognizerMetalImperfections
class Servidor(BaseHTTPRequestHandler):
 # Clase que actua como servidor HTTP
    def __init__(self, *args, **kwargs):
        self.recognizer = RecognizerMetalImperfections()
        BaseHTTPRequestHandler.__init__(self, *args, **kwargs)
    def _set_headers(self):
        # Método privado que escribe la cabecera de las respuestas
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_POST(self):
        # POST Nos envían datos, hacemos la VA y respondemos
        message = self._extract_msg()
        #Primero extraemos la imagen del JSON que debe venir en Base64
        base64_img = message['img']
        jpg_img = base64.b64decode(base64_img)
        img = cv2.imdecode(np.frombuffer(jpg_img,dtype=np.int8),1)
        # Luego llamamos a nuestra fachada para que aplique la VA
        label, bounding_boxes = self.recognizer.recognize(img)
        #Finalmente escribimos el resultado en un mapa para enviarlo
        message_out = {}
        message_out['label'] = label
        message_out['bounding_boxes'] = bounding_boxes
        bytes_message = bytes(json.dumps(message_out),encoding='UTF-8')
        # cv2.showImage('image', img)
        # cv2.waitKey(0)
        # Finalmente enviamos la respuesta
        self._set_headers()
        self.wfile.write(bytes_message)

    def _extract_msg(self):
        #Extraemos el campo content-type de la cabecera que envían
        header = self.headers.get('content-type')
        ctype, pdict = cgi.parse_header(header)

        # Comprobamos que nos envían un JSON
        if ctype != 'application/json':
            self.send_response(400)
            self.end_headers()
            raise Exception()

        # Leemos el JSON y lo metemos en un mapa
        length = int(self.headers.get('content-length'))
        message = json.loads(self.rfile.read(length))
        return message

#Poner en marcha el servidor
server_address = ('', 9000)
httpd = socketserver.TCPServer(server_address, Servidor)
print('Starting httpd on port 9000')
httpd.serve_forever()