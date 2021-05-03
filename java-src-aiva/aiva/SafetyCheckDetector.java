/*
 * This class contains the important detection method. This method 
 * returns the results of the detection of an image. The client will 
 * be able to integrate this class into his application to perform 
 * the detection.
 */
package aiva;

/**
 *
 * @author Bullseye
 */


import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.InetAddress;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import javax.imageio.ImageIO;
import javax.sql.rowset.spi.SyncFactory;


public class SafetyCheckDetector {    
    
    public SafetyCheckDetection detect(BufferedImage bImage) throws IOException{
        String hostname = "localhost";
        int port = 9000;    
        Socket socket = null;
        
        SafetyCheckDetection detection = null;
                         
        InetAddress addr = InetAddress.getByName(hostname);

        socket = new Socket(addr, port);
        BufferedWriter wr = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream(), "UTF8"));

        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ImageIO.write(bImage, "jpg", bos );
        byte [] data = bos.toByteArray();
        String imageB64 = Base64.getEncoder().encodeToString(data);
        String json = "{\n  \"img\":\"" + imageB64 + "\"\n}";


        wr.write("POST / HTTP/1.1\r\n");
        wr.write("Host: localhost\r\n");
        wr.write("Content-Length: "+json.length()+"\r\n");
        wr.write("Content-Type: application/json\r\n");
        wr.write("\r\n");
        wr.write(json);
        wr.flush();
        BufferedReader rd = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        String line;
        List<String> lines = new ArrayList<String>();
        while ((line = rd.readLine()) != null) {
            // System.out.println(line);
            lines.add(line);
        }
        // System.out.println(lines.get(lines.size() - 1));
        String keyLine = lines.get(lines.size() - 1);
        String[] stringSplitted = keyLine.split(":");
        // System.out.println("splited" + stringSplitted[2]);
        String labelSplit = stringSplitted[1].split(",")[0];
        String bounding_boxes = stringSplitted[2];
        bounding_boxes = bounding_boxes.replace("[", "");
        bounding_boxes = bounding_boxes.replace("]", "");
        bounding_boxes = bounding_boxes.replace("}", "");
        bounding_boxes = bounding_boxes.replace(" ", "");
        String[] positions = bounding_boxes.split(",");
        labelSplit = labelSplit.replace(" ", "");
        labelSplit = labelSplit.replace("\"", "");

        int counter = 0;
        detection = new SafetyCheckDetection(labelSplit);

        for(int j = 0; j < positions.length - 1; j = j + 4) {
            int x = Integer.parseInt(positions[j]);
            int y = Integer.parseInt(positions[j+1]);
            int w = Integer.parseInt(positions[j+2]);
            int h = Integer.parseInt(positions[j+3]);

            BoundingBox bb = new BoundingBox(x,y,w,h);
            detection.addBoundingBox(bb);
        }

        socket.close();

        return detection;
    }
    
    
    public static void main(String[] args) {
        if(args.length<1){
            System.out.println("Indique la ruta de la imagen. Por ejemplo:");
            System.out.println("./NEU-DET/IMAGES/crazing_1.jpg");
        } else {
            SafetyCheckDetector scDetector = new SafetyCheckDetector();
            try{
                BufferedImage bImage = ImageIO.read(new File(args[0]));
                SafetyCheckDetection scDetection = scDetector.detect(bImage);
                System.out.println("El tipo de imperfección para "+args[0]+" es: "+scDetection.getLabel());
                System.out.println("Bounding Boxes detectados en:");
                ArrayList<BoundingBox> bbList=scDetection.getBbList();
                for(int i=0;i<bbList.size();i++) {
                    BoundingBox bb = bbList.get(i);
                    System.out.println("\t"+bb.toString());
                }
            }catch(Exception ex) {
                ex.printStackTrace();
            }
        }
    }
    
}
