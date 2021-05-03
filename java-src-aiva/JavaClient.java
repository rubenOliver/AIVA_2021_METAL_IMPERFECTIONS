/*
 * This class is used to detect the images contained in a folder. 
 * Results are displayed by text
 */

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

public class JavaClient{
    public static void main(String[] args) {
        
        BufferedImage bImage;
        
        try {
            String hostname = "localhost";
            int port = 9000;                    
            InetAddress addr = InetAddress.getByName(hostname);
            Socket socket = null;// = new Socket(addr, port);
            String path = "/";
            // Send headers
            
            File images_path = new File("./NEU-DET/IMAGES/");
            String[] paths = images_path.list();
            for(int i = 0; i <= paths.length - 1; i++) {
                socket = new Socket(addr, port);
                BufferedWriter wr = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream(), "UTF8"));
                System.out.println("********** " + paths[i]+ " **************");

                bImage = ImageIO.read(new File("./NEU-DET/IMAGES/" + paths[i]));
                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                ImageIO.write(bImage, "jpg", bos );
                byte [] data = bos.toByteArray();
                String imageB64 = Base64.getEncoder().encodeToString(data);
                String json = "{\n  \"img\":\"" + imageB64 + "\"\n}";
                
                
                wr.write("POST "+path+" HTTP/1.1\r\n");
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
                System.out.println("label : " + labelSplit);
                for(int j = 0; j < positions.length - 1; j = j + 4) {
                    System.out.print("Bounding box number: " + ++counter);
                    System.out.println(positions[j] + "," + positions[j+1] + "," + positions[j+2] + "," + positions[j+3]);
                }
            }
            socket.close();
        } catch (IOException e1) {
            e1.printStackTrace();
        }
        
        
    }
}
