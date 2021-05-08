/*
 * This class contains the main application. With it, the user can 
 * select the image on which to perform the detection.
 */


/**
 *
 * @author Bullseye
 */

import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.io.File;

public class SafetyCheckDemo extends javax.swing.JFrame {

    /**
     * Creates new form SafetyCheckDemo
     */
    public SafetyCheckDemo() {
        initComponents();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel5 = new javax.swing.JPanel();
        jlImagePath = new javax.swing.JLabel();
        jtfImagePath = new javax.swing.JTextField();
        jbDetect = new javax.swing.JButton();
        jbFileChooser = new javax.swing.JButton();
        jPanel2 = new javax.swing.JPanel();
        jlDetection = new javax.swing.JLabel();
        drawablePanel = new DrawablePanel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("SafetyCheck Demo");
        getContentPane().setLayout(new java.awt.BorderLayout(10, 10));

        jlImagePath.setText("Ruta de la imagen");

        jtfImagePath.setToolTipText("");
        jtfImagePath.setMinimumSize(new java.awt.Dimension(100, 19));
        jtfImagePath.setPreferredSize(new java.awt.Dimension(400, 19));
        jtfImagePath.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jtfImagePathActionPerformed(evt);
            }
        });

        jbDetect.setText("Detectar");
        jbDetect.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jbDetectActionPerformed(evt);
            }
        });

        jbFileChooser.setText("Buscar");
        jbFileChooser.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jbFileChooserActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout jPanel5Layout = new javax.swing.GroupLayout(jPanel5);
        jPanel5.setLayout(jPanel5Layout);
        jPanel5Layout.setHorizontalGroup(
            jPanel5Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel5Layout.createSequentialGroup()
                .addGroup(jPanel5Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel5Layout.createSequentialGroup()
                        .addContainerGap()
                        .addComponent(jlImagePath)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jtfImagePath, javax.swing.GroupLayout.PREFERRED_SIZE, 247, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jbFileChooser))
                    .addGroup(jPanel5Layout.createSequentialGroup()
                        .addGap(183, 183, 183)
                        .addComponent(jbDetect, javax.swing.GroupLayout.PREFERRED_SIZE, 124, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addContainerGap(17, Short.MAX_VALUE))
        );
        jPanel5Layout.setVerticalGroup(
            jPanel5Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel5Layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(jPanel5Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jlImagePath, javax.swing.GroupLayout.PREFERRED_SIZE, 25, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jtfImagePath, javax.swing.GroupLayout.PREFERRED_SIZE, 25, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jbFileChooser))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addComponent(jbDetect))
        );

        getContentPane().add(jPanel5, java.awt.BorderLayout.NORTH);

        jPanel2.setBorder(javax.swing.BorderFactory.createEtchedBorder());

        jlDetection.setText("Esperando detección");

        javax.swing.GroupLayout drawablePanelLayout = new javax.swing.GroupLayout(drawablePanel);
        drawablePanel.setLayout(drawablePanelLayout);
        drawablePanelLayout.setHorizontalGroup(
            drawablePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 300, Short.MAX_VALUE)
        );
        drawablePanelLayout.setVerticalGroup(
            drawablePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 250, Short.MAX_VALUE)
        );

        javax.swing.GroupLayout jPanel2Layout = new javax.swing.GroupLayout(jPanel2);
        jPanel2.setLayout(jPanel2Layout);
        jPanel2Layout.setHorizontalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addGap(136, 136, 136)
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(drawablePanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jlDetection, javax.swing.GroupLayout.PREFERRED_SIZE, 204, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap(64, Short.MAX_VALUE))
        );
        jPanel2Layout.setVerticalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addComponent(jlDetection, javax.swing.GroupLayout.PREFERRED_SIZE, 19, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(drawablePanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(42, Short.MAX_VALUE))
        );

        getContentPane().add(jPanel2, java.awt.BorderLayout.CENTER);

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jbDetectActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jbDetectActionPerformed
        detect();
    }//GEN-LAST:event_jbDetectActionPerformed

    private void jtfImagePathActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jtfImagePathActionPerformed
        detect();
    }//GEN-LAST:event_jtfImagePathActionPerformed

    private void jbFileChooserActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jbFileChooserActionPerformed
        javax.swing.JFileChooser jfc=new javax.swing.JFileChooser(new File("./NEU-DET/IMAGES/"));
        jfc.showOpenDialog(this);
        File file=jfc.getSelectedFile();
        if(file!=null) {
            jtfImagePath.setText(file.getAbsolutePath());
            detect();
        }
    }//GEN-LAST:event_jbFileChooserActionPerformed
    
    
    private void detect() {
        BufferedImage bImage = null;
        aiva.SafetyCheckDetector scDetector;
        aiva.SafetyCheckDetection scDetection = null;
        
        jlDetection.setText("Esperando detección");
        drawablePanel.setScDetection(null);
        drawablePanel.setImage(null);
        drawablePanel.repaint();
        
        if(jtfImagePath.getText().trim().equals("")) {
            return;
        }
        
        try {
            bImage = ImageIO.read(new File(jtfImagePath.getText()));
        } catch(Exception ex) {
            System.out.println("Error");
            javax.swing.JOptionPane.showMessageDialog(this,"No se ha podido cargar la imagen");
        }
        if(bImage!=null){
            
            drawablePanel.setImage(bImage);
            drawablePanel.repaint();
            try{
                scDetector = new aiva.SafetyCheckDetector("localhost",9000);
                scDetection = scDetector.detect(bImage);
            } catch(java.io.IOException ex){
                javax.swing.JOptionPane.showMessageDialog(this,"Se ha producido un error en la detección");
            }
        }
        
        if(scDetection!=null){
            jlDetection.setText(scDetection.getLabel());
            drawablePanel.setScDetection(scDetection);
            drawablePanel.repaint();
        }
    }
    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(SafetyCheckDemo.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(SafetyCheckDemo.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(SafetyCheckDemo.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(SafetyCheckDemo.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new SafetyCheckDemo().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private DrawablePanel drawablePanel;
    private javax.swing.JPanel jPanel2;
    private javax.swing.JPanel jPanel5;
    private javax.swing.JButton jbDetect;
    private javax.swing.JButton jbFileChooser;
    private javax.swing.JLabel jlDetection;
    private javax.swing.JLabel jlImagePath;
    private javax.swing.JTextField jtfImagePath;
    // End of variables declaration//GEN-END:variables
}
