/*
 * This class contains the information of a detection, that is, 
 * the category of the imperfection and the boxes, if any.
 */
package aiva;

/**
 *
 * @author Bullseye
 */

import java.util.ArrayList;

public class SafetyCheckDetection {
    private ArrayList<BoundingBox> bbList;
    private String label;

    public SafetyCheckDetection(String label) {
        this.label = label;
        bbList = new ArrayList<BoundingBox>();
    }
    
    public SafetyCheckDetection() {
        this.label = "";
        bbList = new ArrayList<BoundingBox>();
    }

    public ArrayList<BoundingBox> getBbList() {
        return bbList;
    }

    public void setBbList(ArrayList<BoundingBox> bbList) {
        this.bbList = bbList;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }
    
    public void addBoundingBox(BoundingBox bb) {
        this.bbList.add(bb);
    }
    
}
