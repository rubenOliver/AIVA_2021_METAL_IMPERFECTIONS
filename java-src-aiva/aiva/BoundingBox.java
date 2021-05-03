/*
 * This class represents a Bounding Box and stores the coordinates of the 
 * corners of the box. 
 */
package aiva;

/**
 *
 * @author Bullseye
 */
public class BoundingBox {
    private int x1;
    private int y1;
    private int x2;
    private int y2;
    
    public BoundingBox(int x, int y, int width, int height) {
        this.x1 = x;
        this.y1 = y;
        this.x2 = width;
        this.y2 = height;
    }
    
    public BoundingBox() {
        this(0,0,0,0);
    }

    public int getX1() {
        return x1;
    }

    public void setX1(int x1) {
        this.x1 = x1;
    }

    public int getY1() {
        return y1;
    }

    public void setY1(int y1) {
        this.y1 = y1;
    }

    public int getX2() {
        return x2;
    }

    public void setX2(int x2) {
        this.x2 = x2;
    }

    public int getY2() {
        return y2;
    }

    public void setY2(int y2) {
        this.y2 = y2;
    }
    
    public String toString() {
        return ""+x1+", "+y1+", "+x2+", "+y2;
    }

    
}
