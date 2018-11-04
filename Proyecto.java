
import static javafx.scene.input.KeyCode.I;
import static javax.swing.text.html.HTML.Tag.I;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import static org.opencv.core.CvType.CV_16S;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author sala306
 */
public class Proyecto {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        Mat img = Imgcodecs.imread("src\\imagenes\\imgProg.png");
        Mat img2=Imgcodecs.imread("src\\imagenes\\imgProg.png");
        Mat imgGaus = new Mat();
        Mat imgGris = new Mat();
        Mat imgGrisSobel = new Mat();// es el gradiente de la gris osea la derivada
        Mat imgGrisSobelConElabsX = new Mat();
        Mat imgGrisSobelY = new Mat();// es el gradiente de la gris osea la derivada
        Mat imgGrisSobelConElabsY = new Mat();
        Mat unionGrad = new Mat();
        int escala = 1;
        int delta = 0;
        int ddepth = CV_16S;
        Mat kernel = new Mat(3, 3, CvType.CV_32F, Scalar.all(1f));
        Imgproc.medianBlur(img, img, 5);
        /// Convertirlo a gris 
        Imgproc.cvtColor(img, imgGris, Imgproc.COLOR_BGR2GRAY);

        Imgproc.Sobel(imgGris, imgGrisSobelY, ddepth, 1, 0, 3, escala, delta, 0);
        Core.convertScaleAbs(imgGrisSobelY, imgGrisSobelConElabsY);
        Imgproc.Sobel(imgGris, imgGrisSobel, ddepth, 0, 1, 3, escala, delta, 0);
        Core.convertScaleAbs(imgGrisSobel, imgGrisSobelConElabsX);
        Core.addWeighted(imgGrisSobelConElabsX, 0.5, imgGrisSobelConElabsY, 0.5, 0, unionGrad);
      
        //---------------------------------------------xxxxxxxxxxxxxxxxxxxxxxxse trabaja solo con el mat de uingradiente
        // openin sirve para quitar ruido blanco en zonas negras es un erotion despues de un dilation
        // clousin quita ruido negro en zonas blancas el valor entero de morphology debe ser quien decide si hace openin o clousin
        Mat gray = new Mat();
        gray=unionGrad.clone();
        Mat gray2 = new Mat();
        gray2=unionGrad.clone();
        Imgproc.morphologyEx(gray, gray,2, kernel);// con 0 queda bordes del cuadro bien con 1 no tanto y con 2 coge perfecto los circulos
        //Imgproc.medianBlur(img, img, 5);
        //Imgproc.GaussianBlur(img,img, new Size (5,5), 0);
        //Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);
        //Imgproc.medianBlur(gray, gray, 5);// solo dejando el median aca coge el del medio del dado de lado
        Imgproc.GaussianBlur(gray, gray, new Size (5,5), 0);//añadiendo el gaus los cuadrados quedan mejor

        
        
        
        
        
        Mat circles = new Mat();
        Imgproc.Canny(gray,gray, 200, 3);
        Imgproc.HoughCircles(gray, circles, Imgproc.HOUGH_GRADIENT, 1.0,
                (double)gray.rows()/16, // change this value to detect circles with different distances to each other
                100.0, 30.0, 1, 15); // change the last two parameters
                // (min_radius & max_radius) to detect larger circles
        int contadorDeCirculos=0;        
        for (int x = 0; x < circles.cols(); x++) {
            double[] c = circles.get(0, x);
            Point center = new Point(Math.round(c[0]), Math.round(c[1]));
            // circle center
            Imgproc.circle(img, center, 1, new Scalar(0,100,100), 3, 8, 0 );
            // circle outline
            contadorDeCirculos++;
            int radius = (int) Math.round(c[2]);
            Imgproc.circle(img, center, radius, new Scalar(255,0,255), 3, 8, 0 );
        }
        //---------------------------------cuadrados
        Mat img3 = Imgcodecs.imread("src\\imagenes\\imgProg.png");

        Imgproc.morphologyEx(gray2, gray2,1, kernel);// con 0 queda bordes del cuadro bien con 1 no tanto y con 2 coge perfecto los circulos
        //Imgproc.medianBlur(img, img, 5);
        //Imgproc.GaussianBlur(img,img, new Size (5,5), 0);
        //Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);
        //Imgproc.medianBlur(gray, gray, 5);// solo dejando el median aca coge el del medio del dado de lado
        Imgproc.GaussianBlur(gray2, gray2, new Size (5,5), 0);//añadiendo el gaus los cuadrados quedan mejor
     
        Mat dst = new Mat();
        dst=gray2.clone();
        Mat cdst = new Mat();
        Mat cdstP = new Mat();
        // Copy edges to the images that will display the results in BGR
        cdst=img.clone();
        cdstP = img.clone();
        // Standard Hough Line Transform
        Mat lines = new Mat(); // will hold the results of the detection
        Imgproc.HoughLines(dst, lines, 1, Math.PI/180, 150); // runs the actual detection
        // Draw the lines
        for (int x = 0; x < lines.rows(); x++) {
            double rho = lines.get(x, 0)[0],
                    theta = lines.get(x, 0)[1];
            double a = Math.cos(theta), b = Math.sin(theta);
            double x0 = a*rho, y0 = b*rho;
            Point pt1 = new Point(Math.round(x0 + 1000*(-b)), Math.round(y0 + 1000*(a)));
            Point pt2 = new Point(Math.round(x0 - 1000*(-b)), Math.round(y0 - 1000*(a)));
            Imgproc.line(cdst, pt1, pt2, new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
        }
        // Probabilistic Line Transform
        Mat linesP = new Mat(); // will hold the results of the detection
        Imgproc.HoughLinesP(dst, linesP, 1, Math.PI/180, 50, 50, 10); // runs the actual detection
        // Draw the lines
        for (int x = 0; x < linesP.rows(); x++) {
            double[] l = linesP.get(x, 0);
            Imgproc.line(cdstP, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
        }
       
        HighGui.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
        HighGui.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
        
        System.out.println("cantidad de circulos:"+contadorDeCirculos);
        HighGui.imshow("detected ",img2);
        HighGui.imshow("test1",img);
        HighGui.imshow("test2", unionGrad);// con este se trabaja mejor
        HighGui.imshow("test3", gray);
     
        HighGui.waitKey();
        System.exit(0);
    }

}
