/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

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

/**
 *
 * @author wilme
 */
public class deteccionLineas {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        Mat img = Imgcodecs.imread("src\\imagenes\\imgProg.png");
        Mat img2 = Imgcodecs.imread("src\\imagenes\\imgProg.png");

// Declare the output variables
        Mat dst = new Mat();
        Mat src = new Mat();
        Mat cdst = new Mat();
        Mat cdstP = new Mat();
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
        Imgproc.cvtColor(img, src, Imgproc.COLOR_BGR2GRAY);
        //Imgproc.morphologyEx(unionGrad,unionGrad,1, kernel);
        //Imgproc.GaussianBlur(unionGrad, unionGrad, new Size (5,5), 0);
        //Imgproc.Canny(unionGrad,unionGrad, 200, 3);
        //Imgproc.GaussianBlur(unionGrad, unionGrad, new Size (5,5), 0);
        Imgproc.morphologyEx(unionGrad, unionGrad, 0, kernel);
        Imgproc.GaussianBlur(unionGrad, unionGrad, new Size(5, 5), 0);
        Imgproc.medianBlur(unionGrad, unionGrad, 5);
        Imgproc.Canny(unionGrad, unionGrad, 200, 3);
        Imgproc.morphologyEx(unionGrad, unionGrad, 1, kernel);
        Imgproc.morphologyEx(unionGrad, unionGrad, 1, kernel);
        //Imgproc.Canny(unionGrad,unionGrad, 200, 3);
        int kernelSize = 3;
        int elementType = Imgproc.CV_SHAPE_RECT;
        Mat element = Imgproc.getStructuringElement(elementType, new Size(2 * kernelSize + 1, 2 * kernelSize + 1),
                new Point(kernelSize, kernelSize));

        Imgproc.erode(unionGrad, unionGrad, element);

        //Imgproc.dilate(unionGrad, unionGrad, element);

        dst = unionGrad.clone();

        // Copy edges to the images that will display the results in BGR
        Imgproc.cvtColor(unionGrad, cdst, Imgproc.COLOR_GRAY2BGR);
        cdstP = cdst.clone();
        // Standard Hough Line Transform
        Mat lines = new Mat(); // will hold the results of the detection
        Imgproc.HoughLines(dst, lines, 1, Math.PI / 180, 150); // runs the actual detection
        // Draw the lines
        for (int x = 0; x < lines.rows(); x++) {
            double rho = lines.get(x, 0)[0],
                    theta = lines.get(x, 0)[1];
            double a = Math.cos(theta), b = Math.sin(theta);
            double x0 = a * rho, y0 = b * rho;
            Point pt1 = new Point(Math.round(x0 + 1000 * (-b)), Math.round(y0 + 1000 * (a)));
            Point pt2 = new Point(Math.round(x0 - 1000 * (-b)), Math.round(y0 - 1000 * (a)));
            Imgproc.line(cdst, pt1, pt2, new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
        }
        // Probabilistic Line Transform
        Mat linesP = new Mat(); // will hold the results of the detection
        Imgproc.HoughLinesP(dst, linesP, 1, Math.PI / 180, 90, 70, 6); // runs the actual detection
        // Draw the lines
        int contadorLineas=0;
        for (int x = 0; x < linesP.rows(); x++) {
            double[] l = linesP.get(x, 0);
            System.out.println("datos:"+l[0]+","+l[1]+"-"+l[2]+","+l[3]);// buscar cualquiera q este tres pixeles arriba del otro y borrar uno de los dos si funciona
            Imgproc.line(cdstP, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(0, 0, 255), 1, Imgproc.LINE_AA, 0);
            contadorLineas++;
        }
        System.out.println("cantidad de lineas:"+contadorLineas);
        // Show results
        HighGui.imshow("test2-unionGrad", unionGrad);
        // HighGui.imshow("Source", src);

        HighGui.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
        HighGui.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
        // Wait and Exit
        HighGui.waitKey();
        System.exit(0);
    }

}
