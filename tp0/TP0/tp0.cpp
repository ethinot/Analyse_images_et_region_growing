#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <cstdio>
#include <cmath> 

#include <fstream>
#include <iostream>

#include <string>

// Fonction pour l'affichage des histogrammes
void gnuPlot(const cv::Mat& hist, const std::string& fileName, const int histSize) {
    std::ofstream dataFile("./ressources/" + fileName + ".txt");
    for (int i = 0; i < histSize; i++) {
        dataFile << i << " " << hist.at<float>(i) << std::endl;
    }
    dataFile.close();

    FILE* gnuplotPipe = popen("gnuplot -persistent", "w");
    if (gnuplotPipe) {
        fprintf(gnuplotPipe, "set title 'Histogramme de %s'\n", fileName.c_str());
        fprintf(gnuplotPipe, "plot './ressources/%s.txt' with boxes\n", fileName.c_str());
        fflush(gnuplotPipe);
        getchar(); 
        pclose(gnuplotPipe);
    } else {
        std::cerr << "Erreur lors de l'ouverture de Gnuplot." << std::endl;
    }
}

// Fonction qui calcule et renvoie l'histogramme cummulé de celui passé en paramètre
cv::Mat calcHistCumul(const cv::Mat& src, int histSize) {
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
    dst.at<float>(0) = src.at<float>(0);

    for (int i=1; i<histSize; ++i) {
        dst.at<float>(i) = dst.at<float>(i-1) + src.at<float>(i); 
    }
    return dst;
}

// Fonction réalise l'étirement d'une images (en niveau de gris)
cv::Mat etirement(const cv::Mat& image, int Nmin, int Nmax) 
{
    int height = image.rows;
    int width = image.cols;
    cv::Mat image2(height, width, CV_8UC1);
    for (int r=0; r<height; ++r) {
        for (int c=0; c<width; ++c) {
            image2.at<uchar>(r, c) = cv::saturate_cast<uchar>(255 * ((image.at<uchar>(r, c) - Nmin) / static_cast<double>(Nmax - Nmin)));
        }
    }

    return image2;
}

// Fonction réalise l'égalisation d'une images (en niveau de gris)
cv::Mat egalisation(const cv::Mat & inputImage, const cv::Mat& inputHist, int histSize) {
    cv::Mat histoCumul = calcHistCumul(inputHist, histSize);

    histoCumul /= inputImage.total();

    cv::Mat outputImage = inputImage.clone();
   
    for (int i = 0; i < outputImage.rows; ++i) {
        for (int j = 0; j < outputImage.cols; ++j) {
            outputImage.at<uchar>(i, j) = cv::saturate_cast<uchar>(255 * histoCumul.at<float>(inputImage.at<uchar>(i, j)));
        }
    }
    return outputImage;
}

// Fonction réalise le produit de convolution
float convolution(const cv::Mat& image, const cv::Mat& h, int x, int y) 
{
    float sum = 0.0;
    for (int u = -1; u <= 1; ++u) {
        for (int v = -1; v <= 1; ++v) {
            sum += h.at<float>(1 + u, 1 + v) * image.at<uchar>(y + u, x + v);
        }   
    }
    return sum;
}

// Fonction qui applique une matrice de filtrage sur une image via l'utilisation du produit de convolution
void filtrage(const cv::Mat& src, cv::Mat& dst, const cv::Mat& h) 
{
    assert(h.rows == 3 && h.cols == 3);
    int height = src.rows;
    int width = src.cols;
    dst = cv::Mat::zeros(src.size(), src.type());

    for (int r = 1; r < height - 1; ++r) {
        for (int c = 1; c < width - 1; ++c) {
            dst.at<uchar>(r, c) = cv::saturate_cast<uchar>(convolution(src, h, c, r));
        }
    }
}

int main(int argc, char** argv) 
{ 
    if (argc != 3) { 
        printf("usage: DisplayImage.out <Image_Path>\n"); 
        return -1; 
    } 

    cv::Mat image1, image2, copie1Image2, copie2Image2; 
    image1 = cv::imread(argv[1], 0); 
    image2 = cv::imread(argv[2], 0); 

    if (!image1.data && !image2.data) { 
        printf("No image data \n"); 
        return -1; 
    } 

    copie1Image2 = image2.clone();
    copie2Image2 = image2.clone();

    int histSize = 256; // de 0 à 255
    float range[] = { 0, 255 };
    const float* histRange[] = { range };

    int histWidth = 512;
    int histHeight = 400;

    // Partie 1 - Calcule d'histogramme (Lena)
    //---------------------------------------//

    cv::Mat histLena; 
    cv::Mat histLenaCumule;
    // Histogramme de l'image sans transformation 
    // pour le normaliser il suffit de divisé pour le nombre de pixel de l'image (image.total())
    calcHist(&image1, 1, 0, cv::Mat(), histLena, 1, &histSize, histRange);
    // Histogramme cumulé
    histLenaCumule = calcHistCumul(histLena, histSize);

    // Affichages (
    // pour l'affichage l'image suivante il faut le fermer pour appuyer sur n'import quelle touche dans le console
    gnuPlot(histLena, "Histogramme de Lena", histSize);
    gnuPlot(histLenaCumule, "Histogramme cumulé de Lena", histSize);

    // Partie 2 - Etirement (Cameraman surexposé)
    //-----------------------------------------//

    cv::Mat histCameraman; 
    cv::Mat histCameramanEtire;   

    // Images et histogramme sans transformation 
    calcHist(&image2, 1, 0, cv::Mat(), histCameraman, 1, &histSize, histRange);
    cv::imshow("Cameraman surexp avant étitement", image2);

    // Etirement de l'image 
    // éstimation vissuelle du minimum / maximum des valeurs du niveau de gris
    image2 = etirement(image2, 125, 253);
    
    gnuPlot(histCameraman, "Histogramme du cameraman surexposé", histSize);
    // Images et histogramme avec étirement
    cv::imshow("Cameraman surexp après étitement", image2);
    calcHist(&image2, 1, 0, cv::Mat(), histCameramanEtire, 1, &histSize, histRange);
    gnuPlot(histCameramanEtire, "Histogramme du cameraman étiré", histSize);

    // Partie 3 - Egalisation (Cameraman surexposé)
    //-------------------------------------------//

    cv::Mat histCameramanEgalise;   

    // Egalisation de l'image 
    copie1Image2 = egalisation(copie1Image2, histCameraman, histSize);

    // Images et histogramme avec égalisation
    cv::imshow("Cameraman surexp après égalisation", copie1Image2);
    calcHist(&copie1Image2, 1, 0, cv::Mat(), histCameramanEgalise, 1, &histSize, histRange);
    gnuPlot(histCameramanEgalise, "Histogramme du cameraman égalisé", histSize);

    // Partie 4 - Produit de convolution et filtre
    //-------------------------------------------//

    cv::Mat imageFiltree1;
    cv::Mat imageFiltree2;

    float s = 1.0/16.0;
    float filter1[9] = { s, s*2, s,
                        s*2, s*4, s*2,
                        s, s*2, s };

    s = 1.0/9.0;
    float filter2[9] = { s, s, s,
                        s, s, s,
                        s, s, s };
    
    float filter3[9] = { -1, -1, -1,
                        -1,  8, -1,
                        -1, -1, -1};

    cv::Mat h1(3, 3, CV_32F, filter1);  // Utilisez CV_32F pour les données float
    cv::Mat h2(3, 3, CV_32F, filter3);  // Utilisez CV_32F pour les données float

    filtrage(copie2Image2, imageFiltree1, h1);
    filtrage(copie2Image2, imageFiltree2, h2);
    

    cv::imshow("Cameraman après filtrage 1 (floue)", imageFiltree1);
    cv::imshow("Cameraman après filtrage 2 (contour)", imageFiltree2);

    cv::waitKey(0); 
    return 0; 
}