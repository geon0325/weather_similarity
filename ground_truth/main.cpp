#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include <ctime>
#include <thread>
#include <tuple>
#include <random>

#include "Image.cpp"
#include "Similarity.cpp"

using namespace std;

vector<string> split(string str, char delimiter)
{
    vector<string> internal;
    stringstream ss(str);
    string temp;
    while (getline(ss, temp, delimiter)){
        internal.push_back(temp);
    }
    return internal;
}

class image_set
{
    public:
        string file_ir02;
        string file_swir;
        string file_ir01;
        string file_wv;
};

double get_image_sim(string file_i, string file_j, int N, int B)
{
    Image image_i(file_i);
    Image image_j(file_j);
 
    if (not ((image_i.width == 1500 and image_i.height == 1300) and (image_j.width == 1500 and image_j.height == 1300)))
        return -1;
    
    Similarity similarity(N, B);
    double sim = similarity.compute_similarity(image_i, image_j);
    
    delete [] image_i.rgb;
    delete [] image_j.rgb;
    
    return sim;
}

double get_video_sim(string video_path_x, string video_path_y, int video_size, int N, int B)
{
    string file_x, file_y;
    
    double total_sim = 0, sim_ir02, sim_swir, sim_ir01, sim_wv;
    
    for (int i = 0; i < video_size; i++){
        file_x = video_path_x + "/image_" + to_string(i+1) + "/ir02.png";
        file_y = video_path_y + "/image_" + to_string(i+1) + "/ir02.png";
        sim_ir02 = get_image_sim(file_x, file_y, N, B);
        
        file_x = video_path_x + "/image_" + to_string(i+1) + "/swir.png";
        file_y = video_path_y + "/image_" + to_string(i+1) + "/swir.png";
        sim_swir = get_image_sim(file_x, file_y, N, B);
        
        file_x = video_path_x + "/image_" + to_string(i+1) + "/ir01.png";
        file_y = video_path_y + "/image_" + to_string(i+1) + "/ir01.png";
        sim_ir01 = get_image_sim(file_x, file_y, N, B);
        
        file_x = video_path_x + "/image_" + to_string(i+1) + "/wv.png";
        file_y = video_path_y + "/image_" + to_string(i+1) + "/wv.png";
        sim_wv = get_image_sim(file_x, file_y, N, B);
        
        if ((sim_ir02 < 0) or (sim_swir < 0) or (sim_ir01 < 0) or (sim_wv < 0))
            return -1;
        
        total_sim += (sim_ir02 + sim_swir + sim_ir01 + sim_wv);
    }
    total_sim /= (video_size * N * N * 4);
    return total_sim;
}

int main(int argc, char **argv)
{   
    clock_t start;
    string line;
    
    string video_path_x = argv[1];
    string video_path_y = argv[2];
    int N = stoi(argv[3]);
    int B = stoi(argv[4]);
    
    cout << video_path_x << "\t" << video_path_y << endl;
    cout << "N:\t" << N << "\t\tB:\t" << B << endl;
    
    // Compute Video Similarities
    start = clock();
    int video_size = 20;
    
    double sim = get_video_sim(video_path_x, video_path_y, video_size, N, B);
    if (sim < 0) cout << "Similarity computation failed." << endl;
    
    cout << "Similarity:\t" << sim << endl;
    
    cout << "Runtime:\t" << (double)(clock() - start) / CLOCKS_PER_SEC << endl;
    
    return 0;
}


