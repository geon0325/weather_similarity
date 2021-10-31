#include <iostream>
#include <cstdio>
#include <vector>
#include <unordered_map>
#include <cmath>

using namespace std;

class Similarity{
    public:
        Similarity(int N, int B);
        double compute_similarity(Image image_i, Image image_j);
        double compute_similarity_portion(Image image_i, Image image_j, int rx, int ry, int cx, int cy);
        double d_statistics(vector<int>& dist_i, vector<int>& dist_j);
    private:
        int N, B;
};

Similarity::Similarity(int N, int B)
{
    this->N = N;
    this->B = B;
}

double Similarity::compute_similarity(Image image_i, Image image_j)
{
    int h = ceil((double)image_i.height / this->N);
    int w = ceil((double)image_j.width / this->N);
    
    int th = h * this->N;
    int tw = w * this->N;
    
    double total_sim = 0;
    
    for (int ridx = 0; ridx < this->N; ridx ++){
        for (int cidx = 0; cidx < this->N; cidx ++){
            int rx = h * ridx, ry = h * (ridx + 1);
            int cx = w * cidx, cy = w * (cidx + 1);
            //cout << rx << "\t\t\t\t" << cx << endl;
            total_sim += this->compute_similarity_portion(image_i, image_j, rx, ry, cx, cy);
        }
    }

    return total_sim;
}

double Similarity::compute_similarity_portion(Image image_i, Image image_j, int rx, int ry, int cx, int cy)
{
    vector<int> dist_i, dist_j;
    
    for (int i = rx; i < ry; i++){
        for (int j = cx; j < cy; j++){
            float v_i = image_i.rgb[i * image_i.width + j];
            float v_j = image_j.rgb[i * image_j.width + j];

            dist_i.push_back((int)(v_i / 255 * this->B));
            dist_j.push_back((int)(v_j / 255 * this->B));
        }
    }

    double d_stat = this->d_statistics(dist_i, dist_j);
    return 1.0 - d_stat;
}

double Similarity::d_statistics(vector<int>& dist_a, vector<int>& dist_b)
{
    int len_a = (int)dist_a.size(), len_b = (int)dist_b.size(), i = 0, j = 0;
    double cum_a = 0, cum_b = 0, max_D = -1;
    
    sort(dist_a.begin(), dist_a.end());
    sort(dist_b.begin(), dist_b.end());
    
    vector< pair<int, int> > a_dic, b_dic;
    while (i < len_a){
        a_dic.push_back({dist_a[i], 1});
        while (i < len_a - 1 && dist_a[i] == dist_a[i+1]){
            a_dic[a_dic.size()-1].second++;
            i++;
        }
        i++;
    }
    while (j < len_b){
        b_dic.push_back({dist_b[j], 1});
        while (j < len_b - 1 && dist_b[j] == dist_b[j+1]){
            b_dic[b_dic.size()-1].second++;
            j++;
        }
        j++;
    }
    
    int sum_a = (int)a_dic.size(), sum_b = (int)b_dic.size();
    i = 0; j = 0;
    while (i < sum_a || j < sum_b){
        if (j == sum_b){
            cum_a += (double)a_dic[i++].second;
        } else if (i == sum_a){
            cum_b += (double)b_dic[j++].second;
        } else if (a_dic[i].first < b_dic[j].first){
            cum_a += (double)a_dic[i++].second;
        } else if (a_dic[i].first > b_dic[j].first){
            cum_b += (double)b_dic[j++].second;
        } else{
            cum_a += (double)a_dic[i++].second;
            cum_b += (double)b_dic[j++].second;
        }
        double D = abs((cum_a/len_a) - (cum_b/len_b));
        max_D = max(max_D, D);
    }
    
    return max_D;
}