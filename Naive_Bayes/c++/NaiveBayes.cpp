#ifndef NAIVEBAYES_H
#define NAIVEBAYES_H

typedef struct class_summary 
{  
    std::vector<std::vector<float>> Mean_Stdev;
    float class_prob;

} class_summary; 

class Naive_Bayes
{
    private:
    std::vector<class_summary> Summary;
    std::vector<float> unique_label;
    
    public:
    void fit(std::vector<std::vector<float>> dataset);
    int  predict(const  std::vector<float>& test_data);
};

class_summary calculate_Class_Summary(std::vector<std::vector<float>> dataset, float class_label);
float prob_By_Summary(const std::vector<float> &test_data ,const class_summary &summary );
#endif
template <typename T> 
T Mean (std::vector<T> Data)
{
    T mean = std::accumulate(std::begin(Data), std::end(Data), 0.0) / Data.size();
    return mean;
}
template <typename M> 
double StDev (std::vector<M> &Data)
{
    M mean = std::accumulate(std::begin(Data), std::end(Data), 0.0) / Data.size();
    double sq_sum = std::inner_product(Data.begin(), Data.end(), Data.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / Data.size() - (double)(mean * mean));
    return stdev;
}
double calc_prob(double value, double mean, double stdev)
{
    static const double inv_sqrt_2pi = 0.3989422804014327;
    double a = (value - mean) / stdev;
    return inv_sqrt_2pi / stdev * std::exp(-0.5f * a * a);
}
class_summary calculate_Class_Summary (std::vector<std::vector<float>> dataset, float class_label )
{
    auto class_data = split_by_class(dataset,class_label);
    class_summary summary;
    std::vector<float> temp;
    for (auto row = class_data.begin(); row != class_data.end()-1; row++)
    {
        temp.clear();
        temp.push_back(alg_math::Math_Mean(*row));
        temp.push_back(alg_math::Math_Var(*row));
        summary.Mean_Stdev.push_back(temp);
    }
    summary.class_prob = float(class_data[0].size())/ dataset[0].size();
    return summary;
}
typedef struct class_summary 
{  
    std::vector<std::vector<float>> Mean_Stdev;
    float class_prob;

} class_summary; 
void Naive_Bayes::fit(std::vector<std::vector<float>> training_data )
{
    unique_label = training_data[training_data.size()-1];
    std::sort(unique_label.begin(),unique_label.end());
    unique_label.erase(std::unique(unique_label.begin(),unique_label.end()),unique_label.end());
    for (auto row = unique_label.begin(); row != unique_label.end(); row++)
    {
       Summary.push_back( calculate_Class_Summary (training_data, *row));
    }
}
int Naive_Bayes::predict(const  std::vector<float>& test_data)
{
    std::vector<float> out;
    for (auto row = unique_label.begin(); row != unique_label.end(); row++)
    {
       out.push_back( prob_By_Summary(test_data ,Summary[*row] ));
    }
    int maxElementIndex = std::max_element(out.begin(),out.end()) - out.begin();
    return maxElementIndex;
}

float prob_By_Summary(const std::vector<float> &test_data ,const class_summary &summary )
{
    int index =0;
    float prob = 1;
    for (auto row = summary.Mean_Stdev.begin(); row != summary.Mean_Stdev.end()-1; row++)
    {
        prob *= calc_prob(test_data[index],(*row)[0],(*row)[1]);
        index++;
    }
    /* multiplying by the class probability*/
    prob *= summary.class_prob;
    return prob;
}

float Vect_match_Score (const std::vector<float> &labels, const std::vector<float> &predictions )
{
    float result=0; 
    int index=0;
    std::for_each(labels.begin(), labels.end(), [&](float label)
    { 
        if(label == predictions[index])
        {
            result++;
        }
        index++; 
    });
    return float(100*result/labels.size());
}

//Combing all together 

#include <iostream>
#include <vector>
#include "Math.h"
#include "preprocessing.h"
#include "iris.h"
#include <ctime>
#include "Naive_bayes.h"

int main(){
    std::vector<std::vector<float>> dataset =  Read_Iris_Dataset();
    
    Naive_Bayes naive = Naive_Bayes();
    
    std::srand (unsigned(std::time(0)));
    dataset = vect_Transpose(dataset); 
    std::random_shuffle (dataset.begin(),dataset.end());
    dataset = vect_Transpose(dataset); 
    float percentage =70;
    std::vector<std::vector<float>> training_data  = vector_Train_Split(dataset , percentage);
    std::vector<std::vector<float>> testing_data  = vector_Test_Split(dataset , 100 - percentage);
    
    std::cout<< "IRIS testing Data Size is ( " << testing_data.size() << " , "  <<testing_data[0].size()<<" )" <<std::endl;
    
    std::cout<< "IRIS training Data Size is ( " << training_data.size() << " , "  <<training_data[0].size()<<" )" <<std::endl;
    naive.fit(training_data);
    
    std::vector<float> predicitions;
    testing_data = vect_Transpose(testing_data); 
    for(int i=0; i<testing_data.size(); i++ )
    {
        auto index = naive.predict(testing_data[i]);
        predicitions.push_back(index);
    }
    testing_data = vect_Transpose(testing_data); 
    std::cout<<"score is :" << Vect_match_Score (testing_data[4],predicitions)<<std::endl;
    return 0;
}