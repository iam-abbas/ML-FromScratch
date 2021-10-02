#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#define DATA_SIZE 700
#define EPOCHS 3000
#define LEARNING_RATE 0.0005
#define DELAY 5 //in milliseconds

double theta[] = {1.0f, 1.0f}; //Fitting parameters
int total_samples = DATA_SIZE;

double x[DATA_SIZE], y[DATA_SIZE], err[DATA_SIZE];
double n = 700.0f, sumerr0, sumerr1, sumsqerr; //Useful variables
double alpha = LEARNING_RATE;

void grad_descent();
void cost();
double costfunction();
double hypothesis(double);

int main()
{

    int epoch = 0, total_epochs = EPOCHS;
    double cost;
    int count = 0;

    //GNUPlot initialisation stuff
    FILE *gnuplotPipe = popen("gnuplot -persistent", "w"); //For data scatter plot
    FILE *gnulinePipe = popen("gnuplot -persistent", "w"); //For Cost function plot
    fprintf(gnuplotPipe, "set title %s \n", "\'Scatter plot of data\'");
    fprintf(gnulinePipe, "set title %s \n", "\'Cost vs Epoch\'");
    //END

    printf("Loading Dataset from dataset/train.csv ...\n\n");
    FILE *stream = fopen("train.csv", "r");
    if (stream == NULL)
    {
        fprintf(stderr, "Error reading file\n");
        return 1;
    }

    //Load data and plot scatter plot
    fprintf(gnuplotPipe, "plot '-'\n");
    while (fscanf(stream, "%lf,%lf", &x[count], &y[count]) == 2)
    {

        fprintf(gnuplotPipe, " %lf %lf \n", x[count], y[count]);
        fflush(gnuplotPipe);

        count = count + 1;
    }
    fprintf(gnuplotPipe, "e\n");

    //    Uncomment to display loaded data
    //    for (int i = 0; i < (int)total_samples; i++) {
    //        printf(" x[%d]:%lf , y[%d]:%lf\n", i,x[i], i,y[i]);
    //    }

    printf("Training parameters using Gradient Descent..\n\n");
    cost = costfunction();

    fprintf(gnulinePipe, "plot '-' with lines lt -1\n");
    while (epoch < total_epochs)
    {
        grad_descent();
        cost = costfunction();

        printf("\nEpoch: %d Cost: %lf Theta0: %lf Theta1: %lf ", epoch, cost, theta[0], theta[1]);

        fprintf(gnulinePipe, "%d %lf\n", epoch, cost);
        fflush(gnulinePipe);

        epoch++;
        usleep(DELAY * 1000); //Uncomment to get results ~instantly
    }
    fprintf(gnulinePipe, "e\n");

    printf("\n\n\n Parameters after %d iterations", epoch);
    printf("\n\tTheta0 : %lf    Theta1 :  %lf", theta[0], theta[1]);

    return 0;
}

void cost()
{
    int i;
    for (i = 0; i < (int)total_samples; i++)
    {
        err[i] = hypothesis(x[i]) - y[i];
    }
}

void grad_descent()
{
    theta[0] = theta[0] - ((alpha / n) * sumerr0);
    theta[1] = theta[1] - ((alpha / n) * sumerr1);
    //printf("Theta0: %lf Theta1: %lf",theta[0],theta[1]);
}

double costfunction()
{
    int i;
    sumsqerr = 0, sumerr0 = 0, sumerr1 = 0;
    cost();
    for (i = 0; i < (int)total_samples; i++)
    {
        sumsqerr += (err[i] * err[i]);
        sumerr0 += err[i];
        sumerr1 += (err[i] * x[i]);
    }
    sumsqerr = sumsqerr / (2.0f * (double)total_samples);
    //	printf("\nSumSqerr: %lf Sumerr0 %lf Sumerr1: %lf",sumsqerr,sumerr0,sumerr1);
    return sumsqerr;
}

double hypothesis(double data)
{
    return (theta[0] + data * theta[1]);
}