#include <armadillo>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace arma;

mat computeCost(const mat& X, const mat& y, const mat& theta)
{
	mat J;
	int m;
	m = y.n_rows;
	J = arma::sum((pow(((X*theta)-y), 2))/(2*m)) ;
	return J;
}

void gradientDescent(const mat&    X,
                     const mat&    y,
                           double  alpha,
                           int     num_iters,
                           mat&    theta)
{
	mat delta;
	int iter;
	int m ;
	m = y.n_rows;
	//vec J_history = arma::zeros<vec>(num_iters) ;
	for (iter = 0; iter < num_iters; iter++)
	{
		delta = arma::trans(X)*(X*theta-y)/m ;
		theta = theta-alpha*delta ;
	}
}

int main()
{
	mat data;
	data.load("ex1data1.txt");
	mat X = data.col(0);
	mat y = data.col(1);
	
	int m = X.n_elem;
	cout << "m = " << m << endl;
	
	vec X_One(m);
	X_One.ones();
	X.insert_cols(0, X_One);
  
	mat theta = arma::zeros<vec>(2);
	int iterations = 1500 ;
	double alpha = 0.01 ;
	
	mat J = computeCost(X, y, theta);
	J.print("J:");
	
	gradientDescent(X, y, alpha, iterations, theta) ;
	printf("Theta found by gradient descent: \n") ;
	printf("%f %f \n", theta(0), theta(1)) ;
	
	return 0;
}
