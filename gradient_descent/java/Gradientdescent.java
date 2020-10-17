public class GradientDescent {

private static final double TOLERANCE = 1E-11;

private double theta0;
private double theta1;

public double getTheta0() {
    return theta0;
}

public double getTheta1() {
    return theta1;
}

public GradientDescent(double theta0, double theta1) {
     this.theta0 = theta0;
     this.theta1 = theta1;
}

public double getHypothesisResult(double x){
    return theta0 + theta1*x;
}

private double getResult(double[][] trainingData, boolean enableFactor){
    double result = 0;
    for (int i = 0; i < trainingData.length; i++) {
        result = (getHypothesisResult(trainingData[i][0]) - trainingData[i][1]);
        if (enableFactor) result = result*trainingData[i][0]; 
    }
    return result;
}

public void train(double learningRate, double[][] trainingData){
    int iteration = 0;
    double delta0, delta1;
    do{
        iteration++;
        System.out.println("SUBS: " + (learningRate*((double) 1/trainingData.length))*getResult(trainingData, false));
        double temp0 = theta0 - learningRate*(((double) 1/trainingData.length)*getResult(trainingData, false));
        double temp1 = theta1 - learningRate*(((double) 1/trainingData.length)*getResult(trainingData, true));
        delta0 = theta0-temp0; delta1 = theta1-temp1;
        theta0 = temp0; theta1 = temp1;
    }while((Math.abs(delta0) + Math.abs(delta1)) > TOLERANCE);
    System.out.println(iteration);
}
