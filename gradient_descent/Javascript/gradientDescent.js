//Gradient descent on 2 dimensions

// Assuming the quadratic function represents the loss in the y-axis and the weight on the x-axis

class GradientDescent {

    // Constructor parameter
    // lr --> learning rate
    // a,b,c --> coefficients in the equation ax^2 + bx + c
    constructor(lr,a,b,c) {

        this.lr = lr;
        this.a=a;
        this.b=b;
        this.c=c;

        //Weight parameter which needs to be optimized
        this.weight = Math.random()*100;
        
        // quadratic function which calculates y given x
        this.quadratic = (x) => {
            return this.a*(Math.pow(x,2)) + this.b*x + this.c;
        }
        
        // Quadratic differential which gives dy/dx given x
        this.diff = (x) => {
            return 2*(this.a)*x + this.b;
        }

    }

    // Gradient descent function which takes in the number of iterations
    gradient_descent(iterations) {
    

        for(let i=0;i<iterations;i++) {
            console.log(`The gradient is: ${this.diff(this.weight)}`)
            console.log(`Current x-position: ${this.weight}`)
            console.log(`The loss is: ${this.quadratic(this.weight)}`);
            console.log("");

            // Gradient descent step --> weight = weight - lr*(dy/dx)
            this.weight -= this.lr * this.diff(this.weight);
        }
        
        console.log("Gradient descent stopped, number of iterations reached");
        console.log(`The gradient is: ${this.diff(this.weight)}`)
        console.log(`Final x-position: ${this.weight}`)
        console.log(`The loss is: ${this.quadratic(this.weight)}`);
        return this.weight;
    }
    



}    




let test = new GradientDescent(0.05,1,2,3)
console.log(test.gradient_descent(199))