// utils contains distance functions and data loading function
const utils = require("./utils.js");

class KNN {
  constructor(k, distanceName) {
    this.k = k;
    this.neighbors = [];
    this.distance =
      utils.distances[distanceName] || utils.distances["euclidean"];
  }

  fit(X, y) {
    this.X = X;
    this.y = y;
    this.labels = new Set(this.y);
  }

  // sorts current nearest neigbors from closest to furthest to observed sample
  _sortNeighbors() {
    this.neighbors.sort((elem1, elem2) => {
      if (elem1.distance < elem2.distance) return -1;
      if (elem1.distance > elem2.distance) return 1;
      return 0;
    });
  }

  _getNeighbors(sample) {
    let dist, obj;
    let trainingSample, trainingLabel;
    for (let i = 0; i < this.X.length; i++) {
      trainingSample = this.X[i];
      trainingLabel = this.y[i];
      dist = this.distance(trainingSample, sample);
      obj = { label: trainingLabel, distance: dist };
      if (this.neighbors.length < this.k) {
        this.neighbors.push(obj);
        this._sortNeighbors();
      }
      // since this.neighbors is sorted, the last element is the furthest "nearest neighbor"
      if (dist < this.neighbors.slice(-1)[0].distance) {
        this.neighbors.pop();
        this.neighbors.push(obj);
        this._sortNeighbors();
      }
    }
  }

  predictLabel(sample) {
    if (this.neighbors.length < this.k) this._getNeighbors(sample);
    let counts = new Object();
    for (let label of this.labels) {
      counts[label] = 0;
    }
    for (let neighbor of this.neighbors) {
      counts[neighbor.label]++;
    }
    // predicts label with majority rule (ie. label most present in nearest neighbors)
    return Object.keys(counts).reduce((l1, l2) =>
      counts[l1] > counts[l2] ? l1 : l2
    );
  }
}

main = () => {
  let { X, y } = utils.getIris();
  let classifier = new KNN(5, "euclidean");
  classifier.fit(X.slice(0, -1), y.slice(0, -1));
  let pred = classifier.predictLabel(X.slice(-1)[0]);
  console.log(`true label: ${y.slice(-1)[0]}, prediction: ${pred}`);
};

main();
