class CustomDecisionTree {
  constructor() {}

  majorityCnt(classes) {
    let classesCount = {};
    classes.forEach((value) => {
      if (!Object.keys(classesCount).includes(value)) {
        classesCount[value] = 0;
      }
      classesCount[value] += 1;
    });
    let sorted = Object.keys(classesCount).sort(function (a, b) {
      return classesCount[b] - classesCount[b];
    });
    return sorted[0];
  }

  calcShannonEnt(dataSet) {
    let numEntries = dataSet.length;
    let labelCounts = {};
    dataSet.forEach((featVec) => {
      let currentLabel = featVec[featVec.length - 1];
      if (!labelCounts[currentLabel]) {
        labelCounts[currentLabel] = 0;
      }
      labelCounts[currentLabel] += 1;
    });
    let shannonEnt = 0.0;
    Object.keys(labelCounts).forEach((key) => {
      let prob = labelCounts[key] / numEntries;
      shannonEnt -= prob * Math.log(prob, 2);
    });

    return shannonEnt;
  }

  splitDataSet(dataSet, axis, value) {
    let retDataSet = [];
    dataSet.forEach((featVec) => {
      if (featVec[axis] === value) {
        let newVec = [...featVec];
        newVec.splice(axis, 1);
        retDataSet.push(newVec);
      }
    });
    return retDataSet;
  }

  chooseBestFeatureToSplit(dataSet, labels) {
    let numFeatures = Object.keys(dataSet[0]).length - 1;
    let baseEntropy = this.calcShannonEnt(dataSet);
    let bestInfoGain = -1;
    let bestFeature = 0;
    for (let i = 0; i < numFeatures; i++) {
      let uniqueValues = dataSet
        .map((sample) => sample[i])
        .filter((v, i, a) => a.indexOf(v) === i);
      let newEntropy = 0.0;
      uniqueValues.forEach((value) => {
        let subDataSet = this.splitDataSet(dataSet, i, value);
        let prob = subDataSet.length / dataSet.length;
        newEntropy += prob * this.calcShannonEnt(subDataSet);
      });

      let infoGain = baseEntropy - newEntropy;
      console.log(`${labels[i]}: ${infoGain}`);
      if (infoGain > bestInfoGain) {
        bestInfoGain = infoGain;
        bestFeature = i;
      }
    }

    console.log("the best feature to split is", labels[bestFeature]);
    return bestFeature;
  }

  createTree(dataSet, labels) {
    let classList = dataSet.map((sample) => sample[sample.length - 1]);
    if (classList.length === 0) return;

    if (
      classList.filter((value) => value === classList[0]).length ===
      classList.length
    ) {
      return classList[0];
    }

    if (Object.keys(dataSet[0]).length == 1) return this.majorityCnt(classList);

    let featureVectorList = dataSet.map((sample) =>
      [...sample].slice(0, sample.length - 1)
    );
    let bestFeat = this.chooseBestFeatureToSplit(featureVectorList, labels);
    let bestFeatLabel = labels[bestFeat];
    let myTree = {};
    myTree[bestFeatLabel] = {};
    labels.splice(bestFeat, 1);
    let uniqueValues = dataSet
      .map((sample) => sample[bestFeat])
      .filter((v, i, a) => a.indexOf(v) === i);

    uniqueValues.forEach((value) => {
      let subLabels = [...labels];
      myTree[bestFeatLabel][value] = this.createTree(
        this.splitDataSet(dataSet, bestFeat, value),
        subLabels
      );
    });

    return myTree;
  }
}

var dataset = require("./data/train.json");

dataset = dataset.map((sample) => {
  sample.sex = sample.sex === "male" ? 1 : 0;

  if (isNaN(sample.embarked)) sample.embarked = 1;

  return Object.values(sample);
});

const labels = ["pclass", "sex", "embarked", "survived"];

custom_DTree = new CustomDecisionTree();
console.log(JSON.stringify(custom_DTree.createTree(dataset, labels), null, 2)); // spacing level = 2
