import "module-alias/register";
import * as tf from "@tensorflow/tfjs";
import loadCSV from "@helpers/loadCSV";
import knn, { AlgorithmType } from "@algorithms/knn";

const k = 15;

const { features, labels, testFeatures, testLabels } = loadCSV({
  fileName: "./data/winequality-red.csv",
  shuffle: true,
  splitTest: 10,
  featureColumns: [
    "fixed acidity",
    "citric acid",
    "residual sugar",
    "total sulfur dioxide",
  ],
  labelColumns: ["quality"],
});

const featureTensor = tf.tensor(features);
const labelsTensor = tf.tensor(labels);

const testsetCount = testFeatures?.length ?? 1;
let failureCount = 0;

testFeatures?.forEach((testFeature, i) => {
  const result = knn(
    featureTensor,
    labelsTensor,
    tf.tensor(testFeature),
    k,
    AlgorithmType.CLASSIFICATION
  );
  const actualLabel = testLabels?.[i]?.[0] ?? 0;
  if (actualLabel !== result) {
    failureCount++;
  }
});

const err = (failureCount - testsetCount) / testsetCount;
console.log("Error", err * 100);
