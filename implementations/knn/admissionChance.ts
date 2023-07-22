import "module-alias/register";
import * as tf from "@tensorflow/tfjs";
import loadCSV from "@helpers/loadCSV";
import knn, { AlgorithmType } from "@algorithms/knn";

const k = 10;

const { features, labels, testFeatures, testLabels } = loadCSV({
  fileName: "./data/Admission.csv",
  shuffle: true,
  splitTest: 50,
  featureColumns: ["GRE Score", "TOEFL Score", "CGPA"],
  labelColumns: ["Chance"],
});

const featureTensor = tf.tensor(features);
const labelsTensor = tf.tensor(labels);

testFeatures?.forEach((testFeature, i) => {
  const result = knn(
    featureTensor,
    labelsTensor,
    tf.tensor(testFeature),
    k,
    AlgorithmType.REGRESSION
  );
  const actualLabel = testLabels?.[i]?.[0] ?? 0;
  const err = (actualLabel - result) / actualLabel;
  console.log("Error", err * 100);
});

console.log(
  knn(
    featureTensor,
    labelsTensor,
    tf.tensor([[336, 119, 9.8]]),
    k,
    AlgorithmType.REGRESSION
  )
);
