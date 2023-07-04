import * as tf from "@tensorflow/tfjs";
import loadCSV from "../../helpers/loadCSV";
import knn from "./index";

const k = 10;

const { features, labels, testFeatures, testLabels } = loadCSV({
  fileName: "./algorithms/data/knn.csv",
  shuffle: true,
  splitTest: 10,
  featureColumns: ["lat", "long", "sqft_lot", "sqft_living"],
  labelColumns: ["price"],
});

const featureTensor = tf.tensor(features);
const labelsTensor = tf.tensor(labels);

testFeatures?.forEach((testFeature, i) => {
  const result = knn(featureTensor, labelsTensor, tf.tensor(testFeature), k);
  const actualLabel = testLabels?.[i]?.[0] ?? 0;
  const err = (actualLabel - result) / actualLabel;
  console.log("Error", err * 100);
});
