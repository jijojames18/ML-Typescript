import MultinominalLogisticRegression from "../../algorithms/multinominal-logistic-regression/index";
import plot from "node-remote-plot";
import mnist from "mnist-data";
import { flatMap } from "lodash";

const mnistData = mnist.training(0, 6000);

const features = mnistData.images.values.map((image) => flatMap(image));
const labels = mnistData.labels.values.map((label) => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

const regression = new MultinominalLogisticRegression(features, labels, {
  learningRate: 1,
  iterations: 40,
  batchSize: 500,
});

const testMnistData = mnist.testing(0, 10000);
const testFeatures = testMnistData.images.values.map((image) => flatMap(image));
const testEncodedLabels = testMnistData.labels.values.map((label) => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

regression.train();

const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log("Accuracy is", accuracy * 100);

plot({
  x: regression.costHistory.reverse(),
});
