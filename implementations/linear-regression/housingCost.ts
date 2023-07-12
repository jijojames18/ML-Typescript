import 'module-alias/register';
import loadCSV from "@helpers/loadCSV";
import LinearRegression from "@algorithms/linear-regression";
import plot from "node-remote-plot";

const { features, labels, testFeatures, testLabels } = loadCSV({
  fileName: "./data/BostonHousing.csv",
  shuffle: true,
  splitTest: 50,
  featureColumns: ["age", "tax", "rm", "dis"],
  labelColumns: ["medv"],
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 100,
  batchSize: 1,
});

regression.train();
const r2 = regression.test(
  testFeatures as number[][],
  testLabels as number[][]
);

plot({
  x: regression.mseHistory.reverse(),
  xLabel: "Iteration #",
  yLabel: "Mean Squared Error",
});

console.log("R2 is", r2);

regression.predict([[61.8, 307, 5.949, 4.7075]]).print(); //20.4
