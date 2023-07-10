import loadCSV from "../../helpers/loadCSV";
import LinearRegression from "../../algorithms/linear-regression/index";
import plot from "node-remote-plot";

const { features, labels, testFeatures, testLabels } = loadCSV({
  fileName: "./data/regression.csv",
  shuffle: true,
  splitTest: 50,
  featureColumns: ["horsepower", "weight", "displacement"],
  labelColumns: ["mpg"],
});
const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 4,
  batchSize: 10,
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

regression.predict([[120, 2, 380]]).print();
