import loadCSV from "../../helpers/loadCSV";
import LogisticRegression from "../../algorithms/logistic-regression/index";
import plot from "node-remote-plot";

const { features, labels, testFeatures, testLabels } = loadCSV({
  fileName: "./data/regression.csv",
  shuffle: true,
  splitTest: 50,
  featureColumns: ["horsepower", "weight", "displacement"],
  labelColumns: ["passedemissions"],
  converters: {
    passedemissions: (value) => {
      return value === "TRUE" ? 1 : 0;
    },
  },
});
const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
});

regression.train();

console.log(
  regression.test(testFeatures as number[][], testLabels as number[][])
);

plot({
  x: regression.costHistory.reverse(),
});
