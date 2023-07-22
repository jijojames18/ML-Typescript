import "module-alias/register";
import loadCSV from "@helpers/loadCSV";
import LogisticRegression from "@algorithms/logistic-regression";
import plot from "node-remote-plot";

const { features, labels, testFeatures, testLabels } = loadCSV({
  fileName: "./data/adult.csv",
  shuffle: true,
  splitTest: 50,
  featureColumns: [
    "age",
    "workclass",
    "education-num",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
  ],
  labelColumns: ["salary"],
  converters: {
    salary: (value) => {
      return value.trim() === "<=50K" ? 1 : 0;
    },
    sex: (value) => {
      return value.trim() === "Male" ? 1 : 0;
    },
    race: (value) => {
      value = value.trim();
      switch (value) {
        case "White":
          return 1;
        case "Asian-Pac-Islander":
          return 2;
        case "Amer-Indian-Eskimo":
          return 3;
        case "Other":
          return 4;
        case "Black":
          return 5;
        default:
          return 6;
      }
    },
    workclass: (value) => {
      value = value.trim();
      switch (value) {
        case "Private":
          return 1;
        case "Self-emp-not-inc":
          return 2;
        case "Self-emp-inc":
          return 3;
        case "Federal-gov":
          return 4;
        case "Local-gov":
          return 5;
        case "State-gov":
          return 6;
        case "Without-pay":
          return 7;
        case "Never-worked":
          return 8;
        default:
          return 9;
      }
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

// eslint-disable-next-line @typescript-eslint/no-confusing-void-expression
console.log(regression.predict([[42, 1, 4, 3, 1, 0, 0, 45]]).print());
