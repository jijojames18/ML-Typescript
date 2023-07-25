import "module-alias/register";
import loadCSV from "@helpers/loadCSV";
import MultinominalLogisticRegression from "@algorithms/multinominal-logistic-regression";
import plot from "node-remote-plot";

const { features, labels, testFeatures, testLabels } = loadCSV({
  fileName: "./data/adult.csv",
  shuffle: true,
  splitTest: 50,
  featureColumns: ["salary", "workclass", "education-num", "occupation", "age"],
  labelColumns: ["education"],
  converters: {
    salary: (value) => {
      return value.trim() === "<=50K" ? 1 : 0;
    },
    education: (value) => {
      value = value.trim();
      switch (value) {
        case "Bachelors":
          return 1;
        case "Some-college":
          return 2;
        case "11th":
          return 3;
        case "HS-grad":
          return 4;
        case "Prof-school":
          return 5;
        case "Assoc-acdm":
          return 6;
        case "Assoc-voc":
          return 7;
        case "9th":
          return 8;
        case "7th-8th":
          return 10;
        case "12th":
          return 11;
        case "Masters":
          return 12;
        case "1st-4th":
          return 13;
        case "10th":
          return 14;
        case "Doctorate":
          return 15;
        case "5th-6th":
          return 16;
        case "Preschool":
          return 17;
        default:
          return 0;
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
          return 0;
      }
    },
    occupation: (value) => {
      value = value.trim();
      switch (value) {
        case "Tech-support":
          return 1;
        case "Craft-repair":
          return 2;
        case "Other-service":
          return 3;
        case "Sales":
          return 4;
        case "Exec-managerial":
          return 5;
        case "Prof-specialty":
          return 6;
        case "Handlers-cleaners":
          return 7;
        case "Machine-op-inspct":
          return 8;
        case "Adm-clerical":
          return 10;
        case "Farming-fishing":
          return 11;
        case "Transport-moving":
          return 12;
        case "Priv-house-serv":
          return 13;
        case "Protective-serv":
          return 14;
        case "Armed-Forces":
          return 15;
        default:
          return 0;
      }
    },
  },
});

const encodedLabels: number[][] = [];
labels.forEach((entry: number[]) => {
  const label = entry[0];
  const row = new Array(18).fill(0);
  row[label] = 1;
  encodedLabels.push(row);
});

const regression = new MultinominalLogisticRegression(features, encodedLabels, {
  learningRate: 1,
  iterations: 40,
  batchSize: 100,
});

regression.train();

const encodedTestLabels: any = [];
testLabels?.forEach((entry: number[]) => {
  const label = entry[0];
  const row = new Array(18).fill(0);
  row[label] = 1;
  encodedTestLabels.push(row);
});

const accuracy = regression.test(
  testFeatures as number[][],
  encodedTestLabels as number[][]
);
console.log("Accuracy is", accuracy * 100);

plot({
  x: regression.costHistory.reverse(),
});

regression.predict([[0, 5, 9, 14, 49]]).print(); // 4
