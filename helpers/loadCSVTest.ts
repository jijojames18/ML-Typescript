import loadCsv, { type LoadCSVOutputType } from "./loadCSV";

const data: LoadCSVOutputType = loadCsv({
  fileName: "./helpers/test.csv",
  featureColumns: ["height", "value"],
  labelColumns: ["passed"],
  shuffle: true,
  splitTest: false,
  converters: {
    passed: (val) => (val === "TRUE" ? 1 : 0),
  },
});

console.log(data);
