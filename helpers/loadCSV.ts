import { readFileSync } from "fs-extra";
import { first, dropRightWhile, pullAt, isEqual } from "lodash";
import shuffleSeed from "shuffle-seed";

type ConverterType = Record<string, (val: any) => any>;

interface LoadCSVConfigType {
  fileName: string;
  featureColumns: string[];
  labelColumns: string[];
  shuffle?: boolean;
  splitTest?: boolean | number;
  converters?: ConverterType;
}

interface LoadCSVOutputType {
  features: number[][];
  labels: number[][];
  testFeatures?: number[][];
  testLabels?: number[][];
}

/**
 * Extract the required columns
 * @param data Array
 * @param headers Array
 * @param columns Array
 * @returns Array
 */
function extractColumns(
  data: any[][],
  headers: string[],
  columns: string[]
): any[][] {
  const indices: number[] = columns.map((col) => headers.indexOf(col));
  return data.map((row) => pullAt(row, indices));
}

function loadCsv(config: LoadCSVConfigType): LoadCSVOutputType {
  const {
    fileName,
    converters,
    featureColumns,
    labelColumns,
    shuffle,
    splitTest,
  } = config;

  let output: LoadCSVOutputType = {
    features: [],
    labels: [],
  };

  try {
    const fileContents: string = readFileSync(fileName, { encoding: "utf-8" });
    // Convert to 2-D array
    let data: any[][] = fileContents
      .replace(/[\r]/g, "")
      .split("\n")
      .map((row) => row.split(","));
    // Drop empty spaces to the right
    data = dropRightWhile(data, (val) => isEqual(val, [""]));
    const headers: string[] = first(data);

    data = data.map((row, rowIndex) => {
      if (rowIndex === 0) {
        return row;
      }

      // Apply all the converters
      return row.map((col, colIndex) => {
        if (converters?.[headers[colIndex]] != null) {
          const converted = converters[headers[colIndex]](col);
          return Number.isNaN(converted) ? col : converted;
        }

        return parseFloat(col) !== null ? parseFloat(col) : col;
      });
    });

    let featureData: any[][] = extractColumns(data, headers, featureColumns);
    let labelData: any[][] = extractColumns(data, headers, labelColumns);

    // Remove headers
    featureData.shift();
    labelData.shift();

    // Shuffle the data
    if (shuffle ?? false) {
      featureData = shuffleSeed.shuffle(featureData, "phrase");
      labelData = shuffleSeed.shuffle(labelData, "phrase");
    }

    // Split into training and test set
    if (splitTest !== undefined && splitTest !== false) {
      const splitSize =
        typeof splitTest === "number" && isFinite(splitTest)
          ? splitTest
          : Math.floor(data.length / 2);

      output = {
        testFeatures: featureData.slice(0, splitSize),
        testLabels: labelData.slice(0, splitSize),
        features: featureData.slice(splitSize),
        labels: labelData.slice(splitSize),
      };
    } else {
      output = {
        features: featureData,
        labels: labelData,
      };
    }
  } catch (e) {
    console.log(e);
  }

  return output;
}

export default loadCsv;

export type { LoadCSVConfigType, ConverterType, LoadCSVOutputType };
