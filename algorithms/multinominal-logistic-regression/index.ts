import * as tf from "@tensorflow/tfjs-node";

interface MultinominalLogisticRegressionConfig {
  learningRate?: number;
  iterations?: number;
  batchSize: number;
  decisionBoundary?: number;
}

// multinomial logistic regression is a classification method that generalizes logistic regression to multiclass problems,
// with more than two possible discrete outcomes
class MultinominalLogisticRegression {
  /**
   * Features tensor
   */
  features: tf.Tensor2D;
  /**
   * Labels tensor
   */
  labels: tf.Tensor2D;
  /**
   * Iteration count
   */
  iterations: number;
  /**
   * Learning rate
   */
  learningRate: number;
  /**
   * Batch size
   */
  batchSize: number;
  /**
   * Decision Boundary
   */
  decisionBoundary: number;
  /**
   * Cost History
   */
  costHistory: number[];
  /**
   * Weights for the algorithm
   */
  weights: tf.Tensor2D;
  /**
   * Mean value for each feature
   */
  mean: tf.Tensor;
  /**
   * Variance value for each feature
   */
  variance: tf.Tensor;

  constructor(
    features: number[][],
    labels: number[][],
    options: MultinominalLogisticRegressionConfig
  ) {
    this.features = this.standardizeFeatures(tf.tensor(features));
    this.labels = tf.tensor(labels);
    this.iterations = options.iterations ?? 1000;
    this.learningRate = options.learningRate ?? 0.1;
    this.batchSize = options.batchSize;
    this.decisionBoundary = options.decisionBoundary ?? 0.5; // Labels below this are considered as failed
    this.costHistory = [];
    this.weights = tf.zeros([this.features.shape[1] ?? 1, 1]);
  }

  standardizeFeatures(features: tf.Tensor2D): tf.Tensor2D {
    if (this.mean === undefined || this.variance === undefined) {
      const { mean, variance } = tf.moments(features, 0);
      this.mean = mean;
      // Add filler to convert 0 in variance to 1
      const filler = variance.cast("bool").logicalNot().cast("float32");
      this.variance = variance.add(filler);
    }

    const standardizedFeatures: tf.Tensor2D = features
      .sub(this.mean)
      .div(this.variance.pow(0.5));

    return tf
      .ones([standardizedFeatures.shape[0], 1])
      .concat(standardizedFeatures, 1);
  }

  gradientDescent(features: tf.Tensor2D, labels: tf.Tensor2D): tf.Tensor2D {
    const currentGuesses = features.matMul(this.weights).softmax();
    const differences = currentGuesses.sub(labels);

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    return this.weights.sub(slopes.mul(this.learningRate));
  }

  train(): void {
    const batchCount = Math.floor(this.features.shape[0] / this.batchSize);

    for (let i = 0; i < this.iterations; i++) {
      for (let j = 0; j < batchCount; j++) {
        const startIndex = j * this.batchSize;

        this.weights = tf.tidy(() => {
          const featureSlice = this.features.slice(
            [startIndex, 0],
            [this.batchSize, -1]
          );

          const labelSlice = this.labels.slice(
            [startIndex, 0],
            [this.batchSize, -1]
          );

          return this.gradientDescent(featureSlice, labelSlice);
        });
      }
      this.updateLearningRate();
    }
  }

  test(features: number[][], labels: number[][]): number {
    const testPredictions = this.predict(features); // Index of alphabet (0 -> A, 1 -> B ...)
    const testLabels = tf.tensor(labels).argMax(1);
    // argMax Returns the indices of the maximum values along an axis
    // Here the index denotes the alphabet -> (0 -> A, 1 -> B ...)

    // Number of guesses that were wrong
    const wrongGuesses = testPredictions
      .notEqual(testLabels)
      .sum()
      .bufferSync()
      .get();
    return (features.length - wrongGuesses) / features.length;
  }

  predict(features: number[][]): tf.Tensor {
    return this.standardizeFeatures(tf.tensor(features))
      .matMul(this.weights)
      .softmax()
      .argMax(1);
  }

  updateLearningRate(): void {
    const guesses = this.features.matMul(this.weights).sigmoid();

    const termOne = this.labels.transpose().matMul(guesses.log());

    const termTwo = this.labels
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(guesses.mul(-1).add(1).add(1e-7).log());
    // log(0) = -Infinity, So add 1e-7 ~ 0.0000001

    const cost = termOne
      .add(termTwo)
      .div(this.features.shape[0])
      .mul(-1)
      .bufferSync()
      .get(0, 0);

    this.costHistory.unshift(cost);

    if (this.costHistory.length < 2) {
      return;
    }

    if (this.costHistory[0] < this.costHistory[1]) {
      this.learningRate = this.learningRate * 0.05 + this.learningRate;
    } else {
      this.learningRate = this.learningRate / 2;
    }
  }
}

export type { MultinominalLogisticRegressionConfig };

export default MultinominalLogisticRegression;
