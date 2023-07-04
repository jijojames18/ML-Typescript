import * as tf from "@tensorflow/tfjs-node";

interface LinearRegressionConfig {
  learningRate?: number;
  iterations?: number;
  batchSize: number;
}

class LinearRegression {
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
   * Mean Squared Error History
   */
  mseHistory: number[];
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
    options: LinearRegressionConfig
  ) {
    this.features = this.standardizeFeatures(tf.tensor(features));
    this.labels = tf.tensor(labels);
    this.iterations = options.iterations ?? 1000;
    this.learningRate = options.learningRate ?? 0.1;
    this.batchSize = options.batchSize;
    this.mseHistory = [];
    this.weights = tf.zeros([this.features.shape[1] ?? 1, 1]);
  }

  standardizeFeatures(features: tf.Tensor2D): tf.Tensor2D {
    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    if (this.mean === undefined || this.variance === undefined) {
      const { mean, variance } = tf.moments(features, 0);
      this.mean = mean;
      // Add filler to convert 0 in variance to 1
      const filler = variance.cast("bool").logicalNot().cast("float32");
      this.variance = variance.add(filler);
    }

    return features.sub(this.mean).div(this.variance.pow(0.5));
  }

  gradientDescent(features: tf.Tensor2D, labels: tf.Tensor2D): tf.Tensor2D {
    const currentGuesses = features.matMul(this.weights);
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
    const featuresTensor = this.standardizeFeatures(tf.tensor(features));
    const labelsTensor = tf.tensor(labels);

    const prediction = featuresTensor.matMul(this.weights);

    const res = labelsTensor.sub(prediction).pow(2).sum().bufferSync().get();
    const total = labelsTensor
      .sub(labelsTensor.mean())
      .pow(2)
      .sum()
      .bufferSync()
      .get();

    return 1 - res / total;
  }

  predict(features: number[][]): tf.Tensor {
    return this.standardizeFeatures(tf.tensor(features)).matMul(this.weights);
  }

  updateLearningRate(): void {
    const mse = tf.tidy(() =>
      this.features
        .matMul(this.weights)
        .sub(this.labels)
        .pow(2)
        .sum(0)
        .div(this.features.shape[0])
        .pow(0.5)
        .bufferSync()
        .get()
    );

    this.mseHistory.unshift(mse);

    if (this.mseHistory.length < 2) {
      return;
    }

    if (this.mseHistory[0] < this.mseHistory[1]) {
      this.learningRate = this.learningRate * 0.05 + this.learningRate;
    } else {
      this.learningRate = this.learningRate / 2;
    }
  }
}

export type { LinearRegressionConfig };

export default LinearRegression;
