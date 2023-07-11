import * as tf from "@tensorflow/tfjs-node";

enum  AlgorithmType {
  CLASSIFICATION,
  REGRESSION
}

const knn = (
  features: tf.Tensor<tf.Rank>,
  labels: tf.Tensor<tf.Rank>,
  predictionPoint: tf.Tensor<tf.Rank>,
  k: number,
  type: AlgorithmType
): number => {
  const { mean, variance } = tf.moments(features, 0);

  // Apply standard deviation
  const scaledPredicitionTensor = predictionPoint
    .sub(mean)
    .div(variance.pow(0.5));

  const neighbors = features
    .sub(mean)
    .div(variance.pow(0.5)) // Apply standard deviation
    .sub(scaledPredicitionTensor)
    .pow(2)
    .sum(1)
    .pow(0.5) // Calculate euclidean distance
    .expandDims(1) // Convert to 2-D after sum
    .concat(labels, 1) // Add labels column
    .unstack()
    .sort((a, b) => (a.bufferSync().get(0) > b.bufferSync().get(0) ? 1 : -1)) // Sort based on shortest distance
    .slice(0, k); // Take only lowest k values


  if (type === AlgorithmType.REGRESSION) {
    return (
      neighbors
      .reduce(
        (acc: number, pair: tf.Tensor<tf.Rank>) =>
          acc + pair.bufferSync().get(1),
        0
      ) / k // Take average of lowest values
    );
  } else {
    return (
      neighbors
      .reduce(
        (acc: number, pair: tf.Tensor<tf.Rank>) => {
          const current = pair.bufferSync().get(1);
          return acc > current ? acc : current;
        },
        0
      )
    );
  }
};

export { AlgorithmType };

export default knn;
