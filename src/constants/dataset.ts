export const CLASS_NAMES: Record<number, string> = {
  0: 'T-shirt/Top',
  1: 'Trouser',
  2: 'Pullover',
  3: 'Dress',
  4: 'Coat',
  5: 'Sandal',
  6: 'Shirt',
  7: 'Sneaker',
  8: 'Bag',
  9: 'Ankle Boot',
};

export const CLASS_ICONS: Record<number, string> = {
  0: '👕',
  1: '👖',
  2: '🧥',
  3: '👗',
  4: '🧣',
  5: '👡',
  6: '👔',
  7: '👟',
  8: '👜',
  9: '👢',
};

export const DATASET_STATS = {
  totalSamples: 70000,
  trainSamples: 60000,
  testSamples: 10000,
  imageSize: '28×28 px',
  channels: 'Grayscale (1ch)',
  classes: 10,
  source: 'Zalando Research',
};

export const SVM_PARAMS = {
  algorithm: 'Support Vector Classifier',
  kernel: 'linear',
  C: 1.0,
  decision_function: 'ovr',
  max_iter: 2000,
  feature_extraction: 'HOG + PCA(150)',
  scaler: 'StandardScaler',
  cross_validation: 'StratifiedKFold (k=5)',
};

export const TRAINING_METRICS = {
  accuracy: 89.4,
  precision: 88.9,
  recall: 89.1,
  f1Score: 89.0,
  cvMean: 88.7,
  cvStd: 0.8,
  trainTime: '~4.2 min',
};
