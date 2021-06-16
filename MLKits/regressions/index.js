require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');

const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');

let { features,
  labels,
  testFeatures,
  testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower'],
    labelColumns: ['mpg']
  });

const regression = new LinearRegression(features, labels, {
  learningRation: 0.1,
  iterations: 10,
});

regression.train();

console.log('Updated M is:', regression.m, 'Updated B is:', regression.b);
