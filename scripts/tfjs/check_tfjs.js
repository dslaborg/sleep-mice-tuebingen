#! /usr/bin/env node

const tf = require("@tensorflow/tfjs");
const tfn = require("@tensorflow/tfjs-node");
const handler = tfn.io.fileSystem("./results/exp001/tfjs/model.json");

tf.loadLayersModel(handler).then( async model => {
  const input = tf.zeros([1, 1920, 1]);
  const output = await model.predict(input);
  output.print();
});
