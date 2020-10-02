// Reference https://gist.github.com/uhho/7228900
const train_data = require("./data/train.json");
const test_data = require("./data/test.json");

linearRegression = (data) => {
  let lr = {};
  let n = data.length;
  let sum_x = 0;
  let sum_y = 0;
  let sum_xy = 0;
  let sum_xx = 0;
  let sum_yy = 0;

  for (let i = 0; i < data.length; i++) {
    sum_x += data[i].x;
    sum_y += data[i].y;
    sum_xy += data[i].x * data[i].y;
    sum_xx += data[i].x * data[i].x;
    sum_yy += data[i].y * data[i].y;
  }

  lr["slope"] = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
  lr["intercept"] = (sum_y - lr.slope * sum_x) / n;
  lr["r2"] = Math.pow(
    (n * sum_xy - sum_x * sum_y) /
      Math.sqrt((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)),
    2
  );

  return lr;
};

let lr = linearRegression(train_data);

test_data.forEach((point) => {
  console.log(`y test: ${point.y}, y predict: ${lr['slope'] * point.x + lr['intercept']}`);
});
