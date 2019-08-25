import * as axios from "axios";

const baseApiUrl = "http://localhost:8000/";

const predictOpacity = dicomImage => {
  return new Promise((resolve, reject) => {
    axios
      .post(`${baseApiUrl}/predict`, { dicomImage: dicomImage })
      .then(result => {
        resolve(result.data);
      })
      .catch(error => {
        console.log(error);
        reject(error.message);
      });
  });
};

export const PredictService = {
  predictOpacity: predictOpacity
};
