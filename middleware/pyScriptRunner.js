let { PythonShell } = require("python-shell");

const pyScriptRunner = (req, res) => {
  let options = {
    mode: "text",
    pythonOptions: ["-u"], // get print results in real-time
    args: [req.body.product, req.body.productCategory, req.body.additionalInfo],
  };

  try {
    PythonShell.run(
      "Scrapy_test_Akash_2_updated.py",
      options,
      function (err, results) {
        if (err) {
          console.log(err);
          throw err;
        }
        // results is an array consisting of messages collected during execution

        //console.log("results: %j", results);
        console.log(results.length);
        let response = {
          data: results,
        };
        res.json(response);
      }
    );
  } catch (error) {
    res.json(error);
  }
};

module.exports = pyScriptRunner;
