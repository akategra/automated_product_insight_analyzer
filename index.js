const express = require("express");
let { PythonShell } = require("python-shell");
var cors = require("cors");
var bodyParser = require("body-parser");
const { response } = require("express");
const router = express.Router();

const app = express();
const port = 8000;
app.use(cors());
app.use(bodyParser.json());
app.use("/", require("./Routes/homeRoutes"));

app.listen(port, () => {
  console.log(`Example app listening on port ${port}!`);
});
