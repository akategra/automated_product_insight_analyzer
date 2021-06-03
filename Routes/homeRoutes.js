const express = require("express");
const pyScriptRunner = require("../middleware/pyScriptRunner.js");
const router = express.Router();

router.get("/");
router.post("/", pyScriptRunner);

module.exports = router;
