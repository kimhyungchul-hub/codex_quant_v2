#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const htmlPath = path.join(__dirname, "..", "dashboard_v2.html");
const html = fs.readFileSync(htmlPath, "utf8");

const start = html.indexOf("function coercePositions");
if (start === -1) {
  console.error("FAIL: coercePositions not found in dashboard_v2.html");
  process.exit(1);
}
let braceStart = html.indexOf("{", start);
if (braceStart === -1) {
  console.error("FAIL: coercePositions missing opening brace");
  process.exit(1);
}
let depth = 0;
let end = -1;
for (let i = braceStart; i < html.length; i += 1) {
  const ch = html[i];
  if (ch === "{") depth += 1;
  if (ch === "}") depth -= 1;
  if (depth === 0) {
    end = i + 1;
    break;
  }
}
if (end === -1) {
  console.error("FAIL: coercePositions missing closing brace");
  process.exit(1);
}
const fnSrc = html.slice(start, end);

const context = { module: { exports: {} }, exports: {} };
vm.runInNewContext(`${fnSrc}\nmodule.exports = coercePositions;`, context);
const coercePositions = context.module.exports;

const positionsMap = {
  "XRP/USDT:USDT": { side: "LONG", entry_price: 1.0, quantity: 1 },
  "ETH/USDT:USDT": { symbol: "ETH/USDT:USDT", side: "SHORT" }
};

const result = coercePositions(positionsMap);
if (!Array.isArray(result)) {
  console.error("FAIL: coercePositions did not return an array");
  process.exit(1);
}
if (result.length !== 2) {
  console.error(`FAIL: expected 2 positions, got ${result.length}`);
  process.exit(1);
}
if (!result.every((p) => p && p.symbol)) {
  console.error("FAIL: expected every position to include symbol");
  process.exit(1);
}

console.log("PASS: coercePositions normalized map to array");
