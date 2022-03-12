const solc = require('solc');
const fs = require('fs');
const path = require('path');

const filePath = path.resolve(__dirname, 'lottery.sol');
const source = fs.readFileSync(filePath, 'utf-8');

const input = {
  language: 'Solidity',
  sources: {
    'lottery.sol': {
      content: source,
    },
  },
  settings: {
    outputSelection: {
      '*': {
        '*': ['*'],
      },
    },
  },
};

const output = JSON.parse(solc.compile(JSON.stringify(input)));

module.exports = output.contracts['lottery.sol'].Lottery;
