const Web3 = require('web3');
const HDWalletProvider = require('@truffle/hdwallet-provider');

const { abi, evm } = require('./compile.js');

const provider = new HDWalletProvider(
  '<seed phrase>',
  'https://rinkeby.infura.io/v3/4279490033e147e986f9e296f0fad8ea'
);

const web3 = new Web3(provider);

const deploy = async () => {
  const accounts = await web3.eth.getAccounts();

  const result = await new web3.eth.Contract(abi)
    .deploy({
      data: evm.bytecode.object,
    })
    .send({ from: accounts[0], gas: '1000000' });

  console.log('Conatract Address:', result.options.address);

  provider.engine.stop();
};

deploy();
console.log('✅ Deploy Finished');
