import crypto from "crypto";

interface BlockStructure {
  data: string;
  prevHash: string;
  height: number;
  hash: string;
}

class Block implements BlockStructure {
  public hash: string;

  constructor(
    public data: string,
    public prevHash: string,
    public height: number
  ) {
    this.hash = Block.calculateHash(data, prevHash, height);
  }

  static calculateHash(data: string, prevHash: string, height: number) {
    const toHash = `${data}${prevHash}${height}`;
    return crypto.createHash("sha256").update(toHash).digest("hex");
  }
}

class Blockchain {
  private blocks: Block[];

  constructor() {
    this.blocks = [];
  }

  private getPrevHash(): string {
    if (this.blocks.length == 0) return "";
    return this.blocks[this.blocks.length - 1].hash;
  }

  public addBlock(data: string): void {
    const newBlock = new Block(
      data,
      this.getPrevHash(),
      this.blocks.length + 1
    );
    this.blocks.push(newBlock);
  }

  public getBlocks(): Block[] {
    return [...this.blocks];
  }
}

const blockchain = new Blockchain();

blockchain.addBlock("First Block");
blockchain.addBlock("Second Block");
blockchain.addBlock("Third Block");

console.log(blockchain.getBlocks());
