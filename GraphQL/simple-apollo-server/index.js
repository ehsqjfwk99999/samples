const { ApolloServer, gql } = require('apollo-server');
const { faker } = require('@faker-js/faker');

const numData = 10;
const numInfo = 10;

let simpleDatas = [];
for (let i = 0; i < numData; i++) {
  const simpleData = {
    id: i + 1,
    name: faker.name.lastName(),
    gender: faker.name.gender(),
    infoId: Math.floor(Math.random() * numInfo) + 1,
  };
  simpleDatas.push(simpleData);
}

let simpleInfos = [];
for (let i = 0; i < numInfo; i++) {
  const simpleInfo = {
    id: i + 1,
    job: faker.name.jobType(),
  };
  simpleInfos.push(simpleInfo);
}

const typeDefs = gql`
  type Query {
    simpleDatas: [SimpleData]
    simpleData(id: Int): SimpleData
    simpleInfos: [SimpleInfo]
  }
  type Mutation {
    deleteData(id: Int): SimpleData
  }
  type SimpleData {
    id: Int
    name: String
    gender: String
    infoId: Int
    info: SimpleInfo
  }
  type SimpleInfo {
    id: Int
    job: String
  }
`;

const resolvers = {
  Query: {
    simpleDatas: () =>
      simpleDatas.map((d) => {
        d.info = simpleInfos.filter((i) => i.id === d.infoId)[0];
        return d;
      }),
    simpleData: (parent, args, context, info) =>
      simpleDatas.filter((d) => d.id === args.id)[0],
    simpleInfos: () => simpleInfos,
  },
  Mutation: {
    deleteData: (parent, args, context, info) => {
      const deleted = simpleDatas.filter((d) => d.id === args.id)[0];
      simpleDatas = simpleDatas.filter((d) => d.id !== args.id);
      return deleted;
    },
  },
};

const server = new ApolloServer({ typeDefs, resolvers });

server.listen().then(({ url }) => {
  console.log(`🚀 Server ready at ${url}`);
});
