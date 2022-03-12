#!/usr/bin/env bash

protoc --go_out=. simplepb/simple.proto
protoc --go_out=. enumpb/enum.proto
protoc --go_out=. complexpb/complex.proto
