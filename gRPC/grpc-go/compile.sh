#!/bin/bash

protoc simplepb/simple.proto --go_out=plugins=grpc:.
