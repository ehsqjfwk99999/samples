package main

import (
	"context"
	"fmt"
	"grpc-go/simplepb"
	"io"
	"log"
	"time"

	"google.golang.org/grpc"
)

func main() {
	fmt.Printf("\n✅ Starting simple gRPC client ...\n\n")

	cc, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Dial failed: %v", err)
	}
	defer cc.Close()

	c := simplepb.NewSimpleServiceClient(cc)

	doUnary(c)
	doServerStreaming(c)
	doClientStreaming(c)
	doBidirectionalStreaming(c)
}

func doUnary(c simplepb.SimpleServiceClient) {
	fmt.Println("✔ Starting doUnary function ...")
	fmt.Println("--------------------------------")

	req := &simplepb.SimpleUnaryRequest{
		Id:       123,
		IsSimple: true,
		Message:  "Simple Message",
	}
	res, err := c.SimpleUnary(context.Background(), req)
	if err != nil {
		log.Fatalf("Error while calling SimpleUnary: %v\n", err)
	}
	fmt.Printf("Response from SimpleUnary: %v\n", res.Result)
	fmt.Println()
}

func doServerStreaming(c simplepb.SimpleServiceClient) {
	fmt.Println("✔ Starting doServerStreaming function ...")
	fmt.Println("------------------------------------------")

	req := &simplepb.SimpleServerStreamingRequest{
		Times: 10,
	}

	stream, err := c.SimpleServerStreaming(context.Background(), req)
	if err != nil {
		log.Fatalf("Error while calling SimpleServerStreaming: %v\n", err)
	}
	for {
		res, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatalf("Error while reading stream: %v\n", err)
		}
		fmt.Printf("Response: %v\n", res.GetNumber())
	}
	fmt.Println()
}

func doClientStreaming(c simplepb.SimpleServiceClient) {
	fmt.Println("✔ Starting doClientStreaming function ...")
	fmt.Println("------------------------------------------")

	requests := []*simplepb.SimpleClientStreamingRequest{}
	for i := int32(1); i <= 10; i++ {
		requests = append(requests, &simplepb.SimpleClientStreamingRequest{Number: i})
	}

	stream, err := c.SimpleClientStreaming(context.Background())
	if err != nil {
		log.Fatalf("Error while calling SimpleClientStreaming: %v\n", err)
	}

	for _, req := range requests {
		stream.Send(req)
		time.Sleep(500 * time.Millisecond)
	}

	res, err := stream.CloseAndRecv()
	if err != nil {
		log.Fatalf("Error while receiving response from SimpleClientStreaming: %v\n", err)
	}
	fmt.Printf("Response: %v\n", res.GetMessage())
	fmt.Println()
}

func doBidirectionalStreaming(c simplepb.SimpleServiceClient) {
	fmt.Println("✔ Starting doBidirectionalStreaming function ...")
	fmt.Println("-------------------------------------------------")

	stream, err := c.SimpleBidirectionalStreaming(context.Background())
	if err != nil {
		log.Fatalf("Error while calling SimpleBidirectionalStreaming: %v\n", err)
	}

	waitc := make(chan struct{})

	go func() {
		for i := 1; i <= 10; i++ {
			req := &simplepb.SimpleBidirectionalStreamingRequest{Number: int32(i)}
			fmt.Printf("Sending request: %v\n", req)
			stream.Send(req)
			time.Sleep(500 * time.Millisecond)
		}
		stream.CloseSend()
	}()

	go func() {
		for {
			res, err := stream.Recv()
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Fatalf("Error while receiving response from SimpleBidirectionalStreaming: %v\n", err)
			}
			fmt.Printf("Response: %v\n", res.GetNumber())
		}
		close(waitc)
	}()

	<-waitc
}
