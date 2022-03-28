package main

import (
	"context"
	"fmt"
	"grpc-go/simplepb"
	"io"
	"log"
	"net"
	"time"

	"google.golang.org/grpc"
)

type server struct{}

func (*server) SimpleUnary(ctx context.Context, req *simplepb.SimpleUnaryRequest) (*simplepb.SimpleUnaryResponse, error) {
	fmt.Printf("✔ SimpleUnary method was invoked with %v\n", req)

	id := req.GetId()
	isSimple := req.GetIsSimple()
	message := req.GetMessage()

	result := fmt.Sprintf("[ id=%d is_simple=%t message=%s ]", id, isSimple, message)
	res := &simplepb.SimpleUnaryResponse{
		Result: result,
	}

	fmt.Println()
	return res, nil
}

func (*server) SimpleServerStreaming(req *simplepb.SimpleServerStreamingRequest, stream simplepb.SimpleService_SimpleServerStreamingServer) error {
	fmt.Printf("✔ SimpleServerStreaming method was invoked with %v\n", req)

	times := int(req.GetTimes())

	for i := 1; i <= times; i++ {
		res := &simplepb.SimpleServerStreamingResponse{
			Number: int32(i),
		}
		stream.Send(res)
		time.Sleep(500 * time.Millisecond)
	}

	fmt.Println()
	return nil
}

func (*server) SimpleClientStreaming(stream simplepb.SimpleService_SimpleClientStreamingServer) error {
	fmt.Println("✔ SimpleClientStreaming method was invoked with a streaming request")
	fmt.Println("--------------------------------------------------------------------")

	last_number := int32(0)
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			message := fmt.Sprintf("Recieved total %d requests", last_number)
			fmt.Println()
			return stream.SendAndClose(&simplepb.SimpleClientStreamingResponse{
				Message: message,
			})
		}
		if err != nil {
			log.Fatalf("Error while reading client stream: %v\n", err)
		}

		last_number = req.GetNumber()
		fmt.Printf("request with %v\n", req)
	}
}

func (*server) SimpleBidirectionalStreaming(stream simplepb.SimpleService_SimpleBidirectionalStreamingServer) error {
	fmt.Println("✔ SimpleBidirectionalStreaming method was invoked with a streaming request")
	fmt.Println("---------------------------------------------------------------------------")

	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			log.Fatalf("Error while reading client stream: %v\n", err)
		}

		number := req.GetNumber()
		fmt.Printf("request with %v\n", req)
		if number%2 == 1 {
			err = stream.Send(&simplepb.SimpleBidirectionalStreamingResponse{Number: number})
			if err != nil {
				log.Fatalf("Error while sending data to client: %v\n", err)
			}
		}

	}
}

func main() {
	fmt.Printf("\n✅ Starting simple gRPC server ...\n\n")

	lis, err := net.Listen("tcp", "0.0.0.0:50051")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	s := grpc.NewServer()
	simplepb.RegisterSimpleServiceServer(s, &server{})

	if err := s.Serve(lis); err != nil {
		log.Fatalf("Serve failed: %v", err)
	}
}
