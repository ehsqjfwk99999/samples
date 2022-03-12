package main

import (
	"fmt"
	"log"
	"os"
	"protobuf-3/complexpb"
	"protobuf-3/enumpb"
	"protobuf-3/simplepb"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

func main() {
	// sm := createSimpleMessage()

	// readAndWriteDemo(sm)
	// jsonDemo(sm)
	// createComplexMessage()
	// createEnumMessage()
}

func createSimpleMessage() *simplepb.SimpleMessage {
	sm := simplepb.SimpleMessage{
		Id:         12345,
		IsSimple:   true,
		Message:    "Simple Message",
		SimpleList: []int32{1, 2, 3, 4, 5},
	}

	return &sm
}

func readAndWriteDemo(sm *simplepb.SimpleMessage) {
	writeToFile("simple.bin", sm)
	defer os.Remove("simple.bin")

	sm2 := &simplepb.SimpleMessage{}
	readFromFile("simple.bin", sm2)
	fmt.Println("File Content: ", sm2)
}
func writeToFile(fname string, pb proto.Message) error {
	out, err := proto.Marshal(pb)
	if err != nil {
		log.Fatalln("Marshal failed", err)
		return err
	}

	if err := os.WriteFile(fname, out, 0644); err != nil {
		log.Fatalln("WriteFile failed", err)
		return err
	}

	fmt.Println("Data successfully written to simple.bin!")
	return nil
}
func readFromFile(fname string, pb proto.Message) error {
	in, err := os.ReadFile(fname)
	if err != nil {
		log.Fatalln("Readfile failed")
		return err
	}

	err = proto.Unmarshal(in, pb)
	if err != nil {
		log.Fatalln("Unmarshal failed")
		return err
	}

	return nil
}

func jsonDemo(pb proto.Message) {
	smAsString := toJson(pb)
	fmt.Println("JSON : ", smAsString)

	sm2 := &simplepb.SimpleMessage{}
	fromJson(smAsString, sm2)
	fmt.Println("Simple Message : ", sm2)
}
func toJson(pb proto.Message) string {
	m := protojson.MarshalOptions{}
	out := m.Format(pb)

	return out
}
func fromJson(in string, pb proto.Message) {
	if err := protojson.Unmarshal([]byte(in), pb); err != nil {
		log.Fatalln("Unmarshal JSON to pb struct failed")
	}
}

func createComplexMessage() {
	cm := complexpb.ComplexMessage{
		OneDummy: &complexpb.DummyMessage{
			Id:      1,
			Message: "first message",
		},
		MultipleDummies: []*complexpb.DummyMessage{
			&complexpb.DummyMessage{
				Id:      2,
				Message: "second message",
			},
			&complexpb.DummyMessage{
				Id:      3,
				Message: "third message",
			},
		},
	}

	fmt.Println("Complex Message : ", cm)
}

func createEnumMessage() {
	em := enumpb.EnumMessage{
		Id:           123,
		DayOfTheWeek: enumpb.DayOfTheWeek_MONDAY,
	}

	fmt.Println("Enum Message : ", em)
}
