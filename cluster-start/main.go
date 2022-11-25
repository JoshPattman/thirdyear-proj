package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"sync"
	"syscall"
)

var processes = []*exec.Cmd{}
var lock = sync.Mutex{}
var stdoutChan = make(chan string, 5)

func main() {
	// Parse args
	target := flag.String("t", "target", "target job to start")
	flag.Parse()

	// Load job file
	job := []Node{}
	if bs, err := os.ReadFile(*target); err != nil {
		fail(err.Error())
	} else if err := json.Unmarshal(bs, &job); err != nil {
		fail(err.Error())
	}

	// Setup nodes clean exit
	sigchan := make(chan os.Signal, 1)
	signal.Notify(sigchan,
		syscall.SIGINT,
		syscall.SIGKILL,
		syscall.SIGTERM,
		syscall.SIGQUIT)
	go func() {
		<-sigchan
		fmt.Println("Stopping nodes")
		lock.Lock()
		for _, p := range processes {
			if p.Process != nil {
				p.Process.Kill()
			}
		}
		fmt.Println("Stopped all nodes")
		os.Exit(0)
	}()

	// Start nodes
	for _, n := range job {
		err := func() error {
			lock.Lock()
			defer lock.Unlock()
			fmt.Println("Starting ", n)
			cmd := exec.Command(n.Command, n.Args...)
			var outputStream io.Writer
			if n.Output != "" {
				if outfile, err := os.Create(n.Output); err != nil {
					return err
				} else {
					outputStream = outfile
				}
			}
			if n.Show {
				outputStream = NewScreenWriter(outputStream)
			}
			cmd.Stdout = outputStream
			cmd.Stderr = outputStream
			processes = append(processes, cmd)
			return cmd.Start()
		}()
		if err != nil {
			fmt.Println(err)
			sigchan <- syscall.SIGQUIT
			<-make(chan bool)
		}
	}

	for {
		msg := <-stdoutChan
		fmt.Print(msg)
	}
}

type Node struct {
	Command string   `json:"cmd"`
	Args    []string `json:"arg"`
	Output  string   `json:"out"`
	Show    bool     `json:"show"`
}

func (n Node) String() string {
	return fmt.Sprint(n.Command, " ", n.Args)
}

func fail(s string) {
	fmt.Println(s)
	os.Exit(1)
}

type ScreenWriter struct {
	Next   io.Writer
	stream chan byte
}

func NewScreenWriter(w io.Writer) io.Writer {
	sw := &ScreenWriter{w, make(chan byte, 100)}
	go sw.readStream()
	return sw
}

func (sr *ScreenWriter) Write(ps []byte) (n int, err error) {
	for _, p := range ps {
		sr.stream <- p
	}
	if sr.Next != nil {
		return sr.Next.Write(ps)
	}
	return len(ps), nil
}

func (sr *ScreenWriter) readStream() {
	buf := []byte{}
	for {
		b := <-sr.stream
		buf = append(buf, b)
		if b == '\n' {
			stdoutChan <- string(buf)
			buf = []byte{}
		}
	}
}
