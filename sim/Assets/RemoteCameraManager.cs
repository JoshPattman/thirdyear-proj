using System.Collections;
using System.Collections.Generic;
using System.Collections.Concurrent;
using UnityEngine;
using System;
using System.Threading;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class RemoteCameraManager : MonoBehaviour
{
    public int port = 9876;
    ConcurrentQueue<Socket> newSockets;
    public RemoteCamera prefab;
    // Start is called before the first frame update
    void Start()
    {
        newSockets = new ConcurrentQueue<Socket>();
        Thread t = new Thread(new ThreadStart(ListenForNewSockets));
        t.Start();
    }

    // Update is called once per frame
    void Update()
    {
        Socket newSocket;
        while (newSockets.TryDequeue(out newSocket)){
            // This is a new socket and it needs a camera
            Debug.Log("Client connected, spawing camera");
            var newCam = Instantiate(prefab.gameObject).GetComponent<RemoteCamera>();
            newCam.StartSocketConnection(newSocket);
        }
    }

    void ListenForNewSockets(){
        IPHostEntry host = Dns.GetHostEntry("localhost");
        IPAddress addr = host.AddressList[0];
        IPEndPoint localEP = new IPEndPoint(addr, port);
        Socket listener = new Socket(addr.AddressFamily, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(localEP);
        // Max 10 sockets waiting at once
        listener.Listen(10);
        while(true){
            try{
                newSockets.Enqueue(listener.Accept());
            } catch(Exception e){
                Debug.Log(e);
            }
        }
    }
}
