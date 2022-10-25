using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Threading;
using System.Net;
using System.Net.Sockets;
using System.Text;

[RequireComponent(typeof(Camera))]
public class RemoteCamera : MonoBehaviour
{
    Camera cam;
    byte[] currentImageBuffer;
    Socket conn;
    bool destroyFlag;
    // Start is called before the first frame update
    void Setup()
    {
        cam = GetComponent<Camera>(); 
        currentImageBuffer = RenderFrame();
    }

    // Update is called once per frame
    void Update() {
        if (destroyFlag) Destroy(gameObject);
        var bs = RenderFrame();
        lock(currentImageBuffer){
            bs.CopyTo(currentImageBuffer,0);
        }
    }

    byte[] RenderFrame(){
        // Switch current render target to camera target
        var currentRTex = RenderTexture.active;
        RenderTexture.active = cam.targetTexture;

        // Render the camera
        cam.Render();

        // Copy the rendered data to a new image
        var w = cam.targetTexture.width;
        var h = cam.targetTexture.height;
        var image = new Texture2D(w,h);
        image.ReadPixels(new Rect(0,0,w,h),0,0);
        image.Apply();

        // Encode to png
        var bytes = image.EncodeToPNG();

        // Cleanup
        Destroy(image);
        RenderTexture.active = currentRTex;

        return bytes;
    }

    public void StartSocketConnection(Socket c){
        Setup();
        conn = c;
        Thread t = new Thread(new ThreadStart(SocketConnectionThread));
        t.Start();
    }

    public void SocketConnectionThread(){
        var data = "";
        var dataSize = 0;
        while (true){
            var bs = new byte[1];
            var numBs = conn.Receive(bs);
            data += Encoding.ASCII.GetString(bs, 0, numBs);
            // If we are awaiting a command
            if (data.Length > 0){
                if (data[data.Length-1] == ';'){
                    // A command is now in the buffer string
                    var cmd = data.Substring(0, data.Length-1);
                    data = "";
                    switch (cmd){
                        // The client wants a snapshot
                        case "data":
                            Debug.Log("Client requested frame");
                            var imgBuf = new byte[currentImageBuffer.Length];
                            lock(currentImageBuffer){
                                currentImageBuffer.CopyTo(imgBuf, 0);
                            }
                            conn.Send(Encoding.ASCII.GetBytes(imgBuf.Length.ToString()+";"));
                            conn.Send(imgBuf);
                            Debug.Log("Served frame");
                            break;
                        // The client has left
                        case "exit":
                            Debug.Log("Client requested quit");
                            conn.Shutdown(SocketShutdown.Both);
                            conn.Close();
                            destroyFlag = true;
                            return;
                    }
                }
            }
        }
    }

    /* PROTOCOL

    python: 'data;'
    unity: '<len_data>;<data>'

    python: 'exit;'
    unity: closes socket
    */
}