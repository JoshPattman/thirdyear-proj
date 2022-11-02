using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ViewPointVisualiser : MonoBehaviour
{

    public float sphereSize = 1;
    public float lineLength = 5;
    void OnDrawGizmos(){
        foreach (Transform child in transform){
            Gizmos.color = Color.green;
            Gizmos.DrawSphere(child.position, sphereSize);
            Gizmos.DrawRay(child.position, child.forward*lineLength);
        }
    }
}
