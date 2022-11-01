using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
public class Waypoint : MonoBehaviour
{
    public Waypoint[] nextPoints;

    void Update(){
        foreach (var point in nextPoints)
        {
            var dir = point.transform.position - transform.position;
            Debug.DrawLine(transform.position, transform.position + (dir/2), Color.red);
        }
    }
}
