using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Car : MonoBehaviour
{
    Rigidbody rb;

    public bool hasCrashed;

    public float speed = 13;
    public float turnSpeed = 180;

    public Waypoint targetPoint;
    Waypoint lastPoint;
    public float waypointDist = 1;
    public float roadWidth = 2;

    public GameObject[] models;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        lastPoint = targetPoint.nextPoints[0];
        foreach(var m in models) m.SetActive(false);
        models[Random.Range(0, models.Length)].SetActive(true);
    }

    void Update(){
        var laneOffset = Vector3.Cross((targetPoint.transform.position - lastPoint.transform.position).normalized, Vector3.up) * roadWidth;
        var dir = Vector3.ProjectOnPlane((targetPoint.transform.position+laneOffset) - transform.position, Vector3.up);
        Debug.DrawLine(transform.position, transform.position+dir, Color.green);
        if (dir.magnitude < waypointDist){
            Waypoint nextPoint = lastPoint;
            while (nextPoint == lastPoint) nextPoint = targetPoint.nextPoints[Random.Range(0, targetPoint.nextPoints.Length)];
            lastPoint = targetPoint;
            targetPoint = nextPoint;
        }
    }

    void FixedUpdate()
    {
        var laneOffset = Vector3.Cross((targetPoint.transform.position - lastPoint.transform.position).normalized, Vector3.up) * roadWidth;
        var dir = Vector3.ProjectOnPlane((targetPoint.transform.position+laneOffset) - transform.position, Vector3.up);
        rb.angularVelocity = GetTorqueFor(Quaternion.LookRotation(dir), turnSpeed);
        if (rb.angularVelocity.magnitude > turnSpeed/2){
            rb.velocity = transform.forward * speed / 3;
        } else{
            rb.velocity = transform.forward * speed;
        }
    }

    Vector3 GetTorqueFor(Quaternion rot, float maxAngle){
        var a = rot * Quaternion.Inverse(transform.rotation);
        float angle = 0.0f;
        Vector3 axis = Vector3.zero;
        a.ToAngleAxis(out angle, out axis);
        if (angle > 180) angle -= 360;
        if (angle > maxAngle) angle = maxAngle;
        if (angle < -maxAngle) angle = -maxAngle;
        return axis*angle;
    }
}
