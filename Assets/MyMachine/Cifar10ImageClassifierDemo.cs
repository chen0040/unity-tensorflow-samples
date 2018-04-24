using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TensorFlow;

public class Cifar10ImageClassifierDemo : MonoBehaviour {

    private Cifar10ImageClassifier classifier = new Cifar10ImageClassifier();

	// Use this for initialization
	void Start () {
        Debug.Log(TensorFlow.TFCore.Version);
        classifier.LoadModel("tf_models/cnn_cifar10");
	}
	
	// Update is called once per frame
	void Update () {
		
	}
}
