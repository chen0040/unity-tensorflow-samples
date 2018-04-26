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

        string[] image_names = new string[9];
        int index = 0;
        for(int i=1; i <= 3; ++i)
        {
            image_names[index++] = "cat" + i;
            image_names[index++] = "bird" + i;
            image_names[index++] = "automobile" + i;
        }
        foreach(string image_name in image_names)
        {
            Debug.Log(image_name);
            Texture2D img = Resources.Load<Texture2D>("images/cifar10/" + image_name);
            Debug.Log("Predicted: " + classifier.PredictLabel(img));
        }
        
	}
	
	// Update is called once per frame
	void Update () {
		
	}
}
