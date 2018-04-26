# unity-tensorflow-samples

Unity project that loads pretrained tensorflow pb model files and use them to predict

# Important

* In order to run the samples, your unity must be at least version 2017.3.0f3 and must be 64 bit
* The pre-trained tensorflow *.pb graph models in keras_codes should be trained using tensorflow 1.2 as the version of "Unity TensorFlow Plugin" that I was using has tensorflow 1.2 only.
* If your graph model is named cifar10.pb, it should be changed to cifar10.bytes and put under "Resources" folder (which is under Asset folder)
* If your graph model has the full path "Asset/Resources/tf_models/cifar10.bytes", you should call Resources.load<TextAsset>("tf_models/cifar10") to load it. 


# Usage

Below is the sample code on how to use the Cifar10ImageClassifier to classify images stored in the "Resources/images/cifar10" folder:

```cs 
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
```

# Note:

* Cifar10ImageClassifier takes image having width = 32, height = 32, channels = 3
* In order for the Cifar10ImageClassifier to read the images in the "Resources/images/cifar10" folder, you make make the images readable in their Import Settings.




