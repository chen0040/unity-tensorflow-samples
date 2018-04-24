# unity-tensorflow-samples

Unity project that loads pretrained tensorflow pb model files and use them to predict

# Important

* In order to run the samples, your unity must be at least version 2017.3.0f3 and must be 64 bit
* Your tensorflow *.pb graph models should be trained using tensorflow 1.2 as the version of "Unity TensorFlow Plugin" that I was using has tensorflow 1.2 only.
* If your graph model is named cifar10.pb, it should be changed to cifar10.bytes and put under "Resources" folder (which is under Asset folder)
* If your graph model has the full path "Asset/Resources/tf_models/cifar10.bytes", you should call Resources.load<TextAsset>("tf_models/cifar10") to load it. 




