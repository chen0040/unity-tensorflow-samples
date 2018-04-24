using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TensorFlow;

public class Cifar10ImageClassifier {

    private TFGraph graph;
    private TFSession session;

    private static string[] labels = new string[] {
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
    };

    public void LoadModel(string your_name_graph)
    {
        if (graph == null)
        {
            Debug.Log("Loading tensor graph " + your_name_graph);
            TextAsset graphModel = Resources.Load<TextAsset>(your_name_graph);

            if(graphModel == null)
            {
                Debug.LogError("Failed to load tensor graph " + your_name_graph);
            }

            graph = new TFGraph();
            graph.Import(graphModel.bytes);
            session = new TFSession(graph);
        }
    }

    public int PredictClass(Texture2D image)
    {
        Debug.Log(image);


        float[] imageBytes = new float[image.width * image.height * 3];

        int idx = 0;
        for (int i = 0; i < image.width; i++)
        {
            for (int j = 0; j < image.height; j++)
            {
                Color pixel = image.GetPixel(i, j);

                imageBytes[idx++] = pixel.r / 255.0f;
                imageBytes[idx++] = pixel.g / 255.0f;
                imageBytes[idx++] = pixel.b / 255.0f;
            }
        }

        
        var runner = session.GetRunner();
        runner
            .AddInput(graph["conv2d_1_input"][0], imageBytes)
            .AddInput(graph["dropout_1/keras_learning_phase"][0], new TFTensor(0));

        runner.Fetch(graph["output_node0"][0]);
        float[,] recurrent_tensor = runner.Run()[0].GetValue() as float[,];

        float maxVal = float.MinValue;
        int bestIdx = -1;
        for(int i=0; i < recurrent_tensor.GetUpperBound(1); ++i)
        {
            float val = recurrent_tensor[0, i];
            if(val > maxVal)
            {
                maxVal = val;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    public string PredictLabel(Texture2D image)
    {
        int classId = PredictClass(image);
        if(classId >= 0 && classId < labels.Length)
        {
            return labels[classId];
        }
        return "unknown";
    }
}
