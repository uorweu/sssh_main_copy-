#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def export():
    print("Loading YAMNet from TensorFlow Hub...")
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def yamnet_embeddings(waveform):
        scores, embeddings, spectrogram = yamnet_model(waveform)
        return scores, embeddings

    print("Tracing concrete function...")
    concrete_func = yamnet_embeddings.get_concrete_function()

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    output_path = "yamnet_with_embeddings.tflite"
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"\nSaved: {output_path} ({len(tflite_model) / 1024 / 1024:.1f} MB)")

if __name__ == "__main__":
    export()

