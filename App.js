import React, { useEffect, useRef, useState } from 'react';
import { Dimensions, LogBox, Platform, StyleSheet, View } from 'react-native';
import { Camera } from 'expo-camera';
import Canvas from 'react-native-canvas';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs';
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';

const TensorCamera = cameraWithTensors(Camera);

LogBox.ignoreAllLogs(false);

const { width, height } = Dimensions.get('window');

export default function App() {
  const [model, setModel] = useState(null);
  let context = useRef(null);
  const canvas = useRef(null);

  function handleCameraStream(images) {
    const loop = async () => {
      try {
        const nextImageTensor = images.next().value;

        if (!model || !nextImageTensor) {
          return;
        }

        model
          .detect(nextImageTensor)
          .then((predictions) => {
            drawRectangle(predictions, nextImageTensor);
          })
          .catch((err) => {
            console.log(err);
          });

        requestAnimationFrame(loop);
      } catch (err) {
        console.log(err);
      }
    };
    loop();
  }

  function drawRectangle(predictions, nextImageTensor) {
    if (!context.current || !canvas.current) {
      console.log('no context or canvas');
      return;
    }

    //console.log(predictions);

    const scaleWidth = width / nextImageTensor.shape[1];
    const scaleHeight = height / nextImageTensor.shape[0];
    const flipHorizontal = true;

    context.current.clearRect(0, 0, width, height);

    for (const prediction of predictions) {
      const [x, y, width, height] = prediction.bbox;
      const boundingBoxX = flipHorizontal
        ? canvas.current.width - x * scaleWidth - width * scaleWidth
        : x * scaleWidth;
      const boundingBoxY = y * scaleHeight;

      context.current.strokeRect(
        boundingBoxX,
        boundingBoxY,
        width * scaleWidth,
        height * scaleHeight
      );

      context.current.fillText(
        prediction.class,
        boundingBoxX - 5,
        boundingBoxY - 5
      );
    }
  }

  const handleCanvas = async (can) => {
    if (can) {
      can.width = width;
      can.height = height;
      const ctx = can.getContext('2d');
      context.current = ctx;
      ctx.strokeStyle = 'red';
      ctx.fillStyle = 'red';
      ctx.lineWidth = 3;
      canvas.current = can;
    }
  };

  let textureDims;
  Platform.OS === 'ios'
    ? (textureDims = { height: 1920, width: 1080 })
    : (textureDims = { height: 1200, width: 1600 });

  useEffect(() => {
    (async () => {
      try {
        const { status } = await Camera.requestCameraPermissionsAsync();
        if (status !== 'granted') {
          console.error('Camera permission not granted');
          return;
        }
        console.log('Camera persmission granted.');

        await tf.ready();
        console.log('TensorFlow ready.');

        const loadedModel = await cocoSsd.load();
        setModel(loadedModel);
        console.log('Model loaded successfully.');
      } catch (error) {
        console.error('Error loading model:', error);
      }
    })();
  }, []);

  return (
    <View style={styles.container}>
      <TensorCamera
        style={styles.camera}
        type={Camera.Constants.Type.back}
        cameraTextureHeight={textureDims.height}
        cameraTextureWidth={textureDims.width}
        resizeHeight={200}
        resizeWidth={152}
        resizeDepth={3}
        onReady={handleCameraStream}
        autorender={true}
        useCustomShadersToResize={false}
      />
      <Canvas style={styles.canvas} ref={handleCanvas} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  camera: {
    width: '100%',
    height: '100%',
  },
  canvas: {
    position: 'absolute',
    zIndex: 10,
    width: '100%',
    height: '100%',
  },
});
