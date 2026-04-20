import { Accelerometer, Gyroscope } from 'expo-sensors';
import React, { useEffect, useRef, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';

export default function HomeScreen() {
  const [prediction, setPrediction] = useState<string>('Waiting for movement...');
  const [isReady, setIsReady] = useState<boolean>(false);
  
  // Refs for data and model
  const dataBuffer = useRef<number[][]>([]); 
  const modelRef = useRef<TensorflowModel | null>(null);

  useEffect(() => {
    async function setup() {
      try {
        // 1. Load the model from your assets folder
        const model = await loadTensorflowModel(require('../../assets/model.tflite'), []);
        modelRef.current = model;
        setIsReady(true);

        // 2. Set sensors to 50Hz (20ms interval)
        Accelerometer.setUpdateInterval(20);
        Gyroscope.setUpdateInterval(20);

        // 3. Listen to sensors
        const accelSub = Accelerometer.addListener(data => {
          // This is where we will feed the buffer
        });

        return () => {
          accelSub.remove();
        };
      } catch (error) {
        console.error("Model failed to load:", error);
        setPrediction("Error loading brain");
      }
    }
    setup();
  }, []);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>HIT Tracker LIVE</Text>
      <View style={styles.displayBox}>
        <Text style={styles.predictionText}>
          {isReady ? prediction : "Loading AI Brain..."}
        </Text>
      </View>
      <Text style={styles.footer}>Keep the intensity high.</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000', alignItems: 'center', justifyContent: 'center' },
  title: { color: '#fff', fontSize: 24, fontWeight: 'bold', marginBottom: 20 },
  displayBox: { width: '85%', height: 250, backgroundColor: '#111', borderRadius: 30, justifyContent: 'center', alignItems: 'center', borderWidth: 2, borderColor: '#00ff00' },
  predictionText: { color: '#00ff00', fontSize: 28, textAlign: 'center', fontWeight: 'bold' },
  footer: { color: '#444', marginTop: 30, textTransform: 'uppercase', letterSpacing: 2 }
});