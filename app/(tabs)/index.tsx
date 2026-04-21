import { Accelerometer, Gyroscope } from 'expo-sensors';
import React, { useEffect, useRef, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import LABELS from '../../assets/labels.json';

const WINDOW_SIZE = 100;
const ACCEL_TO_MS2 = 9.80665;
const GYRO_FIRST = false;

export default function HomeScreen() {
  const [prediction, setPrediction] = useState<string>('Waiting for movement...');
  const [isReady, setIsReady] = useState<boolean>(false);
  const [accel, setAccel] = useState<{ x: number; y: number; z: number }>({ x: 0, y: 0, z: 0 });
  const [gyro, setGyro] = useState<{ x: number; y: number; z: number }>({ x: 0, y: 0, z: 0 });
  const [accelCount, setAccelCount] = useState<number>(0);
  const [gyroCount, setGyroCount] = useState<number>(0);
  const [inferenceCount, setInferenceCount] = useState<number>(0);
  const [lastOutput, setLastOutput] = useState<string>('—');
  const [modelSpec, setModelSpec] = useState<string>('(loading)');

  const accelBuffer = useRef<{ x: number; y: number; z: number }[]>([]);
  const gyroBuffer = useRef<{ x: number; y: number; z: number }[]>([]);
  const modelRef = useRef<TensorflowModel | null>(null);
  const inferenceN = useRef<number>(0);

  useEffect(() => {
    let accelSub: { remove: () => void } | null = null;
    let gyroSub: { remove: () => void } | null = null;

    async function setup() {
      try {
        console.log('[setup] requesting sensor permissions');
        const accelPerm = await Accelerometer.requestPermissionsAsync();
        const gyroPerm = await Gyroscope.requestPermissionsAsync();
        console.log('[setup] accel permission:', accelPerm.status, 'gyro permission:', gyroPerm.status);

        const accelAvailable = await Accelerometer.isAvailableAsync();
        const gyroAvailable = await Gyroscope.isAvailableAsync();
        console.log('[setup] accel available:', accelAvailable, 'gyro available:', gyroAvailable);

        console.log('[setup] loading tflite model');
        const model = await loadTensorflowModel(require('../../assets/model.tflite'), []);
        modelRef.current = model;
        setIsReady(true);
        console.log('[setup] model loaded');
        try {
          const insStr = JSON.stringify(model.inputs);
          const outsStr = JSON.stringify(model.outputs);
          console.log('[setup] model inputs:', insStr);
          console.log('[setup] model outputs:', outsStr);
          const inSummary = model.inputs.map(t => `${t.dataType}${JSON.stringify(t.shape)}`).join(' | ');
          const outSummary = model.outputs.map(t => `${t.dataType}${JSON.stringify(t.shape)}`).join(' | ');
          setModelSpec(`in: ${inSummary}\nout: ${outSummary}`);
        } catch (e) {
          console.log('[setup] could not stringify model i/o:', e);
          setModelSpec('spec-error');
        }

        Accelerometer.setUpdateInterval(20);
        Gyroscope.setUpdateInterval(20);

        let accelN = 0;
        let gyroN = 0;

        accelSub = Accelerometer.addListener(data => {
          accelN += 1;
          if (accelN % 25 === 0) {
            console.log(`[accel #${accelN}] x=${data.x.toFixed(3)} y=${data.y.toFixed(3)} z=${data.z.toFixed(3)}`);
            setAccelCount(accelN);
            setAccel({ x: data.x, y: data.y, z: data.z });
          }
          const mag = Math.sqrt(data.x * data.x + data.y * data.y + data.z * data.z);
          if (mag > 1.5) setPrediction(`Movement (a=${mag.toFixed(2)})`);

          accelBuffer.current.push({ x: data.x, y: data.y, z: data.z });
          if (accelBuffer.current.length > WINDOW_SIZE) accelBuffer.current.shift();

          if (
            accelBuffer.current.length === WINDOW_SIZE &&
            gyroBuffer.current.length === WINDOW_SIZE &&
            accelN % WINDOW_SIZE === 0 &&
            modelRef.current
          ) {
            const flat = new Float32Array(WINDOW_SIZE * 6);
            let aMin = Infinity, aMax = -Infinity, gMin = Infinity, gMax = -Infinity;
            for (let i = 0; i < WINDOW_SIZE; i++) {
              const a = accelBuffer.current[i];
              const g = gyroBuffer.current[i];
              const ax = a.x * ACCEL_TO_MS2, ay = a.y * ACCEL_TO_MS2, az = a.z * ACCEL_TO_MS2;
              if (GYRO_FIRST) {
                flat[i * 6 + 0] = g.x; flat[i * 6 + 1] = g.y; flat[i * 6 + 2] = g.z;
                flat[i * 6 + 3] = ax;  flat[i * 6 + 4] = ay;  flat[i * 6 + 5] = az;
              } else {
                flat[i * 6 + 0] = ax;  flat[i * 6 + 1] = ay;  flat[i * 6 + 2] = az;
                flat[i * 6 + 3] = g.x; flat[i * 6 + 4] = g.y; flat[i * 6 + 5] = g.z;
              }
              aMin = Math.min(aMin, ax, ay, az); aMax = Math.max(aMax, ax, ay, az);
              gMin = Math.min(gMin, g.x, g.y, g.z); gMax = Math.max(gMax, g.x, g.y, g.z);
            }
            console.log(`[input] accel range [${aMin.toFixed(2)}, ${aMax.toFixed(2)}]  gyro range [${gMin.toFixed(2)}, ${gMax.toFixed(2)}]  layout=${GYRO_FIRST ? 'gyro-first' : 'accel-first'}`);
            try {
              inferenceN.current += 1;
              const n = inferenceN.current;
              const outBuffers = modelRef.current.runSync([flat.buffer as ArrayBuffer]);
              const firstBuf = outBuffers[0];
              const outSpec = modelRef.current.outputs[0];
              const dtype = outSpec?.dataType ?? 'float32';
              let view: ArrayLike<number>;
              switch (dtype) {
                case 'float32': view = new Float32Array(firstBuf); break;
                case 'int32':   view = new Int32Array(firstBuf); break;
                case 'int8':    view = new Int8Array(firstBuf); break;
                case 'uint8':   view = new Uint8Array(firstBuf); break;
                case 'int16':   view = new Int16Array(firstBuf); break;
                case 'float64': view = new Float64Array(firstBuf); break;
                default:        view = new Float32Array(firstBuf); break;
              }
              const len = view.length;
              const vals = Array.from({ length: len }, (_, i) => Number(view[i]));
              const anyNaN = vals.some(v => Number.isNaN(v));
              if (anyNaN) {
                console.log(`[infer #${n}] NaN in output, dtype=${dtype} len=${len}`);
                setLastOutput('NaN');
              } else {
                const maxV = Math.max(...vals);
                const exps = vals.map(v => Math.exp(v - maxV));
                const sum = exps.reduce((a, b) => a + b, 0);
                const probs = exps.map(e => e / sum);
                const top3 = probs
                  .map((p, i) => ({ p, i }))
                  .sort((a, b) => b.p - a.p)
                  .slice(0, 3);
                const labelOf = (i: number) => LABELS[i] ?? `#${i}`;
                const summary = top3.map(t => `${labelOf(t.i)}: ${(t.p * 100).toFixed(1)}%`).join('\n');
                console.log(`[infer #${n}] ${top3.map(t => `${labelOf(t.i)}:${(t.p * 100).toFixed(1)}%`).join(' ')}`);
                setLastOutput(summary);
                setPrediction(`${labelOf(top3[0].i)} (${(top3[0].p * 100).toFixed(1)}%)`);
              }
              setInferenceCount(n);
            } catch (err) {
              console.error('[infer] failed:', err);
              setLastOutput(`error: ${String(err).slice(0, 60)}`);
            }
          }
        });

        gyroSub = Gyroscope.addListener(data => {
          gyroN += 1;
          gyroBuffer.current.push({ x: data.x, y: data.y, z: data.z });
          if (gyroBuffer.current.length > WINDOW_SIZE) gyroBuffer.current.shift();
          if (gyroN % 25 === 0) {
            console.log(`[gyro #${gyroN}] x=${data.x.toFixed(3)} y=${data.y.toFixed(3)} z=${data.z.toFixed(3)}`);
            setGyroCount(gyroN);
            setGyro({ x: data.x, y: data.y, z: data.z });
          }
        });

        console.log('[setup] listeners attached');
      } catch (error) {
        console.error('[setup] failed:', error);
        setPrediction('Error loading brain');
      }
    }
    setup();

    return () => {
      accelSub?.remove();
      gyroSub?.remove();
    };
  }, []);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>HIT Tracker LIVE</Text>
      <View style={styles.displayBox}>
        <Text style={styles.predictionText}>
          {isReady ? prediction : "Loading AI Brain..."}
        </Text>
        <Text style={styles.sensorText}>
          accel #{accelCount}{"\n"}
          x={accel.x.toFixed(3)}  y={accel.y.toFixed(3)}  z={accel.z.toFixed(3)}
        </Text>
        <Text style={styles.sensorText}>
          gyro #{gyroCount}{"\n"}
          x={gyro.x.toFixed(3)}  y={gyro.y.toFixed(3)}  z={gyro.z.toFixed(3)}
        </Text>
        <Text style={styles.sensorText}>
          infer #{inferenceCount}{"\n"}
          out=[{lastOutput}]
        </Text>
        <Text style={styles.sensorText}>
          {modelSpec}
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
  predictionText: { color: '#00ff00', fontSize: 24, textAlign: 'center', fontWeight: 'bold', marginBottom: 16 },
  sensorText: { color: '#00ff00', fontSize: 14, textAlign: 'center', fontFamily: 'Courier', marginTop: 8 },
  footer: { color: '#444', marginTop: 30, textTransform: 'uppercase', letterSpacing: 2 }
});