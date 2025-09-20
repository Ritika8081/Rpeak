"use client";
import React, { useEffect, useRef, useState } from "react";
import { Bluetooth, Eye, EyeOff, Activity, Zap, BarChart3, TrendingUp, Play, Square, Clock } from "lucide-react";
import { WebglPlot, WebglLine, ColorRGBA } from "webgl-plot";
import { BPMCalculator, filterQRS } from '../lib/bpmCalculator';
import { HighpassFilter, NotchFilter, LowpassFilter } from "../lib/filters";
import { HRVCalculator } from '../lib/hrvCalculator';
import { PQRSTDetector, PQRSTPoint } from '../lib/pqrstDetector';
import { PanTompkinsDetector } from '../lib/panTompkinsDetector';
import { ECGIntervalCalculator, ECGIntervals } from '../lib/ecgIntervals';
import * as tf from "@tensorflow/tfjs";
import { checkModelExists } from '../lib/modelTester';
import { zscoreNorm } from "../lib/modelTrainer";
import SessionRecording, { PatientInfo, RecordingSession } from './SessionRecording';
import { SessionAnalyzer, SessionAnalysisResults } from '../lib/sessionAnalyzer';
import SessionReport from './SessionReport';
import { AAMI_CLASSES } from "../lib/modelTrainer";

const SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b";
const DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8";
const CONTROL_CHAR_UUID = "0000ff01-0000-1000-8000-00805f9b34fb";

const NUM_POINTS = 1000; // Updated: 1000 points ≈ 2.78 seconds at 360Hz
const SAMPLE_RATE = 360; // 360Hz sampling rate
const MODEL_INPUT_LENGTH = 135; // 135 samples ≈ 375ms at 360Hz
const SINGLE_SAMPLE_LEN = 7;
const NEW_PACKET_LEN = 7 * 10;
const BATCH_SIZE = 20;

export default function EcgFullPanel() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [connected, setConnected] = useState(false);
    const [startTime, setStartTime] = useState<number | null>(null);
    const [bpmDisplay, setBpmDisplay] = useState("-- BPM");
    const [peaksVisible, setPeaksVisible] = useState(true);
    const [timer, setTimer] = useState("00:00");
    const [showHRV, setShowHRV] = useState(false);
    const [classLabels, setClassLabels] = useState<string[]>(AAMI_CLASSES);
    const [showPQRST, setShowPQRST] = useState(false);
    const [showIntervals, setShowIntervals] = useState(false);
    const [signalQuality, setSignalQuality] = useState<'good' | 'poor' | 'no-signal'>('no-signal');

    // Add these states to your component
    const [isRecording, setIsRecording] = useState(false);
    const [recordingStartTime, setRecordingStartTime] = useState<number | null>(null);
    const [recordingTime, setRecordingTime] = useState("00:00");
    const [recordedData, setRecordedData] = useState<number[]>([]);
    const [currentSession, setCurrentSession] = useState<RecordingSession | null>(null);
    const [sessionResults, setSessionResults] = useState<SessionAnalysisResults | null>(null);
    const [showSessionReport, setShowSessionReport] = useState(false);
    const sessionAnalyzer = useRef(new SessionAnalyzer(SAMPLE_RATE));

    // Update this state for physiological state
    const [physioState, setPhysioState] = useState<{ state: string; confidence: number }>({
        state: "Analyzing",
        confidence: 0
    });

    type HRVMetrics = {
        sampleCount: number;
        assessment: {
            color: string;
            status: string;
            description: string;
        };
        rmssd: number;
        sdnn: number;
        pnn50: number;
        triangularIndex: number;
        lfhf: {
            lf: number;
            hf: number;
            ratio: number;
        };
        // Add any other fields returned by getAllMetrics()
    };

    const [hrvMetrics, setHrvMetrics] = useState<HRVMetrics | null>(null);
    const [ecgIntervals, setEcgIntervals] = useState<ECGIntervals | null>(null);
    const [gender, setGender] = useState<'male' | 'female'>('male');

    const [modelLoaded, setModelLoaded] = useState(false);
    const [ecgModel, setEcgModel] = useState<tf.LayersModel | null>(null);
    const [modelPrediction, setModelPrediction] = useState<{
        prediction: string;
        confidence: number;
    } | null>(null);

    // Auto Analyze state and toggle function
    const [autoAnalyze, setAutoAnalyze] = useState(false);


    useEffect(() => {
        if (typeof window !== "undefined") {
            const labels = JSON.parse(localStorage.getItem('ecg-class-labels') || 'null');
            setClassLabels(Array.isArray(labels) && labels.length > 0 ? labels : AAMI_CLASSES);
        }
    }, []);

    // Effect to run analyzeCurrent automatically if autoAnalyze is enabled
    useEffect(() => {
        if (!autoAnalyze) return;
        if (!modelLoaded || !ecgIntervals) return;
        const interval = setInterval(() => {
            analyzeCurrent();
        }, 60000); // 1 minute refresh
        return () => clearInterval(interval);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [autoAnalyze, modelLoaded, ecgIntervals]);

    const wglpRef = useRef<WebglPlot | null>(null);
    const lineRef = useRef<WebglLine | null>(null);
    const peakLineRef = useRef<WebglLine | null>(null);
    const dataCh0 = useRef(new Array(NUM_POINTS).fill(0));
    const peakData = useRef(new Array(NUM_POINTS).fill(0));
    const sampleIndex = useRef(0);
    const highpass = useRef(new HighpassFilter()); // Updated filter for 360Hz
    const notch = useRef(new NotchFilter()); // Updated filter for 360Hz
    const ecg = useRef(new LowpassFilter()); // Updated filter for 360Hz
    const bpmCalculator = useRef(new BPMCalculator(SAMPLE_RATE, 5, 40, 200));
    const hrvCalculator = useRef(new HRVCalculator());
    const pqrstDetector = useRef(new PQRSTDetector(SAMPLE_RATE));
    const pqrstPoints = useRef<PQRSTPoint[]>([]);
    const pLineRef = useRef<WebglLine | null>(null);
    const qLineRef = useRef<WebglLine | null>(null);
    const rLineRef = useRef<WebglLine | null>(null);
    const sLineRef = useRef<WebglLine | null>(null);
    const tLineRef = useRef<WebglLine | null>(null);
    const panTompkins = useRef(new PanTompkinsDetector(SAMPLE_RATE));
    const intervalCalculator = useRef(new ECGIntervalCalculator(SAMPLE_RATE));

    // Add this state to store currently visible PQRST points
    const [visiblePQRST, setVisiblePQRST] = useState<PQRSTPoint[]>([]);

    // Add this type definition with your other types
    type STSegmentData = {
        deviation: number;
        status: 'normal' | 'elevation' | 'depression';
    };

    // Add this state inside your component
    const [stSegmentData, setSTSegmentData] = useState<STSegmentData | null>(null);

    // Add to EcgFullPanel component
    const [beatPredictions, setBeatPredictions] = useState<{ prediction: string, confidence: number }[]>([]);
    const [beatBuffer, setBeatBuffer] = useState<{ prediction: string, confidence: number }[]>([]);
    const [batchResult, setBatchResult] = useState<{
        summary: ReturnType<typeof getRollingSummary> | null,
        latest: { prediction: string, confidence: number } | null
    } | null>(null);


    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = canvas.clientWidth * dpr;
        canvas.height = canvas.clientHeight * dpr;

        const wglp = new WebglPlot(canvas);

        // Create ECG line (main signal) - updated for 1000 points
        const line = new WebglLine(new ColorRGBA(0, 1, 0.2, 1), NUM_POINTS);
        line.arrangeX();

        // Create PQRST lines - updated for 1000 points
        const pLine = new WebglLine(new ColorRGBA(1, 0.7, 0, 1), NUM_POINTS); // Orange for P
        pLine.arrangeX();

        const qLine = new WebglLine(new ColorRGBA(0.2, 0.6, 1, 1), NUM_POINTS); // Blue for Q
        qLine.arrangeX();

        const rLine = new WebglLine(new ColorRGBA(1, 0, 0, 1), NUM_POINTS); // Red for R
        rLine.arrangeX();

        const sLine = new WebglLine(new ColorRGBA(0, 0.8, 1, 1), NUM_POINTS); // Cyan for S
        sLine.arrangeX();

        const tLine = new WebglLine(new ColorRGBA(0.8, 0.3, 1, 1), NUM_POINTS); // Purple for T
        tLine.arrangeX();

        // Add all lines to the plot
        wglp.addLine(line);
        wglp.addLine(pLine);
        wglp.addLine(qLine);
        wglp.addLine(rLine);
        wglp.addLine(sLine);
        wglp.addLine(tLine);

        // Store references
        wglpRef.current = wglp;
        lineRef.current = line;
        pLineRef.current = pLine;
        qLineRef.current = qLine;
        rLineRef.current = rLine;
        sLineRef.current = sLine;
        tLineRef.current = tLine;

        const render = () => {
            requestAnimationFrame(render);
            const scale = getScaleFactor();
            for (let i = 0; i < NUM_POINTS; i++) {
                line.setY(i, dataCh0.current[i] * scale);

                // Update PQRST lines if visible
                if (showPQRST) {
                    pLine.setY(i, pLineRef.current?.getY(i) || 0);
                    qLine.setY(i, qLineRef.current?.getY(i) || 0);
                    rLine.setY(i, rLineRef.current?.getY(i) || 0);
                    sLine.setY(i, sLineRef.current?.getY(i) || 0);
                    tLine.setY(i, tLineRef.current?.getY(i) || 0);
                } else {
                    pLine.setY(i, 0);
                    qLine.setY(i, 0);
                    rLine.setY(i, 0);
                    sLine.setY(i, 0);
                    tLine.setY(i, 0);
                }
            }
            wglp.update();
        };
        render();
    }, [peaksVisible, showPQRST]);

    function getScaleFactor() {
        const maxAbs = Math.max(...dataCh0.current.map(Math.abs), 0.1);
        return maxAbs > 0.9 ? 0.9 / maxAbs : 1;
    }

    // Add this function to see what data we have
    function getRollingSummary(buffer?: { prediction: string, confidence: number }[]) {
        const arr = buffer ?? beatBuffer;
        if (arr.length === 0) return null;
        const counts: Record<string, number> = {};
        arr.forEach(bp => {
            counts[bp.prediction] = (counts[bp.prediction] || 0) + 1;
        });
        const total = arr.length;
        const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
        const majorityClass = sorted[0][0];
        const majorityPercent = (sorted[0][1] / total) * 100;
        return {
            majorityClass,
            majorityPercent,
            counts,
            total
        };
    }

    function updatePeaks() {
        // Add debug for signal diagnostics
        const maxAbs = Math.max(...dataCh0.current.map(Math.abs));
        const mean = dataCh0.current.reduce((sum, val) => sum + val, 0) / dataCh0.current.length;
        const variance = dataCh0.current.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / dataCh0.current.length;

        // Skip peak detection if the signal is too weak or too flat
        if (maxAbs < 0.05 || variance < 0.0002) {

            pqrstPoints.current = [];
            if (showPQRST) {
                setVisiblePQRST([]);
            }
            return;
        }

        // Use Pan-Tompkins algorithm for peak detection
        const peaks = panTompkins.current.detectQRS(dataCh0.current);

        // Fall back to original algorithm if Pan-Tompkins doesn't find peaks
        let usedPanTompkins = peaks.length > 0;

        if (!usedPanTompkins) {

            const originalPeaks = bpmCalculator.current.detectPeaks(dataCh0.current);
            if (originalPeaks.length > 0) {
                peaks.push(...originalPeaks);
            }
        }


        // Generate visualization (same as before)
        peakData.current = bpmCalculator.current.generatePeakVisualization(dataCh0.current, peaks);


        // Try to detect PQRST waves
        let pqrstDetected = false;

        if (peaks.length >= 1) {
            // Existing PQRST detection with peaks
            pqrstPoints.current = pqrstDetector.current.detectWaves(dataCh0.current, peaks);
            pqrstDetected = pqrstPoints.current.length > 0;

            if (showPQRST) {
                setVisiblePQRST([...pqrstPoints.current]);
            }
        }

        // If standard detection failed, try direct detection
        if (!pqrstDetected) {

            pqrstPoints.current = pqrstDetector.current.detectDirectWaves(dataCh0.current, sampleIndex.current);

            if (showPQRST && pqrstPoints.current.length > 0) {
                setVisiblePQRST([...pqrstPoints.current]);
            } else {
                setVisiblePQRST([]);
            }
        }


        // Extract RR intervals for HRV analysis
        if (peaks.length >= 2) {

            hrvCalculator.current.extractRRFromPeaks(peaks, SAMPLE_RATE);

            // Force update HRV metrics
            const metrics = hrvCalculator.current.getAllMetrics();

            setHrvMetrics(metrics);
        } else {
            console.log('Not enough peaks for HRV analysis');
        }

        // Calculate ECG intervals when PQRST points are available
        if (pqrstPoints.current.length > 0) {
            const intervals = intervalCalculator.current.calculateIntervals(pqrstPoints.current);
            if (intervals) {
                // Add ST segment analysis
                const stAnalysis = analyzeSTSegment(pqrstPoints.current);
                if (stAnalysis) {
                    setSTSegmentData(stAnalysis);
                    // Add ST data to intervals object if your ECGIntervals type supports it
                    // If your ECGIntervals type doesn't have st fields, you can modify it or use the separate state
                }
                setEcgIntervals(intervals);
                return; // <-- Only return if intervals are set
            }
        }

        // Fallback: If enough R-peaks, estimate BPM directly
        if (peaks.length >= 2) {
            const rrIntervals = [];
            for (let i = 1; i < peaks.length; i++) {
                rrIntervals.push((peaks[i] - peaks[i - 1]) / SAMPLE_RATE * 1000);
            }
            const avgRR = rrIntervals.reduce((a, b) => a + b, 0) / rrIntervals.length;
            const bpm = avgRR > 0 ? 60000 / avgRR : 0;
            setEcgIntervals({
                rr: avgRR,
                pr: 0,
                qrs: 0,
                qt: 0,
                qtc: 0,
                bpm,
                status: {
                    rr: avgRR < 600 ? 'short' : avgRR > 1000 ? 'long' : 'normal',
                    pr: 'unknown',
                    qrs: 'unknown',
                    qt: 'unknown',
                    qtc: 'unknown',
                    bpm: bpm < 60 ? 'bradycardia' : bpm > 100 ? 'tachycardia' : 'normal'
                }
            });
        } else {
            setEcgIntervals(null);
        }
    }

    // Add an effect to clear visible points when toggling off
    useEffect(() => {
        if (!showPQRST) {
            setVisiblePQRST([]);
        } else if (pqrstPoints.current.length > 0) {
            setVisiblePQRST([...pqrstPoints.current]);
        }
    }, [showPQRST]);

    useEffect(() => {
        const timerInterval = setInterval(() => {
            if (startTime) {
                const elapsed = Math.floor((Date.now() - startTime) / 1000); // Define elapsed here
                const min = String(Math.floor(elapsed / 60)).padStart(2, "0");
                const sec = String(elapsed % 60).padStart(2, "0");
                setTimer(`${min}:${sec}`);
            }

            // Fix: Use the improved peak detection
            if (connected) {
                // Get peaks using the most reliable method
                const peaks = panTompkins.current.detectQRS(dataCh0.current);

                // If Pan-Tompkins fails, use backup method
                let finalPeaks = peaks;
                if (peaks.length === 0) {
                    finalPeaks = bpmCalculator.current.detectPeaks(dataCh0.current);
                }

                // Calculate BPM from peaks
                const bpm = bpmCalculator.current.calculateBPMFromPeaks(finalPeaks);

                if (bpm && bpm >= 40 && bpm <= 200) {
                    const smoothedBPM = bpmCalculator.current.smoothBPM(bpm);
                    setBpmDisplay(Math.round(smoothedBPM) + " BPM");
                } else {
                    setBpmDisplay("-- BPM");
                }
            }

            // Always try to update HRV metrics when connected
            if (connected) {
                const metrics = hrvCalculator.current.getAllMetrics();

                if (metrics.sampleCount > 0) {
                    setHrvMetrics(metrics);
                    // Update this line to use the new method name
                    setPhysioState(hrvCalculator.current.getPhysiologicalState());
                }
            }
        }, 1000);
        return () => clearInterval(timerInterval);
    }, [startTime, connected]);

    // Add effect to set gender
    useEffect(() => {
        intervalCalculator.current.setGender(gender);
    }, [gender]);

    // Add this useEffect to load the model when the component mounts
    useEffect(() => {
        async function loadModel() {
            try {
                // Always try to load the model directly
                const basePath = window.location.pathname.startsWith('/ECG_Monitor') ? '/ECG_Monitor/' : '/';
                const model = await tf.loadLayersModel(`${basePath}models/beat-level-ecg-model.json`);
                setEcgModel(model);
                setModelLoaded(true);
                console.log('ECG model loaded successfully');
            } catch (err) {
                setModelLoaded(false);
                setEcgModel(null);
                console.error('Failed to load model:', err);
            }
        }
        loadModel();
    }, []);

    async function connect() {
        try {
            // Check if navigator.bluetooth is available
            if (!('bluetooth' in navigator)) {
                alert("Web Bluetooth API is not supported in this browser.");
                return;
            }
            const device = await (navigator as any).bluetooth.requestDevice({
                filters: [{ namePrefix: "NPG" }],
                optionalServices: [SERVICE_UUID]
            });
            const server = await device.gatt?.connect();
            const service = await server?.getPrimaryService(SERVICE_UUID);
            const controlChar = await service?.getCharacteristic(CONTROL_CHAR_UUID);
            const dataChar = await service?.getCharacteristic(DATA_CHAR_UUID);

            await controlChar?.writeValue(new TextEncoder().encode("START"));
            await dataChar?.startNotifications();

            dataChar?.addEventListener("characteristicvaluechanged", (event: any) => {
                const value = event.target.value;
                if (value.byteLength === NEW_PACKET_LEN) {
                    for (let i = 0; i < NEW_PACKET_LEN; i += SINGLE_SAMPLE_LEN) {
                        const view = new DataView(value.buffer.slice(i, i + SINGLE_SAMPLE_LEN));
                        const raw = view.getInt16(1, false);
                        const norm = (raw - 2048) / 2048;

                        // Apply high-pass, then notch, then bandpass
                        let filtered = highpass.current.process(norm);
                        filtered = notch.current.process(filtered);
                        filtered = ecg.current.process(filtered);

                        if (!isFinite(filtered) || isNaN(filtered)) filtered = 0;
                        filtered = Math.max(-1, Math.min(1, filtered));

                        // Store and use filtered value
                        dataCh0.current[sampleIndex.current] = filtered;
                        sampleIndex.current = (sampleIndex.current + 1) % NUM_POINTS;
                    }

                    // Call updatePeaks to refresh the PQRST points with each new data packet
                    updatePeaks();
                }
            });

            setConnected(true);
            setStartTime(Date.now());
            bpmCalculator.current.reset();
            hrvCalculator.current.reset();
            panTompkins.current.reset(); // Reset Pan-Tompkins detector
            intervalCalculator.current.reset(); // Reset interval calculator

        } catch (e) {
            console.error("BLE Connection failed:", e);
        }
    }

    // Replace your existing adaptSignalForModel function with this enhanced version:

    const adaptSignalForModel = (ecgWindow: number[]): number[] => {
        // Step 1: Convert your normalized signal back to MIT-BIH-like scale
        // Your signal: -1 to +1 → Convert to MIT-BIH: ~500-1500 range
        const mitBihLikeSignal = ecgWindow.map(x => {
            // Scale from [-1, +1] to MIT-BIH range [0, 2048] with 1024 baseline
            return (x * 400) + 1024;  // Assumes typical ±400 unit variation
        });

        // Step 2: Detect R-peak in the window
        const centerIdx = Math.floor(ecgWindow.length / 2);
        const searchRange = 30;

        let maxIdx = centerIdx;
        let maxValue = mitBihLikeSignal[centerIdx];

        for (let i = Math.max(0, centerIdx - searchRange);
            i < Math.min(ecgWindow.length, centerIdx + searchRange);
            i++) {
            if (Math.abs(mitBihLikeSignal[i] - 1024) > Math.abs(maxValue - 1024)) {
                maxValue = mitBihLikeSignal[i];
                maxIdx = i;
            }
        }

        // Step 3: Apply MIT-BIH-style polarity correction
        let needsFlip = false;

        // In MIT-BIH, R-peaks are typically positive deflections above 1024
        if (maxValue < 1024) {
            needsFlip = true;

        }

        const polarityCorrectedSignal = needsFlip ?
            mitBihLikeSignal.map(x => 2048 - x) :  // Flip around 1024 baseline
            mitBihLikeSignal;

        // Step 4: Apply Z-score normalization (same as training)
        const mean = polarityCorrectedSignal.reduce((a, b) => a + b, 0) / polarityCorrectedSignal.length;
        const std = Math.sqrt(polarityCorrectedSignal.reduce((a, b) => a + (b - mean) ** 2, 0) / polarityCorrectedSignal.length);

        if (std < 10) {  // Minimum std in MIT-BIH units

            return new Array(ecgWindow.length).fill(0);
        }

        const normalizedSignal = polarityCorrectedSignal.map(x => (x - mean) / std);



        return normalizedSignal;
    };

    // Now replace your existing analyzeCurrent function with this updated version:
    // Update the R-R interval filtering to be less strict
    const analyzeCurrent = async () => {

        if (!ecgModel) {

            setModelPrediction({ prediction: "Analyzing", confidence: 0 });
            return;
        }

        // Convert signal quality check to MIT-BIH-like scale
        const mitBihLikeData = dataCh0.current.map(x => (x * 400) + 1024);
        const maxAbs = Math.max(...mitBihLikeData.map(x => Math.abs(x - 1024)));
        const variance = mitBihLikeData.reduce((sum, val) => sum + Math.pow(val - 1024, 2), 0) / mitBihLikeData.length;



        // MIT-BIH-style quality thresholds
        if (maxAbs < 50 || variance < 100) {  // 50 units ≈ 244 μV, reasonable for ECG

            setModelPrediction({ prediction: "Poor Signal", confidence: 0 });
            return;
        }

        // Use the same peak detection pipeline as your main application
        const detectedPeaks = panTompkins.current.detectQRS(dataCh0.current);

        // If Pan-Tompkins fails, use backup method
        const recentPeaks = detectedPeaks.length > 0
            ? detectedPeaks
            : bpmCalculator.current.detectPeaks(dataCh0.current);

        // RELAXED: More forgiving filtering for consumer ECG devices
        const filteredPeaks = recentPeaks.filter((peak, index) => {
            if (index === 0) return true;
            const timeDiff = (peak - recentPeaks[index - 1]) / SAMPLE_RATE * 1000;
            // Expanded range: 300-1500ms (40-200 BPM range) - your suggestion
            return timeDiff >= 300 && timeDiff <= 1500;
        });

        if (filteredPeaks.length === 0) {

            setModelPrediction({ prediction: "No Valid Beats", confidence: 0 });
            return;
        }



        // Get the most recent R-peak
        const latestRPeak = filteredPeaks[filteredPeaks.length - 1];
        const halfBeat = Math.floor(MODEL_INPUT_LENGTH / 2); // 67 samples



        // FIXED: Better circular buffer handling
        let ecgWindow: number[] = [];
        const startIdx = latestRPeak - halfBeat;

        // Create a properly ordered window
        for (let i = 0; i < MODEL_INPUT_LENGTH; i++) {
            const actualIdx = (startIdx + i + NUM_POINTS) % NUM_POINTS;
            ecgWindow.push(dataCh0.current[actualIdx]);
        }


        // CRITICAL: Adapt signal to match training data characteristics
        const adaptedSignal = adaptSignalForModel(ecgWindow);

        // Now apply z-score normalization to adapted signal
        const windowMean = adaptedSignal.reduce((a, b) => a + b, 0) / adaptedSignal.length;
        const windowStd = Math.sqrt(adaptedSignal.reduce((a, b) => a + (b - windowMean) ** 2, 0) / adaptedSignal.length);

        // RELAXED variance threshold for consumer devices
        if (windowStd < 0.005) {  // Reduced from 0.01

            setModelPrediction({ prediction: "Flat Signal", confidence: 0 });
            return;
        }

        const normWindow = adaptedSignal.map(x => (x - windowMean) / windowStd);


        // Validate normalized data
        const normMean = normWindow.reduce((a, b) => a + b, 0) / normWindow.length;
        const normStd = Math.sqrt(normWindow.reduce((a, b) => a + (b - normMean) ** 2, 0) / normWindow.length);


        // More lenient validation for consumer devices
        if (Math.abs(normMean) > 0.3) {  // Increased from 0.2

            setModelPrediction({ prediction: "Normalization Failed", confidence: 0 });
            return;
        }

        // Create input tensor with correct shape [1, 135, 1]
        const inputTensor = tf.tensor3d([normWindow.map((v: number) => [v])], [1, MODEL_INPUT_LENGTH, 1]);


        try {
            const outputTensor = ecgModel.predict(inputTensor) as tf.Tensor;
            const probabilities = await outputTensor.data();



            if (!probabilities || probabilities.length === 0) {
                console.error("Model output is empty or invalid");
                setModelPrediction({ prediction: "Model Error", confidence: 0 });
                inputTensor.dispose();
                return;
            }

            const predArray = Array.from(probabilities);


            const deviceBiasCorrection = [
                1.4,  // Normal: moderate boost (reduced from 1.8)
                0.9,  // Supraventricular: mild reduction (increased from 0.7)
                1.0,  // Ventricular: no change
                0.8,  // Fusion: mild reduction (increased from 0.6)
                0.7   // Other: mild reduction (increased from 0.5)
            ];

            const correctedProbs = predArray.map((prob, idx) => prob * deviceBiasCorrection[idx]);
            const correctedSum = correctedProbs.reduce((a, b) => a + b, 0);
            const normalizedProbs = correctedProbs.map(p => p / correctedSum);


            const maxIndex = normalizedProbs.indexOf(Math.max(...normalizedProbs));
            const confidence = normalizedProbs[maxIndex] * 100;


            // Slightly reduced confidence threshold
            if (confidence < 40) {  // Reduced from 45

                setModelPrediction({ prediction: "Uncertain", confidence });
                inputTensor.dispose();
                outputTensor.dispose();
                return;
            }

            if (maxIndex < 0 || maxIndex >= classLabels.length) {

                setModelPrediction({ prediction: "Classification Error", confidence: 0 });
                inputTensor.dispose();
                outputTensor.dispose();
                return;
            }

            const predictedClass = classLabels[maxIndex];

            setModelPrediction({
                prediction: predictedClass,
                confidence: confidence
            });

            // RAISED confidence floor for rolling analysis as you suggested
            if (confidence >= 60) {  // REDUCED from 70 to 60
                setBeatPredictions(prev => {
                    const updated = [...prev, { prediction: predictedClass, confidence }];
                    const rolling = updated.slice(-BATCH_SIZE);

                    if (rolling.length > 0) {
                        const summary = getRollingSummary(rolling);
                        setBatchResult({
                            summary,
                            latest: rolling[rolling.length - 1]
                        });

                    }
                    return rolling;
                });
            } else {

            }

            inputTensor.dispose();
            outputTensor.dispose();
        } catch (err) {

            setModelPrediction({ prediction: "Prediction Error", confidence: 0 });
            inputTensor.dispose();
        }
    };

    // Add this to keep PQRST labels moving with the wave
    useEffect(() => {
        if (!showPQRST) return;

        // Update the PQRST labels position when the data is refreshed
        const pqrstUpdateInterval = setInterval(() => {
            if (pqrstPoints.current.length > 0 && showPQRST) {
                setVisiblePQRST([...pqrstPoints.current]);
            }
        }, 200); // Update at 5fps for smoother movement

        return () => clearInterval(pqrstUpdateInterval);
    }, [showPQRST]);

    // Add this effect to update signal quality
    useEffect(() => {
        const signalQualityInterval = setInterval(() => {
            if (!connected) {
                setSignalQuality('no-signal');
                return;
            }

            // Calculate signal quality metrics
            const maxAbs = Math.max(...dataCh0.current.map(Math.abs));
            const variance = dataCh0.current.reduce((sum, val) => sum + Math.pow(val, 2), 0) / dataCh0.current.length;

            if (maxAbs < 0.1 || variance < 0.001) {
                setSignalQuality('no-signal');
            } else if (maxAbs < 0.3 || variance < 0.01) {
                setSignalQuality('poor');
            } else {
                setSignalQuality('good');
            }
        }, 1000);

        return () => clearInterval(signalQualityInterval);
    }, [connected]);

    // Add this before the return statement in your EcgFullPanel component
    const renderAbnormalityWarnings = (intervals: ECGIntervals) => {
        const warnings = [];

        // QRS wide: check AI prediction for R/L
        if (intervals.status.qrs === 'wide') {
            if (modelPrediction?.prediction === "R") {
                warnings.push({ text: 'Right bundle branch block (RBBB)', color: 'text-red-400' });
            } else if (modelPrediction?.prediction === "L") {
                warnings.push({ text: 'Left bundle branch block (LBBB)', color: 'text-red-400' });
            } else {
                warnings.push({ text: 'Possible bundle branch block', color: 'text-red-400' });
            }
        }

        if (intervals.status.pr === 'long')
            warnings.push({ text: 'Possible 1st degree AV block', color: 'text-red-400' });

        if (intervals.status.qtc === 'prolonged')
            warnings.push({ text: 'QT prolongation - arrhythmia risk', color: 'text-red-400' });

        if (warnings.length === 0) return null;

        return (
            <div className="mt-4 p-3 rounded-lg border border-red-500/30 bg-red-500/10">
                <h4 className="text-sm font-medium text-red-400 mb-2">Potential Findings:</h4>
                <ul className="space-y-1 text-sm">
                    {warnings.map((warning, i) => (
                        <li key={i} className={`flex items-center gap-2 ${warning.color}`}>
                            <span>•</span>
                            <span>{warning.text}</span>
                        </li>
                    ))}
                </ul>
            </div>
        );
    };

    // Add this function inside your EcgFullPanel component
    const analyzeSTSegment = (pqrstPoints: PQRSTPoint[]): STSegmentData | null => {
        // Find relevant points
        const sPoint = pqrstPoints.find(p => p.type === 'S');
        const tPoint = pqrstPoints.find(p => p.type === 'T');
        const qPoint = pqrstPoints.find(p => p.type === 'Q');

        if (!sPoint || !tPoint || !qPoint) {
            return null;
        }

        // Find J-point (end of S-wave)
        const jPointIndex = sPoint.index;

        // Get ST segment point (80ms after J-point)
        const stPointIndex = jPointIndex + Math.floor(0.08 * SAMPLE_RATE);

        // Get baseline as PR segment level (or use isoelectric line)
        const baseline = qPoint.amplitude;

        // Find ST point value (interpolate if needed)
        let stValue;
        const stPoint = pqrstPoints.find(p => p.index === stPointIndex);
        if (stPoint) {
            stValue = stPoint.amplitude;
        } else {
            // Interpolate between S and T if exact point not available
            const ratio = (stPointIndex - sPoint.index) / (tPoint.index - sPoint.index);
            stValue = sPoint.amplitude + ratio * (tPoint.amplitude - sPoint.amplitude);
        }

        // Calculate ST deviation in mm (1mm = 0.1mV in standard ECG)
        const deviation = (stValue - baseline) * 10;

        // Determine status
        let status: 'normal' | 'elevation' | 'depression' = 'normal';
        if (deviation >= 0.2) status = 'elevation';
        else if (deviation <= -0.1) status = 'depression';

        return { deviation, status };
    };

    // Add this function inside your EcgFullPanel component
    const generateSummaryReport = () => {
        if (!ecgIntervals) {
            alert("No ECG data available for report generation");
            return;
        }

        // Create CSV content
        let csvContent = "data:text/csv;charset=utf-8,";
        csvContent += "ECG Summary Report\n";
        csvContent += `Generated on,${new Date().toLocaleString()}\n`;
        csvContent += `Sampling Rate,${SAMPLE_RATE} Hz\n`;
        csvContent += `Model Input Length,${MODEL_INPUT_LENGTH} samples\n\n`;

        // Add patient info
        csvContent += "Patient Information\n";
        csvContent += `Gender,${gender === 'male' ? 'Male' : 'Female'}\n\n`;

        // Add heart rate and rhythm information
        csvContent += "Vital Signs\n";
        csvContent += `Heart Rate,${ecgIntervals.bpm.toFixed(1)} BPM\n`;
        csvContent += `Heart Rate Status,${ecgIntervals.status.bpm}\n\n`;

        // Add ECG intervals
        csvContent += "ECG Intervals\n";
        csvContent += `RR Interval,${ecgIntervals.rr.toFixed(0)} ms\n`;
        csvContent += `PR Interval,${ecgIntervals.pr.toFixed(0)} ms\n`;
        csvContent += `QRS Duration,${ecgIntervals.qrs.toFixed(0)} ms\n`;
        csvContent += `QT Interval,${ecgIntervals.qt ? ecgIntervals.qt.toFixed(0) : "N/A"} ms\n`;
        csvContent += `QTc Interval,${ecgIntervals.qtc.toFixed(0)} ms\n`;

        // Add ST segment data if available
        if (stSegmentData) {
            csvContent += `ST Deviation,${stSegmentData.deviation.toFixed(2)} mm\n`;
            csvContent += `ST Status,${stSegmentData.status}\n`;
        }

        csvContent += "\nInterval Status\n";
        csvContent += `PR Status,${ecgIntervals.status.pr}\n`;
        csvContent += `QRS Status,${ecgIntervals.status.qrs}\n`;
        csvContent += `QTc Status,${ecgIntervals.status.qtc}\n`;

        // Add HRV metrics if available
        if (hrvMetrics && hrvMetrics.sampleCount > 0) {
            csvContent += "\nHeart Rate Variability Analysis\n";
            csvContent += `RMSSD,${hrvMetrics.rmssd.toFixed(1)} ms\n`;
            csvContent += `SDNN,${hrvMetrics.sdnn.toFixed(1)} ms\n`;
            csvContent += `pNN50,${hrvMetrics.pnn50.toFixed(1)}%\n`;
            csvContent += `Triangular Index,${hrvMetrics.triangularIndex.toFixed(1)}\n`;
            csvContent += `LF/HF Ratio,${hrvMetrics.lfhf.ratio.toFixed(2)}\n`;

            // Add physiological state
            if (physioState) {
                csvContent += `Physiological State,${physioState.state}\n`;
                csvContent += `State Confidence,${(physioState.confidence * 100).toFixed(0)}%\n`;
            }
        }

        csvContent += "\nPotential Findings\n";

        // Add abnormalities
        const findings = [];
        if (ecgIntervals.status.bpm === 'bradycardia') findings.push("Bradycardia (slow heart rate)");
        if (ecgIntervals.status.bpm === 'tachycardia') findings.push("Tachycardia (fast heart rate)");
        if (ecgIntervals.status.pr === 'long') findings.push("Prolonged PR interval - Possible 1st degree AV block");
        if (ecgIntervals.status.qrs === 'wide') findings.push("Wide QRS complex - Possible bundle branch block");
        if (ecgIntervals.status.qtc === 'prolonged') findings.push("Prolonged QTc interval - Increased arrhythmia risk");
        if (stSegmentData?.status === 'elevation') findings.push("ST segment elevation - Possible myocardial injury");
        if (stSegmentData?.status === 'depression') findings.push("ST segment depression - Possible ischemia");

        if (findings.length > 0) {
            findings.forEach(finding => {
                csvContent += `${finding}\n`;
            });
        } else {
            csvContent += "No abnormalities detected\n";
        }

        csvContent += "\nDISCLAIMER: This is not a medical device. Do not use for diagnosis or treatment decisions.\n";
        csvContent += "Analysis is based on a limited dataset and should be confirmed by a qualified healthcare professional.\n";

        // Create download link
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `ecg-report-${new Date().toISOString().slice(0, 10)}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const [showAIAnalysis, setShowAIAnalysis] = useState(false); // Add this state to control AI Analysis panel visibility

    // Effect to run analyzeCurrent automatically when panel is visible
    useEffect(() => {
        if (!showAIAnalysis) return;
        if (!modelLoaded || !connected) return; // Changed from ecgIntervals to connected


        // Run initial analysis immediately
        analyzeCurrent();

        // Set up auto-refresh every 1 minute (60,000 ms)
        const interval = setInterval(() => {
            analyzeCurrent();
        }, 60000); // 1 minute refresh

        return () => {
            clearInterval(interval);
        };
    }, [showAIAnalysis, modelLoaded, connected]); // Removed ecgIntervals dependency


    // Initialize the session analyzer with model
    useEffect(() => {
        const loadModel = async () => {
            await sessionAnalyzer.current.loadModel();
        };

        loadModel();
    }, []);

    // Add this effect to update recording time
    useEffect(() => {
        if (!isRecording || !recordingStartTime) return;

        const timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
            const min = String(Math.floor(elapsed / 60)).padStart(2, "0");
            const sec = String(elapsed % 60).padStart(2, "0");
            setRecordingTime(`${min}:${sec}`);
        }, 1000);

        return () => clearInterval(timerInterval);
    }, [isRecording, recordingStartTime]);

    // Add these functions to handle recording
    const startRecording = (patientInfo: PatientInfo) => {
        setIsRecording(true);
        setRecordingStartTime(Date.now());
        setRecordedData([]);

        // Create a new session
        setCurrentSession({
            id: Date.now().toString(),
            startTime: Date.now(),
            endTime: null,
            duration: 0,
            patientInfo,
            ecgData: [],
            sampleRate: SAMPLE_RATE,
            rPeaks: [],
            pqrstPoints: []
        });
    };

    const stopRecording = () => {
        if (!isRecording || !currentSession || !recordingStartTime) {

            return null;
        }

        const endTime = Date.now();
        const duration = (endTime - recordingStartTime) / 1000;

        const freshRPeaks = panTompkins.current.detectQRS(recordedData);
        const freshPQRST = pqrstDetector.current.detectWaves(recordedData, freshRPeaks, 0);



        const freshIntervals = intervalCalculator.current.calculateIntervals(freshPQRST);

        if (!freshIntervals) {

        }

        const updatedSession: RecordingSession = {
            ...currentSession,
            endTime,
            duration,
            ecgData: [...recordedData],
            rPeaks: freshRPeaks,
            pqrstPoints: freshPQRST,
            intervals: freshIntervals || null
        };

        setCurrentSession(updatedSession);
        setIsRecording(false);

        analyzeSession(updatedSession);

        return updatedSession;
    };

    // Add this at the beginning of your analyzeSession function
    const analyzeSession = async (session: RecordingSession) => {
        try {
            // Only use the real, data-driven analysis
            const results = await sessionAnalyzer.current.analyzeSession(session);
            setSessionResults(results);
            setShowSessionReport(true);
        } catch (err) {
            console.error('Session analysis failed:', err);
        }
    };

    const saveSessionReport = () => {
        if (!sessionResults || !currentSession) return;

        // Create CSV content
        let csvContent = "data:text/csv;charset=utf-8,";
        csvContent += "ECG Summary Report\n";
        csvContent += `Generated on,${new Date().toLocaleString()}\n\n`;

        // Add patient info
        csvContent += "Patient Information\n";
        csvContent += `Age,${currentSession.patientInfo.age}\n`;
        csvContent += `Gender,${currentSession.patientInfo.gender === 'male' ? 'Male' : 'Female'}\n`;
        csvContent += `Weight,${currentSession.patientInfo.weight} kg\n`;
        csvContent += `Height,${currentSession.patientInfo.height} cm\n\n`;

        // Add summary
        csvContent += "Summary\n";
        csvContent += `Recording Duration,${sessionResults.summary.recordingDuration}\n`;
        csvContent += `Average Heart Rate,${sessionResults.summary.heartRate.average.toFixed(1)} BPM\n`;
        csvContent += `Heart Rate Range,${sessionResults.summary.heartRate.min.toFixed(0)}-${sessionResults.summary.heartRate.max.toFixed(0)} BPM\n`;
        csvContent += `ECG Classification,${sessionResults.aiClassification.prediction}\n`;
        csvContent += `Classification Confidence,${sessionResults.aiClassification.confidence.toFixed(1)}%\n\n`;

        // Add more sections for intervals, HRV, etc.

        // Create download link
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `ecg-session-report-${new Date().toISOString().slice(0, 10)}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    // Modify your data processing to record data
    useEffect(() => {
        // Existing data processing code...

        // Add this to record data when in recording mode
        if (isRecording) {
            // Take a copy of the last N samples that came in
            const newData = dataCh0.current.slice(
                Math.max(0, sampleIndex.current - 10),
                Math.min(NUM_POINTS, sampleIndex.current)
            );

            // If we wrapped around, also get the data from the end
            if (sampleIndex.current < 10) {
                const endData = dataCh0.current.slice(NUM_POINTS - (10 - sampleIndex.current));
                newData.unshift(...endData);
            }

            // Add to recorded data
            setRecordedData(prev => [...prev, ...newData]);
        }
    }, [isRecording, sampleIndex.current]);

    // Patient Info modal state
    const [showPatientInfo, setShowPatientInfo] = useState(false);

    const predictionDetails: Record<string, { label: string, description: string, symbols: string[] }> = {
        "Normal": {
            label: "Normal beat",
            description: "A typical heartbeat with no detected abnormalities.",
            symbols: ['N', '.', 'L', 'R', 'e', 'j']
        },
        "Supraventricular": {
            label: "Supraventricular ectopic beat",
            description: "A heartbeat originating above the ventricles (e.g., atria).",
            symbols: ['A', 'a', 'J', 'S']
        },
        "Ventricular": {
            label: "Ventricular ectopic beat",
            description: "A heartbeat originating in the ventricles (lower chambers).",
            symbols: ['V', 'E', 'r']
        },
        "Fusion": {
            label: "Fusion beat",
            description: "A beat formed by the combination of normal and abnormal impulses.",
            symbols: ['F']
        },
        "Other": {
            label: "Other/unknown beat",
            description: "A beat that does not fit standard categories or is unclassified.",
            symbols: ['Q', '/', 'f', 'n', 'unknown']
        }
    };

    return (
        <div className="relative w-full h-full bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 ">
            {/* Patient Info Modal (overlay, not in sidebar) */}
            {showPatientInfo && (
                <SessionRecording
                    connected={connected}
                    onStartRecording={startRecording}
                    onStopRecording={stopRecording}
                    isRecording={isRecording}
                    recordingTime={recordingTime}
                    setShowPatientInfo={setShowPatientInfo}
                />
            )}
            {showSessionReport && sessionResults && (
                <SessionReport
                    analysisResults={sessionResults}
                    patientInfo={currentSession?.patientInfo ?? {
                        age: 0,
                        gender: 'male',
                        weight: 0,
                        height: 0,
                        medicalHistory: [],
                        medications: []
                    }}
                    sessionDate={new Date(currentSession?.startTime ?? Date.now())}
                    recordingTime={recordingTime}
                    onClose={() => setShowSessionReport(false)}
                    onSaveReport={saveSessionReport}
                />
            )}

            {/* Grid background */}
            <div className="absolute inset-0 opacity-10">
                <div className="h-full w-full bg-grid-pattern bg-[size:40px_40px]"></div>
            </div>

            {/* Main canvas */}
            <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full"
                style={{ background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)' }}
            />

            {/* Improved Fixed Sidebar */}
            <div className="fixed left-0 top-0 h-full z-60 flex items-center">
                <div
                    className="group h-full py-6 px-2 bg-black backdrop-blur border-r border-white/10 flex flex-col items-center justify-center transition-all duration-300 hover:w-[240px] w-16"
                >
                    {/* Connect Device Button */}
                    <div className="relative w-full mb-5">
                        <div className="flex">
                            <div className="w-16 flex justify-center">
                                <button
                                    onClick={connected ? undefined : connect}
                                    className={`w-10 h-10 flex items-center justify-center rounded-full transition-all ${connected
                                        ? 'bg-green-500/20 text-green-400 border border-green-500/30 cursor-not-allowed'
                                        : 'bg-blue-500/20 text-blue-400 border border-blue-500/30 hover:bg-blue-500/30'
                                        }`}
                                    title={connected ? 'Connected' : 'Connect Device'}
                                >
                                    <Bluetooth className="w-5 h-5" />
                                </button>
                            </div>
                            <div className="whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center">
                                <span className={`text-sm font-medium ${connected ? 'text-green-400' : 'text-blue-400'}`}>
                                    {connected ? 'Connected' : 'Connect Device'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Peaks Button */}
                    <div className="relative w-full mb-5">
                        <div className="flex">
                            <div className="w-16 flex justify-center">
                                <button
                                    onClick={() => setPeaksVisible(!peaksVisible)}
                                    className={`w-10 h-10 flex items-center justify-center rounded-full transition-all ${peaksVisible
                                        ? 'bg-green-500/20 text-green-400 border border-green-500/30 hover:bg-green-500/30'
                                        : 'bg-gray-500/20 text-gray-400 border border-gray-500/30 hover:bg-gray-500/30'
                                        }`}
                                    title={peaksVisible ? 'Hide Peaks' : 'Show Peaks'}
                                >
                                    {peaksVisible ? <Eye className="w-5 h-5" /> : <EyeOff className="w-5 h-5" />}
                                </button>
                            </div>
                            <div className="whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center">
                                <span className={`text-sm font-medium ${peaksVisible ? 'text-green-400' : 'text-gray-400'}`}>
                                    {peaksVisible ? 'Hide Peaks' : 'Show Peaks'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* PQRST Button */}
                    <div className="relative w-full mb-5">
                        <div className="flex">
                            <div className="w-16 flex justify-center">
                                <button
                                    onClick={() => setShowPQRST(!showPQRST)}
                                    className={`w-10 h-10 flex items-center justify-center rounded-full transition-all ${showPQRST
                                        ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30 hover:bg-orange-500/30'
                                        : 'bg-gray-500/20 text-gray-400 border border-gray-500/30 hover:bg-gray-500/30'
                                        }`}
                                    title={showPQRST ? 'Hide PQRST' : 'Show PQRST'}
                                >
                                    <Activity className="w-5 h-5" />
                                </button>
                            </div>
                            <div className="whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center">
                                <span className={`text-sm font-medium ${showPQRST ? 'text-orange-400' : 'text-gray-400'}`}>
                                    {showPQRST ? 'Hide PQRST' : 'Show PQRST'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* HRV Button */}
                    <div className="relative w-full mb-5">
                        <div className="flex">
                            <div className="w-16 flex justify-center">
                                <button
                                    onClick={() => setShowHRV(!showHRV)}
                                    className={`w-10 h-10 flex items-center justify-center rounded-full transition-all ${showHRV
                                        ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30 hover:bg-purple-500/30'
                                        : 'bg-gray-500/20 text-gray-400 border border-gray-500/30 hover:bg-gray-500/30'
                                        }`}
                                    title={showHRV ? 'Hide HRV' : 'Show HRV'}
                                >
                                    <TrendingUp className="w-5 h-5" />
                                </button>
                            </div>
                            <div className="whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center">
                                <span className={`text-sm font-medium ${showHRV ? 'text-purple-400' : 'text-gray-400'}`}>
                                    {showHRV ? 'Hide HRV' : 'Show HRV'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Intervals Button */}
                    <div className="relative w-full mb-5">
                        <div className="flex">
                            <div className="w-16 flex justify-center">
                                <button
                                    onClick={() => setShowIntervals(!showIntervals)}
                                    className={`w-10 h-10 flex items-center justify-center rounded-full transition-all ${showIntervals
                                        ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30 hover:bg-blue-500/30'
                                        : 'bg-gray-500/20 text-gray-400 border border-gray-500/30 hover:bg-gray-500/30'
                                        }`}
                                    title={showIntervals ? 'Hide Intervals' : 'Show Intervals'}
                                >
                                    <Activity className="w-5 h-5" />
                                </button>
                            </div>
                            <div className="whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center">
                                <span className={`text-sm font-medium ${showIntervals ? 'text-blue-400' : 'text-gray-400'}`}>
                                    {showIntervals ? 'Hide Intervals' : 'Show Intervals'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Start/Stop Recording Button Group in Sidebar */}
                    <div className="relative w-full mb-5">
                        <div className="flex">
                            <div className="w-16 flex justify-center">
                                {!isRecording ? (
                                    <button
                                        onClick={() => {
                                            console.log("Button clicked, connected:", connected);
                                            if (connected) {
                                                setShowPatientInfo(true);
                                                console.log("setShowPatientInfo called");
                                            } else {
                                                console.log("Device not connected");
                                            }
                                        }}
                                        disabled={!connected}
                                        className={`w-10 h-10 flex items-center justify-center rounded-full transition-all shadow-md
            ${connected
                                                ? 'bg-green-500/20 text-green-400 border border-green-500/30 hover:bg-green-500/30'
                                                : 'bg-gray-500/20 text-gray-400 border border-gray-700 cursor-not-allowed'
                                            }`}
                                        title={connected ? "Start Recording" : "Connect device to record"}
                                    >
                                        <Play className="w-5 h-5" />
                                    </button>
                                ) : (
                                    <button
                                        onClick={stopRecording}
                                        className="w-10 h-10 flex items-center justify-center rounded-full transition-all shadow-md
            bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30"
                                        title="Stop Recording"
                                    >
                                        <Square className="w-5 h-5" />
                                    </button>
                                )}
                            </div>
                            <div className="flex items-center gap-2 whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                                <span className={`text-sm font-medium ${!isRecording ? (connected ? 'text-green-400' : 'text-gray-400') : 'text-red-400'}`}>
                                    {!isRecording ? "Start Recording" : "Stop Recording"}
                                </span>

                            </div>
                        </div>
                    </div>


                    {/* AI Analysis Button */}
                    <div className="relative w-full mb-5">
                        <div className="flex">
                            <div className="w-16 flex justify-center">
                                <button
                                    onClick={() => {
                                        setShowAIAnalysis((prev) => {
                                            const next = !prev;
                                            // Always try to analyze if opening panel and model is loaded
                                            if (next && modelLoaded) {
                                                analyzeCurrent();
                                            }
                                            return next;
                                        });
                                    }}
                                    className={`w-10 h-10 flex items-center justify-center rounded-full transition-all ${showAIAnalysis
                                        ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 hover:bg-yellow-500/30'
                                        : 'bg-gray-500/20 text-gray-400 border border-gray-500/30 hover:bg-gray-500/30'
                                        }`}
                                    title={showAIAnalysis ? 'Hide AI Analysis' : 'Show AI Analysis'}
                                >
                                    <Zap className="w-5 h-5" />
                                </button>
                            </div>
                            <div className="whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center">
                                <span className={`text-sm font-medium ${showAIAnalysis ? 'text-yellow-400' : 'text-gray-400'}`}>
                                    {showAIAnalysis ? 'Hide AI Analysis' : 'Show AI Analysis'}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* HRV Panel */}
            {showHRV && (
                <div className="absolute left-20 top-1/2 transform -translate-y-1/2 w-80 bg-black/60 backdrop-blur-sm border border-white/20 rounded-xl p-4 text-white z-30">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-bold flex items-center gap-2">
                            <TrendingUp className="w-5 h-5 text-blue-400" />
                            HRV Analysis
                        </h3>
                        <button
                            onClick={() => setShowHRV(false)}
                            className="text-gray-400 hover:text-white"
                        >
                            ✕
                        </button>
                    </div>

                    {hrvMetrics && hrvMetrics.sampleCount > 0 ? (
                        <>
                            {/* Physiological State (previously Mental State) */}
                            <div className="mb-4 p-3 rounded-lg border border-white/20 bg-black/40">
                                <div className="flex items-center justify-between mb-1">
                                    <span className="text-sm text-gray-300">Physiological State:</span>
                                    <span className="font-bold text-lg" style={{
                                        color:
                                            physioState.state === "High Stress" ? "#ef4444" :
                                                physioState.state === "Relaxed" ? "#22c55e" :
                                                    physioState.state === "Focused" ? "#3b82f6" :
                                                        physioState.state === "Fatigue" ? "#f97316" :
                                                            physioState.state === "Analyzing" ? "#94a3b8" : "#94a3b8"
                                    }}>
                                        {physioState.state}
                                    </span>
                                </div>
                                <div className="w-full bg-gray-700 rounded-full h-1.5">
                                    <div
                                        className="h-1.5 rounded-full"
                                        style={{
                                            width: `${physioState.confidence * 100}%`,
                                            backgroundColor:
                                                physioState.state === "High Stress" ? "#ef4444" :
                                                    physioState.state === "Relaxed" ? "#22c55e" :
                                                        physioState.state === "Focused" ? "#3b82f6" :
                                                            physioState.state === "Fatigue" ? "#f97316" :
                                                                physioState.state === "Analyzing" ? "#94a3b8" : "#94a3b8"
                                        }}
                                    ></div>
                                </div>
                                <p className="text-xs text-gray-400 mt-1">
                                    Confidence: {(physioState.confidence * 100).toFixed(0)}%
                                </p>
                            </div>

                            {/* HRV Status */}
                            <div className="mb-4 p-3 rounded-lg border" style={{
                                backgroundColor: `${hrvMetrics.assessment.color}20`,
                                borderColor: `${hrvMetrics.assessment.color}40`
                            }}>
                                <div className="flex items-center justify-between">
                                    <span className="font-medium">Status:</span>
                                    <span className="font-bold" style={{ color: hrvMetrics.assessment.color }}>
                                        {hrvMetrics.assessment.status}
                                    </span>
                                </div>
                                <p className="text-sm text-gray-300 mt-1">
                                    {hrvMetrics.assessment.description}
                                </p>
                            </div>

                            {/* Time Domain Metrics */}
                            <div className="space-y-3">
                                <div className="flex justify-between">
                                    <span className="text-gray-300">RMSSD:</span>
                                    <span className="font-mono text-green-400">{hrvMetrics.rmssd.toFixed(1)} ms</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-300">SDNN:</span>
                                    <span className="font-mono text-blue-400">{hrvMetrics.sdnn.toFixed(1)} ms</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-300">pNN50:</span>
                                    <span className="font-mono text-yellow-400">{hrvMetrics.pnn50.toFixed(1)}%</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-300">Triangular:</span>
                                    <span className="font-mono text-purple-400">{hrvMetrics.triangularIndex.toFixed(1)}</span>
                                </div>
                            </div>

                            {/* Frequency Domain */}
                            <div className="mt-4 pt-4 border-t border-white/20">
                                <h4 className="text-sm font-medium text-gray-300 mb-2">Frequency Domain</h4>
                                <div className="space-y-2">
                                    <div className="flex justify-between">
                                        <span className="text-gray-400 text-sm">LF Power:</span>
                                        <span className="font-mono text-blue-400 text-sm">
                                            {hrvMetrics.lfhf.lf.toFixed(2)} ms²
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-400 text-sm">HF Power:</span>
                                        <span className="font-mono text-green-400 text-sm">
                                            {hrvMetrics.lfhf.hf.toFixed(2)} ms²
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-400 text-sm">LF/HF Ratio:</span>
                                        <span className="font-mono text-orange-400 text-sm">
                                            {hrvMetrics.lfhf.ratio.toFixed(2)}
                                            <span className="text-xs ml-1 text-gray-400">
                                                {hrvMetrics.lfhf.ratio > 2.0 ? '(Sympathetic ↑)' :
                                                    hrvMetrics.lfhf.ratio < 0.5 ? '(Parasympathetic ↑)' : '(Balanced)'}
                                            </span>
                                        </span>
                                    </div>
                                </div>
                            </div>


                        </>
                    ) : (
                        <div className="text-center text-gray-400 py-8">
                            <div className="animate-spin w-8 h-8 border-2 border-blue-400 border-t-transparent rounded-full mx-auto mb-4"></div>
                            <p>Collecting heart beats...</p>
                            <p className="text-sm mt-2">Need at least 2 peaks for analysis</p>
                            {connected && (
                                <p className="text-xs mt-2">
                                    Connected - waiting for ECG data...
                                </p>
                            )}
                        </div>
                    )}
                </div>
            )}


            {/* AI Prediction Results Panel */}
            {showAIAnalysis && (
                <div className="absolute right-4 top-[calc(40%+40px)] transform -translate-y-1/2 w-96 bg-black/60 backdrop-blur-sm border border-white/20 rounded-xl p-4 text-white z-40">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-bold flex items-center gap-2">
                            <Zap className="w-5 h-5 text-yellow-400" />
                            AI Beat Classification
                        </h3>
                        <button
                            onClick={() => setShowAIAnalysis(false)}
                            className="text-gray-400 hover:text-white"
                        >
                            ✕
                        </button>
                    </div>
                    {/* Development Phase Disclaimer */}
                    <div className="mb-4 p-3 rounded-lg border border-orange-500/30 bg-orange-500/10 text-orange-400">
                        <div className="flex items-start gap-2">
                            <span className="text-lg">🚧</span>
                            <div>
                                <div className="text-sm font-semibold mb-1">Development Phase Notice</div>
                                <div className="text-xs text-orange-300">
                                    The AI feature is currently in testing and development phase. Results may not be accurate or reliable.
                                    We will inform you as soon as this feature is fully operational and validated.
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Real-time Analysis Status */}
                    <div className="mb-3 p-2 rounded-lg border border-blue-500/30 bg-blue-500/10">
                        <div className="flex items-center justify-between text-sm">
                            <span className="text-blue-400">Analysis Status:</span>
                            <span className={`font-semibold ${modelLoaded && connected ? 'text-green-400' : 'text-yellow-400'}`}>
                                {modelLoaded && connected ? '🟢 Active (1 min refresh)' :
                                    modelLoaded ? '🟡 Ready (connect device)' : '🔴 Loading model...'}
                            </span>
                        </div>

                    </div>

                    {/* Current Beat Analysis */}
                    {modelPrediction && (
                        <div className="mb-4 p-3 rounded-lg border border-white/20 bg-black/40">
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm text-gray-300">Latest Beat:</span>
                                <span className={`font-bold text-lg ${modelPrediction.prediction === "Normal" ? 'text-green-400' :
                                    modelPrediction.prediction === "Uncertain" || modelPrediction.prediction === "Poor Signal" ? 'text-yellow-400' :
                                        modelPrediction.prediction.includes("Error") ? 'text-red-400' : 'text-orange-400'
                                    }`}>
                                    {modelPrediction.prediction}
                                </span>
                            </div>

                            {modelPrediction.confidence > 0 && (
                                <>
                                    <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
                                        <div
                                            className="h-2 rounded-full transition-all duration-300"
                                            style={{
                                                width: `${modelPrediction.confidence}%`,
                                                backgroundColor:
                                                    modelPrediction.confidence >= 80 ? '#22c55e' :
                                                        modelPrediction.confidence >= 60 ? '#eab308' : '#ef4444'
                                            }}
                                        ></div>
                                    </div>
                                    <div className="text-xs text-gray-400">
                                        Confidence: {modelPrediction.confidence.toFixed(1)}%
                                        {modelPrediction.confidence >= 70 && <span className="text-green-400 ml-2">✓ High</span>}
                                        {modelPrediction.confidence >= 45 && modelPrediction.confidence < 70 && <span className="text-yellow-400 ml-2">~ Medium</span>}
                                        {modelPrediction.confidence < 45 && <span className="text-red-400 ml-2">⚠ Low</span>}
                                    </div>
                                </>
                            )}
                        </div>
                    )}


                    {/* Enhanced disclaimer */}
                    <div className="mt-3 text-xs text-gray-300 italic border-t border-gray-700 pt-3">
                        <div className="mb-1">⚠️ <strong>Medical Disclaimer:</strong> This is not a diagnostic device.</div>
                        <div className="text-gray-400">
                            AI predictions are experimental and should not replace professional medical evaluation.
                        </div>
                    </div>
                </div>
            )}

            {/* ECG Intervals Panel */}
            {showIntervals && (
                <div className="absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[700px] 
          bg-black/60 backdrop-blur-sm border border-white/20 rounded-xl p-6 text-white z-10">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-bold flex items-center gap-2">
                            <Activity className="w-5 h-5 text-blue-400" />
                            Heart Signal Analysis
                        </h3>
                        <button
                            onClick={() => setShowIntervals(false)}
                            className="text-gray-400 hover:text-white"
                        >
                            ✕
                        </button>
                    </div>

                    <div className="flex gap-4">
                        {/* Left column - description and gender */}
                        <div className="w-1/3">
                            {/* Add a simple explanation */}
                            <p className="text-sm text-gray-300 mb-4">
                                This panel analyzes your heartbeat timing patterns. These measurements can reveal important information about heart health.
                            </p>

                            {/* Gender selector with explanation */}
                            <div className="mb-4">
                                <p className="text-sm text-gray-300 mb-2">
                                    Select your gender (affects normal ranges):
                                </p>
                                <div className="flex gap-2">
                                    <button
                                        onClick={() => setGender('male')}
                                        className={`flex-1 py-2 rounded-lg text-sm ${gender === 'male'
                                            ? 'bg-blue-500/30 border border-blue-500/60 text-blue-400'
                                            : 'bg-gray-800/50 border border-gray-700 text-gray-400'
                                            }`}
                                    >
                                        Male
                                    </button>
                                    <button
                                        onClick={() => setGender('female')}
                                        className={`flex-1 py-2 rounded-lg text-sm ${gender === 'female'
                                            ? 'bg-pink-500/30 border border-pink-500/60 text-pink-400'
                                            : 'bg-gray-800/50 border border-gray-700 text-gray-400'
                                            }`}
                                    >
                                        Female
                                    </button>
                                </div>
                            </div>

                            {/* Disclaimer */}
                            <div className="mt-auto pt-4 text-xs text-gray-500 italic">
                                This is not a medical device. Do not use for diagnosis or treatment decisions.
                            </div>
                            <div className="mt-4 text-xs text-gray-300 italic border-t border-gray-700 pt-3">

                                <div>Features are currently experimental and under development.</div>
                            </div>
                        </div>

                        {/* Right column - metrics */}
                        <div className="w-2/3">
                            {ecgIntervals ? (
                                <>
                                    {/* Heart Rate (BPM) - Full width */}
                                    <div className="p-3 rounded-lg border border-white/20 bg-black/40 mb-4">
                                        <div className="flex items-center justify-between">
                                            <span className="text-gray-300">Heart Rate:</span>
                                            <span className={`font-mono font-bold text-xl ${ecgIntervals?.status.bpm === 'normal' ? 'text-green-400' :
                                                ecgIntervals?.status.bpm === 'bradycardia' ? 'text-yellow-400' :
                                                    ecgIntervals?.status.bpm === 'tachycardia' ? 'text-red-400' : 'text-gray-400'
                                                }`}>
                                                {
                                                    ecgIntervals?.bpm > 0
                                                        ? ecgIntervals.bpm.toFixed(1)
                                                        : (() => {
                                                            // Use your actual R-peak indices array here
                                                            const rPeaks = pqrstPoints.current.filter(p => p.type === "R").map(p => p.index);
                                                            if (rPeaks && rPeaks.length >= 2) {
                                                                const rrIntervals = [];
                                                                for (let i = 1; i < rPeaks.length; i++) {
                                                                    const rr = (rPeaks[i] - rPeaks[i - 1]) / SAMPLE_RATE * 1000;
                                                                    if (rr >= 300 && rr <= 2000) rrIntervals.push(rr);
                                                                }
                                                                const avgRR = rrIntervals.length > 0
                                                                    ? rrIntervals.reduce((a, b) => a + b, 0) / rrIntervals.length
                                                                    : 0;
                                                                return avgRR > 0 ? (60000 / avgRR).toFixed(1) : "--";
                                                            }
                                                            return "--";
                                                        })()
                                                } BPM
                                            </span>
                                        </div>
                                        <div className="text-xs text-gray-400 mt-1">
                                            How many times your heart beats per minute. Normal is 60-100 BPM.
                                        </div>
                                    </div>

                                    {/* Two-column layout for metrics */}
                                    <div className="grid grid-cols-2 gap-3">
                                        {/* RR Interval with explanation */}
                                        <div className="p-3 rounded-lg border border-white/20 bg-black/40">
                                            <div className="flex justify-between items-center">
                                                <span className="text-gray-300 text-sm">Beat-to-Beat:</span>
                                                <span className={`font-mono ${ecgIntervals.status.rr === 'normal' ? 'text-green-400' :
                                                    ecgIntervals.status.rr === 'short' ? 'text-yellow-400' :
                                                        ecgIntervals.status.rr === 'long' ? 'text-blue-400' : 'text-gray-400'
                                                    }`}>
                                                    {ecgIntervals.rr.toFixed(0)} ms
                                                </span>
                                            </div>
                                            <div className="text-xs text-gray-400 mt-1">
                                                R-R interval: 600-1000ms normal
                                            </div>
                                        </div>

                                        {/* PR Interval with explanation */}
                                        <div className="p-3 rounded-lg border border-white/20 bg-black/40">
                                            <div className="flex justify-between items-center">
                                                <span className="text-gray-300 text-sm">Conduction:</span>
                                                <span className={`font-mono ${ecgIntervals.status.pr === 'normal' ? 'text-green-400' :
                                                    ecgIntervals.status.pr === 'short' ? 'text-yellow-400' :
                                                        ecgIntervals.status.pr === 'long' ? 'text-red-400' : 'text-gray-400'
                                                    }`}>
                                                    {ecgIntervals.pr.toFixed(0)} ms
                                                </span>
                                            </div>
                                            <div className="text-xs text-gray-400 mt-1">
                                                PR interval: atria to ventricles
                                            </div>
                                        </div>



                                        {/* QRS Duration with explanation */}
                                        <div className="p-3 rounded-lg border border-white/20 bg-black/40">
                                            <div className="flex justify-between items-center">
                                                <span className="text-gray-300 text-sm">Activation:</span>
                                                <span className={`font-mono ${ecgIntervals.status.qrs === 'normal' ? 'text-green-400' :
                                                    ecgIntervals.status.qrs === 'wide' ? 'text-red-400' : 'text-gray-400'
                                                    }`}>
                                                    {ecgIntervals.qrs.toFixed(0)} ms
                                                </span>
                                            </div>
                                            <div className="text-xs text-gray-400 mt-1">
                                                QRS duration: ventricular activation
                                            </div>
                                        </div>

                                        {/* ST Segment data - added section */}
                                        {stSegmentData && (
                                            <div className="p-3 rounded-lg border border-white/20 bg-black/40">
                                                <div className="flex justify-between items-center">
                                                    <span className="text-gray-300 text-sm">ST Segment:</span>
                                                    <span className={`font-mono ${stSegmentData.status === 'normal' ? 'text-green-400' :
                                                        stSegmentData.status === 'elevation' ? 'text-red-400' :
                                                            'text-yellow-400'
                                                        }`}>
                                                        {stSegmentData.deviation.toFixed(2)} mm
                                                    </span>
                                                </div>
                                                <div className="text-xs text-gray-400 mt-1">
                                                    {stSegmentData.status === 'normal' ? 'Normal ST segment' :
                                                        stSegmentData.status === 'elevation' ? 'ST elevation detected' :
                                                            'ST depression detected'}
                                                </div>
                                            </div>
                                        )}
                                    </div>



                                    {/* Abnormality indicators - full width */}
                                    {ecgIntervals.status.pr === 'long' || ecgIntervals.status.qrs === 'wide' || ecgIntervals.status.qtc === 'prolonged' ? (
                                        <div className="mt-4 p-3 rounded-lg border border-yellow-500/30 bg-yellow-500/10">
                                            <h4 className="text-sm font-medium text-yellow-400 mb-1">Patterns Detected:</h4>
                                            <ul className="space-y-1 text-xs">
                                                {ecgIntervals.status.pr === 'long' && (
                                                    <li className="flex items-center gap-1 text-yellow-400">
                                                        <span>•</span>
                                                        <span>Prolonged conduction time</span>
                                                    </li>
                                                )}
                                                {ecgIntervals.status.qrs === 'wide' && (
                                                    <li className="flex items-center gap-1 text-yellow-400">
                                                        <span>•</span>
                                                        <span>Wide QRS complex</span>
                                                    </li>
                                                )}

                                            </ul>
                                        </div>
                                    ) : (
                                        <div className="mt-4 p-3 rounded-lg border border-green-500/30 bg-green-500/10">
                                            <h4 className="text-sm font-medium text-green-400">All Timing Patterns Normal</h4>
                                        </div>
                                    )}

                                    <div className="mt-3 text-xs text-gray-400 text-center">
                                        Based on your most recent complete heartbeat
                                    </div>
                                </>
                            ) : (
                                <div className="text-center text-gray-400 py-10">
                                    <div className="animate-spin w-10 h-10 border-2 border-blue-400 border-t-transparent rounded-full mx-auto mb-4"></div>
                                    <p>Analyzing your heart signal...</p>
                                    <p className="text-sm mt-2">We need a complete heartbeat for analysis</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* PQRST text labels overlay - updated for 1000 points */}
            {showPQRST && (
                <div className="absolute inset-0 pointer-events-none">
                    {(() => {
                        const validRPeakIndices = visiblePQRST.filter(p => p.type === "R").map(p => p.index);
                        return visiblePQRST
                            .filter(point => {
                                if (point.type !== "R") return true;
                                return validRPeakIndices.includes(point.index);
                            })
                            .map((point, index) => {
                                // Show ALL points across the 1000-point window
                                const xPercent = (point.index / NUM_POINTS) * 100;
                                const yOffset = 50 - (point.amplitude * getScaleFactor() * 50);

                                let color;
                                switch (point.type) {
                                    case 'P': color = 'text-orange-400'; break;
                                    case 'Q': color = 'text-blue-400'; break;
                                    case 'R': color = 'text-red-500'; break;
                                    case 'S': color = 'text-cyan-400'; break;
                                    case 'T': color = 'text-purple-400'; break;
                                    default: color = 'text-white'; break;
                                }

                                return (
                                    <div
                                        key={`pqrst-${index}`}
                                        className={`absolute font-bold ${color}`}
                                        style={{
                                            left: `${xPercent}%`,
                                            top: `${yOffset}%`,
                                            transform: 'translate(-50%, -50%)',
                                            textShadow: '0 0 4px rgba(0,0,0,0.8)'
                                        }}
                                    >
                                        {point.type}
                                    </div>
                                );
                            });
                    })()}
                </div>
            )}

            {/* Recording indicator - new addition */}
            {isRecording && (
                <div className="fixed top-6 left-1/2 transform -translate-x-1/2 z-80 flex items-center px-4 py-2 rounded-full bg-red-900/80 border border-red-500/30 shadow-lg">
                    <Clock className="w-5 h-5 text-red-400 mr-2" />
                    <span className="font-mono text-lg text-red-400">{recordingTime}</span>
                    <span className="ml-1 text-xs text-red-400 font-semibold animate-pulse">Recording...</span>
                </div>
            )}
        </div>
    );
}
