import { ECGIntervalCalculator } from './ecgIntervals';
import { HRVCalculator } from './hrvCalculator';
import { PQRSTDetector } from './pqrstDetector';
import { PanTompkinsDetector } from './panTompkinsDetector';
import { RecordingSession, PatientInfo } from '../components/SessionRecording';
import { AAMI_CLASSES, zscoreNorm } from './modelTrainer';
import * as tf from '@tensorflow/tfjs';

export type SessionAnalysisResults = {
    summary: {
        recordingDuration: string;
        recordingDurationSeconds?: number;
        rPeaks?: number[];
        heartRate: {
            average: number;
            min: number;
            max: number;
            status: string;
        };
        rhythm: {
            classification: string;
            confidence: number;
            irregularBeats: number;
            percentIrregular: number;
        };
    };
    intervals: {
        pr: {
            average: number;
            status: string;
        };
        qrs: {
            average: number;
            status: string;
        };
        qt: {
            average: number;
        };
        qtc: {
            average: number;
            status: string;
        };
        st: {
            deviation: number;
            status: string;
        };
    };
    hrv: {
        timeMetrics: {
            rmssd: number;
            sdnn: number;
            pnn50: number;
            triangularIndex: number;
        };
        frequencyMetrics: {
            lf: number;
            hf: number;
            lfhfRatio: number;
        };
        assessment: {
            status: string;
            description: string;
        };
        physiologicalState: {
            state: string;
            confidence: number;
        };
    };
    aiClassification: {
        prediction: string;
        confidence: number;
        explanation: string;
        beatClassifications?: {
            normal: number;
            supraventricular: number;
            ventricular: number;
            fusion: number;
            other: number;
        };
    };
    abnormalities: {
        type: string;
        severity: 'low' | 'medium' | 'high';
        description: string;
    }[];
    recommendations: string[];
};

export class SessionAnalyzer {
    private panTompkins: PanTompkinsDetector;
    private pqrstDetector: PQRSTDetector;
    private intervalCalculator: ECGIntervalCalculator;
    private hrvCalculator: HRVCalculator;
    private model: tf.LayersModel | null = null;
    private sampleRate: number;

    constructor(sampleRate: number) {
        this.sampleRate = sampleRate;
        this.panTompkins = new PanTompkinsDetector(sampleRate);
        this.pqrstDetector = new PQRSTDetector(sampleRate);
        this.intervalCalculator = new ECGIntervalCalculator(sampleRate);
        this.hrvCalculator = new HRVCalculator();
    }

    async loadModel(): Promise<boolean> {
        try {
            const modelSources = [
                'localstorage://beat-level-ecg-model',
                this.getModelPath(),
                'models/beat-level-ecg-model.json',
            ];

            for (const modelUrl of modelSources) {
                try {
                  
                    if (modelUrl.startsWith('localstorage://')) {
                        const models = await tf.io.listModels();
                        if (!models[modelUrl]) {
                          
                            continue;
                        }
                    }
                    this.model = await tf.loadLayersModel(modelUrl);
                   
                    return true;
                } catch (err) {
                  
                    continue;
                }
            }
            console.warn('No beat-level ECG model could be loaded from any source');
            return false;
        } catch (err) {
            console.error('Failed to load beat-level ECG model:', err);
            this.model = null;
            return false;
        }
    }

    /**
     * Returns the correct model path for both local and GitHub Pages deployments.
     */
    private getModelPath(): string {
        // If running on GitHub Pages, the path should include the repo name.
        if (typeof window !== "undefined") {
            const path = window.location.pathname;
            // Adjust 'ECG_Monitor' to your actual repo name if different
            if (path.startsWith('/ECG_Monitor')) {
                return '/ECG_Monitor/models/beat-level-ecg-model.json';
            }
        }
        // Default for local/dev
        return 'models/beat-level-ecg-model.json';
    }

    public async analyzeSession(session: RecordingSession): Promise<SessionAnalysisResults> {
        let intervals = session.intervals;

        if (!intervals) {
            console.warn("No interval data in session");
            if (session.pqrstPoints && session.pqrstPoints.length > 0) {
                const calculator = new ECGIntervalCalculator(session.sampleRate);
                session.intervals = calculator.calculateIntervals(session.pqrstPoints);
            } else {
                session.intervals = {
                    rr: 0,
                    bpm: 0,
                    pr: 0,
                    qrs: 0,
                    qt: 0,
                    qtc: 0,
                    status: {
                        rr: 'unknown',
                        bpm: 'unknown',
                        pr: 'unknown',
                        qrs: 'unknown',
                        qt: 'unknown',
                        qtc: 'unknown'
                    }
                };
            }
            intervals = session.intervals;
        }

        const { ecgData, patientInfo, sampleRate, duration } = session;

        this.intervalCalculator.setGender(patientInfo.gender);

        // 1. Detect R-peaks
        const peaks = this.panTompkins.detectQRS(ecgData);

        // 2. Detect PQRST waves
        const pqrstPoints = this.pqrstDetector.detectWaves(ecgData, peaks, 0);

        // 3. Calculate ECG intervals
        intervals = session.intervals || this.intervalCalculator.calculateIntervals(pqrstPoints);

        // 4. Calculate HRV metrics
        this.hrvCalculator.extractRRFromPeaks(peaks, sampleRate);
        const hrvMetrics = this.hrvCalculator.getAllMetrics();
        const physioState = this.hrvCalculator.getPhysiologicalState();

        // 5. Analyze ST segment
        const stSegmentData = this.analyzeSTSegment(pqrstPoints);

        // 6. Run AI classification using beat-level model
        const aiClassification = await this.runBeatLevelClassification(
            ecgData,
            peaks,
            intervals,
            stSegmentData,
            hrvMetrics,
            patientInfo
        );

        // 7. Determine abnormalities
        const abnormalities = this.detectAbnormalities(
            intervals,
            stSegmentData,
            hrvMetrics,
            aiClassification,
            patientInfo
        );

        // 8. Generate recommendations
        const recommendations = this.generateRecommendations(
            abnormalities,
            aiClassification,
            patientInfo
        );

        // 9. Calculate summary statistics
        const heartRates = this.calculateHeartRateStats(peaks, sampleRate, duration);

        return {
            summary: {
                recordingDuration: this.formatDuration(duration),
                recordingDurationSeconds: duration,
                rPeaks: peaks,
                heartRate: {
                    average: heartRates.average,
                    min: heartRates.min,
                    max: heartRates.max,
                    status: this.determineHeartRateStatus(heartRates.average)
                },
                rhythm: {
                    classification: aiClassification.prediction,
                    confidence: aiClassification.confidence,
                    irregularBeats: this.countIrregularBeats(peaks, sampleRate),
                    percentIrregular: this.calculatePercentIrregular(peaks, sampleRate)
                }
            },
            intervals: {
                pr: {
                    average: intervals?.pr || 0,
                    status: intervals?.status.pr || 'unknown'
                },
                qrs: {
                    average: intervals?.qrs || 0,
                    status: intervals?.status.qrs || 'unknown'
                },
                qt: {
                    average: intervals?.qt || 0
                },
                qtc: {
                    average: intervals?.qtc || 0,
                    status: intervals?.status.qtc || 'unknown'
                },
                st: {
                    deviation: stSegmentData?.deviation || 0,
                    status: stSegmentData?.status || 'unknown'
                }
            },
            hrv: {
                timeMetrics: {
                    rmssd: hrvMetrics.rmssd,
                    sdnn: hrvMetrics.sdnn,
                    pnn50: hrvMetrics.pnn50,
                    triangularIndex: hrvMetrics.triangularIndex
                },
                frequencyMetrics: {
                    lf: hrvMetrics.lfhf.lf,
                    hf: hrvMetrics.lfhf.hf,
                    lfhfRatio: hrvMetrics.lfhf.ratio
                },
                assessment: hrvMetrics.assessment,
                physiologicalState: {
                    state: physioState.state,
                    confidence: physioState.confidence
                }
            },
            aiClassification,
            abnormalities,
            recommendations
        };
    }

    private analyzeSTSegment(pqrstPoints: any[]): { deviation: number; status: string } | null {
        // Basic ST segment analysis
        if (!pqrstPoints || pqrstPoints.length === 0) {
            return null;
        }

        // Simplified ST analysis - in a real implementation, this would be more sophisticated
        const stDeviations: number[] = [];
        
        pqrstPoints.forEach(point => {
            if (point.S && point.T) {
                // Calculate ST segment deviation (simplified)
                const stDeviation = point.T.amplitude - point.S.amplitude;
                stDeviations.push(stDeviation);
            }
        });

        if (stDeviations.length === 0) {
            return { deviation: 0, status: 'unknown' };
        }

        const avgDeviation = stDeviations.reduce((sum, dev) => sum + dev, 0) / stDeviations.length;
        
        let status = 'normal';
        if (avgDeviation > 0.1) {
            status = 'elevation';
        } else if (avgDeviation < -0.1) {
            status = 'depression';
        }

        return {
            deviation: avgDeviation,
            status: status
        };
    }

    private async runBeatLevelClassification(
        ecgData: number[],
        peaks: number[],
        intervals: any,
        stSegmentData: any,
        hrvMetrics: any,
        patientInfo: PatientInfo
    ): Promise<{ 
        prediction: string; 
        confidence: number; 
        explanation: string;
        beatClassifications?: {
            normal: number;
            supraventricular: number;
            ventricular: number;
            fusion: number;
            other: number;
        };
    }> {
        if (!this.model || !peaks || peaks.length === 0) {
            return {
                prediction: "Analysis Failed",
                confidence: 0,
                explanation: "Could not run AI analysis due to missing model or insufficient data."
            };
        }

        try {
            // Updated to match modelTrainer.ts: 135 samples for 360Hz
            const beatLength = 135; // Updated from 94 to 135
            const halfBeat = Math.floor(beatLength / 2); // 67 samples
            const beatClassifications = {
                normal: 0,
                supraventricular: 0,
                ventricular: 0,
                fusion: 0,
                other: 0
            };

            let totalBeats = 0;
            let validPredictions = 0;

           
            // Analyze individual beats around R-peaks
            for (const peak of peaks) {
                const startIdx = peak - halfBeat;
                const endIdx = peak + halfBeat + (beatLength % 2); // For odd beatLength

                // Check bounds
                if (startIdx < 0 || endIdx >= ecgData.length) {
                    continue;
                }

                const beat = ecgData.slice(startIdx, endIdx);
                if (beat.length !== beatLength) {
                    continue;
                }

                // Z-score normalization (matching modelTrainer.ts approach)
                const mean = beat.reduce((a, b) => a + b, 0) / beat.length;
                const std = Math.sqrt(beat.reduce((a, b) => a + (b - mean) ** 2, 0) / beat.length);
                
                if (std <= 0.001) {
                   
                    continue;
                }

                const normalizedBeat = beat.map(x => (x - mean) / std);
                
                // Create input tensor for the model - shape [1, 135, 1]
                const inputTensor = tf.tensor3d([normalizedBeat.map(v => [v])], [1, beatLength, 1]);
                
                try {
                    const outputTensor = this.model.predict(inputTensor) as tf.Tensor;
                    const probabilities = await outputTensor.data();
                    
                    const predArray = Array.from(probabilities);
                    const maxIndex = predArray.indexOf(Math.max(...predArray));
                    const confidence = predArray[maxIndex];
                    
                    if (maxIndex >= 0 && maxIndex < AAMI_CLASSES.length && confidence > 0.5) {
                        const predictedClass = AAMI_CLASSES[maxIndex].toLowerCase();
                        
                        // Count beat classifications
                        switch (predictedClass) {
                            case 'normal':
                                beatClassifications.normal++;
                                break;
                            case 'supraventricular':
                                beatClassifications.supraventricular++;
                                break;
                            case 'ventricular':
                                beatClassifications.ventricular++;
                                break;
                            case 'fusion':
                                beatClassifications.fusion++;
                                break;
                            case 'other':
                                beatClassifications.other++;
                                break;
                        }
                        validPredictions++;
                    }
                    
                    outputTensor.dispose();
                } catch (err) {
                    console.warn('Failed to predict beat:', err);
                }
                
                inputTensor.dispose();
                totalBeats++;
            }

           

            // Determine overall rhythm classification based on beat analysis
            let overallPrediction = "Normal Sinus Rhythm";
            let overallConfidence = 0;

            if (validPredictions > 0) {
                const totalValidBeats = Object.values(beatClassifications).reduce((sum, count) => sum + count, 0);
                
                // Calculate percentages
                const normalPercent = (beatClassifications.normal / totalValidBeats) * 100;
                const ventricularPercent = (beatClassifications.ventricular / totalValidBeats) * 100;
                const supraventricularPercent = (beatClassifications.supraventricular / totalValidBeats) * 100;
                const fusionPercent = (beatClassifications.fusion / totalValidBeats) * 100;
                const otherPercent = (beatClassifications.other / totalValidBeats) * 100;

                // Determine overall classification (matching modelTrainer.ts logic)
                if (normalPercent >= 80) {
                    overallPrediction = "Normal Sinus Rhythm";
                    overallConfidence = normalPercent;
                } else if (ventricularPercent > 10) {
                    overallPrediction = "Ventricular Arrhythmia";
                    overallConfidence = ventricularPercent;
                } else if (supraventricularPercent > 10) {
                    overallPrediction = "Supraventricular Arrhythmia";
                    overallConfidence = supraventricularPercent;
                } else if (fusionPercent > 5) {
                    overallPrediction = "Fusion Beats Detected";
                    overallConfidence = fusionPercent;
                } else if (otherPercent > 15) {
                    overallPrediction = "Abnormal Rhythm";
                    overallConfidence = otherPercent;
                } else {
                    // Mixed rhythm
                    overallPrediction = "Mixed Rhythm Pattern";
                    overallConfidence = Math.max(normalPercent, ventricularPercent, supraventricularPercent);
                }

                // Additional checks based on HRV and intervals
                if (intervals && intervals.status.bpm === 'bradycardia') {
                    overallPrediction = "Bradycardia";
                } else if (intervals && intervals.status.bpm === 'tachycardia') {
                    overallPrediction = "Tachycardia";
                }
            }

            return {
                prediction: overallPrediction,
                confidence: overallConfidence,
                explanation: this.getExplanationForClassification(overallPrediction, beatClassifications),
                beatClassifications: beatClassifications
            };

        } catch (err) {
            console.error('Beat-level classification failed:', err);
            return {
                prediction: "Error",
                confidence: 0,
                explanation: "An error occurred during beat-level analysis."
            };
        }
    }

    private getExplanationForClassification(
        prediction: string, 
        beatClassifications: {
            normal: number;
            supraventricular: number;
            ventricular: number;
            fusion: number;
            other: number;
        }
    ): string {
        const total = Object.values(beatClassifications).reduce((sum, count) => sum + count, 0);
        
        if (total === 0) {
            return "Insufficient data for reliable analysis.";
        }

        const explanations: { [key: string]: string } = {
            "Normal Sinus Rhythm": `Analysis shows predominantly normal beats (${((beatClassifications.normal / total) * 100).toFixed(1)}% of ${total} analyzed beats).`,
            "Ventricular Arrhythmia": `Detected ${beatClassifications.ventricular} ventricular beats out of ${total} analyzed beats (${((beatClassifications.ventricular / total) * 100).toFixed(1)}%).`,
            "Supraventricular Arrhythmia": `Detected ${beatClassifications.supraventricular} supraventricular beats out of ${total} analyzed beats (${((beatClassifications.supraventricular / total) * 100).toFixed(1)}%).`,
            "Fusion Beats Detected": `Found ${beatClassifications.fusion} fusion beats out of ${total} analyzed beats, indicating mixed conduction patterns.`,
            "Abnormal Rhythm": `Analysis detected irregular patterns in ${beatClassifications.other} out of ${total} beats.`,
            "Mixed Rhythm Pattern": `Complex rhythm with multiple beat types: Normal (${beatClassifications.normal}), Ventricular (${beatClassifications.ventricular}), Supraventricular (${beatClassifications.supraventricular}).`,
            "Bradycardia": "Slow heart rate detected, which may indicate an underlying conduction issue.",
            "Tachycardia": "Elevated heart rate detected, which could be due to various factors."
        };

        return explanations[prediction] || 
            `Beat analysis completed on ${total} beats using updated 360Hz model. The AI model identified patterns requiring further evaluation.`;
    }

    private detectAbnormalities(
        intervals: any,
        stSegmentData: any,
        hrvMetrics: any,
        aiClassification: any,
        patientInfo: PatientInfo
    ): { type: string; severity: 'low' | 'medium' | 'high'; description: string }[] {
        const abnormalities: { type: string; severity: 'low' | 'medium' | 'high'; description: string }[] = [];

        if (intervals) {
            if (intervals.status.bpm === 'bradycardia') {
                abnormalities.push({
                    type: 'Bradycardia',
                    severity: 'medium',
                    description: 'Slow heart rate detected, which may indicate an underlying condition.'
                });
            }

            if (intervals.status.bpm === 'tachycardia') {
                abnormalities.push({
                    type: 'Tachycardia',
                    severity: 'medium',
                    description: 'Elevated heart rate detected, which could be due to exertion, stress, or cardiac issues.'
                });
            }

            if (intervals.status.pr === 'long') {
                abnormalities.push({
                    type: 'Prolonged PR Interval',
                    severity: 'medium',
                    description: 'Delayed conduction from atria to ventricles detected.'
                });
            }

            if (intervals.status.qrs === 'wide') {
                abnormalities.push({
                    type: 'Wide QRS Complex',
                    severity: 'medium',
                    description: 'Delayed ventricular conduction detected, possibly indicating bundle branch block.'
                });
            }

            if (intervals.status.qtc === 'prolonged') {
                abnormalities.push({
                    type: 'Prolonged QTc',
                    severity: 'high',
                    description: 'Prolonged QTc interval increases risk of dangerous arrhythmias.'
                });
            }
        }

        // Check for ventricular arrhythmias based on beat classification
        if (aiClassification.beatClassifications && aiClassification.beatClassifications.ventricular > 0) {
            const totalBeats = (Object.values(aiClassification.beatClassifications) as number[]).reduce((sum, count) => sum + count, 0);
            const ventricularPercent = (aiClassification.beatClassifications.ventricular / totalBeats) * 100;
            
            if (ventricularPercent > 10) {
                abnormalities.push({
                    type: 'Ventricular Arrhythmia',
                    severity: 'high',
                    description: `${ventricularPercent.toFixed(1)}% of beats classified as ventricular, indicating possible PVCs or VT.`
                });
            }
        }

        // Check ST segment
        if (stSegmentData) {
            if (stSegmentData.status === 'elevation') {
                abnormalities.push({
                    type: 'ST Elevation',
                    severity: 'high',
                    description: 'ST segment elevation detected, which may indicate myocardial injury.'
                });
            } else if (stSegmentData.status === 'depression') {
                abnormalities.push({
                    type: 'ST Depression',
                    severity: 'medium',
                    description: 'ST segment depression detected, which may indicate ischemia.'
                });
            }
        }

        return abnormalities;
    }

    private generateRecommendations(
        abnormalities: { type: string; severity: 'low' | 'medium' | 'high'; description: string }[],
        aiClassification: { 
            prediction: string; 
            confidence: number; 
            explanation: string;
            beatClassifications?: {
                normal: number;
                supraventricular: number;
                ventricular: number;
                fusion: number;
                other: number;
            };
        },
        patientInfo: PatientInfo
    ): string[] {
        const recommendations: string[] = [];

        recommendations.push(
            "Remember that this device is not a medical diagnostic tool. Always consult with a healthcare professional."
        );

        if (abnormalities.length > 0 || aiClassification.prediction !== "Normal Sinus Rhythm") {
            recommendations.push(
                "Based on the patterns detected, consider scheduling a consultation with a cardiologist."
            );
        }

        // Specific recommendations based on beat classifications
        if (aiClassification.beatClassifications) {
            const total = Object.values(aiClassification.beatClassifications).reduce((sum, count) => sum + count, 0);
            const ventricularPercent = (aiClassification.beatClassifications.ventricular / total) * 100;
            
            if (ventricularPercent > 10) {
                recommendations.push(
                    "Frequent ventricular beats detected. Avoid excessive caffeine and monitor for symptoms like palpitations."
                );
            }
        }

        if (abnormalities.some(abn => abn.severity === 'high')) {
            recommendations.push(
                "High-priority abnormalities detected. Seek immediate medical attention if experiencing symptoms."
            );
        }

        return recommendations;
    }

    private calculateHeartRateStats(peaks: number[], sampleRate: number, duration: number): {
        average: number;
        min: number;
        max: number;
    } {
        if (!peaks || peaks.length < 2) {
            console.warn("No valid R-peaks detected.");
            return { average: 0, min: 0, max: 0 };
        }

        const rrIntervals = [];
        for (let i = 1; i < peaks.length; i++) {
            const rr = (peaks[i] - peaks[i - 1]) * (1000 / sampleRate);
            rrIntervals.push(rr);
        }

        // Filter RR intervals to physiological range (300ms to 2000ms)
        const filteredRRs = rrIntervals.filter(rr => rr >= 300 && rr <= 2000);

        if (filteredRRs.length === 0) {
            console.warn("No valid RR intervals after filtering.");
            return { average: 0, min: 0, max: 0 };
        }

        const instantHRs = filteredRRs.map(rr => 60000 / rr);

        const average = instantHRs.reduce((sum, hr) => sum + hr, 0) / instantHRs.length;
        const min = Math.min(...instantHRs);
        const max = Math.max(...instantHRs);

        return {
            average: isNaN(average) ? 0 : average,
            min: isNaN(min) ? 0 : min,
            max: isNaN(max) ? 0 : max
        };
    }

    private countIrregularBeats(peaks: number[], sampleRate: number): number {
        if (!peaks || peaks.length < 3) return 0;
        const rrIntervals = [];
        for (let i = 1; i < peaks.length; i++) {
            rrIntervals.push((peaks[i] - peaks[i - 1]) * (1000 / sampleRate));
        }
        const median = rrIntervals.sort((a, b) => a - b)[Math.floor(rrIntervals.length / 2)];
        return rrIntervals.filter(rr => Math.abs(rr - median) > median * 0.2).length;
    }

    private calculatePercentIrregular(peaks: number[], sampleRate: number): number {
        if (!peaks || peaks.length < 3) return 0;
        const totalIntervals = peaks.length - 1;
        const irregularCount = this.countIrregularBeats(peaks, sampleRate);
        return (irregularCount / totalIntervals) * 100;
    }

    private formatDuration(seconds: number): string {
        const min = Math.floor(seconds / 60);
        const sec = Math.floor(seconds % 60);
        return `${min}:${sec.toString().padStart(2, '0')}`;
    }

    private determineHeartRateStatus(bpm: number): string {
        if (isNaN(bpm) || bpm <= 0) return 'unknown';
        if (bpm < 60) return 'bradycardia';
        if (bpm > 100) return 'tachycardia';
        return 'normal';
    }
}