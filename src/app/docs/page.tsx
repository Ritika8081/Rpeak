"use client";

import React from 'react';
import Link from 'next/link';

export default function DocsPage() {
  return (
    <div className="h-full h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="h-full scrollable-content p-6">
        <div className="max-w-4xl mx-auto pb-8">
          <h1 className="text-3xl font-bold text-white mb-8">Rpeak - User Guide</h1>
          
          <div className="bg-black/40 backdrop-blur-sm border border-white/20 rounded-xl p-6 mb-8">
            <h2 className="text-2xl font-bold text-white mb-4">Getting Started</h2>
            <ol className="list-decimal list-inside text-gray-300 space-y-4 mb-4">
              <li>
                <span className="font-semibold text-blue-400">Connect Your ECG Device</span>
                <ul className="list-disc list-inside ml-6 mt-2 text-gray-400">
                  <li>Click the <span className="font-semibold text-purple-400">🔵 Bluetooth</span> button in the left sidebar</li>
                  <li>Your browser will show available Bluetooth devices</li>
                  <li>Select your ECG device from the list</li>
                  <li>Wait for "Connected" status to appear</li>
                </ul>
              </li>
              
              <li>
                <span className="font-semibold text-blue-400">Start Monitoring</span>
                <ul className="list-disc list-inside ml-6 mt-2 text-gray-400">
                  <li>Once connected, ECG data will automatically start flowing</li>
                  <li>You'll see a real-time waveform on your screen</li>
                  <li>Heart rate will be calculated and displayed</li>
                  <li>The timer shows how long you've been monitoring</li>
                </ul>
              </li>
              
              <li>
                <span className="font-semibold text-blue-400">Use Analysis Tools</span>
                <ul className="list-disc list-inside ml-6 mt-2 text-gray-400">
                  <li>Click the buttons in the sidebar to enable different analysis features</li>
                  <li>Each tool provides different insights about your heart rhythm</li>
                  <li>You can enable multiple tools at the same time</li>
                </ul>
              </li>
            </ol>
          </div>

          <div className="bg-black/40 backdrop-blur-sm border border-white/20 rounded-xl p-6 mb-8">
            <h2 className="text-2xl font-bold text-white mb-4">Analysis Tools</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-purple-400 mb-2">📈 Peaks</h3>
                <p className="text-gray-300 text-sm mb-2">
                  Shows R peaks labeled on your ECG where heartbeats are detected.
                </p>
                <p className="text-gray-400 text-xs">
                  Useful for: Verifying that the system is correctly detecting your heartbeats.
                </p>
              </div>
              
              <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-green-400 mb-2">🫀 PQRST</h3>
                <p className="text-gray-300 text-sm mb-2">
                  Identifies the different parts of each heartbeat (P, Q, R, S, T waves).
                </p>
                <p className="text-gray-400 text-xs">
                  Useful for: Detailed cardiac analysis and understanding heart rhythm patterns.
                </p>
              </div>
              
              <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-yellow-400 mb-2">⏱️ Intervals</h3>
                <p className="text-gray-300 text-sm mb-2">
                  Displays timing measurements between different parts of your heartbeat.
                </p>
                <p className="text-gray-400 text-xs">
                  Shows: Heart rate, PR interval, QRS duration, QT interval with normal/abnormal indicators.
                </p>
              </div>
              
              <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-blue-400 mb-2">💓 HRV</h3>
                <p className="text-gray-300 text-sm mb-2">
                  Heart Rate Variability - measures how much your heart rate varies between beats.
                </p>
                <p className="text-gray-400 text-xs">
                  Useful for: Stress monitoring, fitness assessment, and overall heart health.
                </p>
              </div>
              
              <div className="p-4 bg-pink-500/10 border border-pink-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-pink-400 mb-2">🤖 AI Analysis</h3>
                <p className="text-gray-300 text-sm mb-2">
                  it's in development phase and may not be fully accurate. Uses machine learning to classify heartbeats.
                </p>
                <p className="text-gray-400 text-xs">
                  Note: Requires training a model first on the <Link href="/train" className="text-blue-400 hover:underline">Training Page</Link>.
                </p>
              </div>
              
              <div className="p-4 bg-orange-500/10 border border-orange-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-orange-400 mb-2">📊 Session</h3>
                <p className="text-gray-300 text-sm mb-2">
                  Records your ECG data for later analysis and generates detailed reports.
                </p>
                <p className="text-gray-400 text-xs">
                  Useful for: Keeping records, tracking progress, and sharing with healthcare providers.
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-black/40 backdrop-blur-sm border border-white/20 rounded-xl p-6 mb-8">
            <h2 className="text-2xl font-bold text-white mb-4">Understanding Your Results</h2>
            
            <div className="space-y-4">
              <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-green-400 mb-2">Heart Rate</h3>
                <ul className="text-gray-300 space-y-1 list-disc list-inside text-sm">
                  <li><span className="font-semibold">Normal:</span> 60-100 beats per minute (BPM)</li>
                  <li><span className="font-semibold">Bradycardia:</span> Below 60 BPM (may be normal for athletes)</li>
                  <li><span className="font-semibold">Tachycardia:</span> Above 100 BPM (may indicate stress or exercise)</li>
                </ul>
              </div>
              
              <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-blue-400 mb-2">ECG Intervals</h3>
                <ul className="text-gray-300 space-y-1 list-disc list-inside text-sm">
                  <li><span className="font-semibold">PR Interval:</span> Time from start of P wave to start of QRS (normal: 120-200ms)</li>
                  <li><span className="font-semibold">QRS Duration:</span> Width of QRS complex (normal: 80-120ms)</li>
                  <li><span className="font-semibold">QT Interval:</span> Time from start of QRS to end of T wave (varies with heart rate)</li>
                </ul>
              </div>
              
              <div className="p-4 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-purple-400 mb-2">HRV Metrics</h3>
                <ul className="text-gray-300 space-y-1 list-disc list-inside text-sm">
                  <li><span className="font-semibold">Higher HRV:</span> Generally indicates better cardiovascular fitness and stress resilience</li>
                  <li><span className="font-semibold">Lower HRV:</span> May indicate stress, fatigue, or overtraining</li>
                  <li><span className="font-semibold">RMSSD:</span> Higher values (&gt;30ms) are generally better</li>
                  <li><span className="font-semibold">Stress Level:</span> Low/Medium/High based on multiple HRV parameters</li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className="bg-black/40 backdrop-blur-sm border border-white/20 rounded-xl p-6 mb-8">
            <h2 className="text-2xl font-bold text-white mb-4">AI Heartbeat Classification</h2>
            <p className="text-gray-300 mb-4">
              The AI system can identify different types of heartbeats:
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-green-400 mb-2">Normal (N)</h3>
                <p className="text-gray-300 text-sm">
                  Regular, healthy heartbeats including normal beats and bundle branch blocks.
                </p>
              </div>
              
              <div className="p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-yellow-400 mb-2">Supraventricular (S)</h3>
                <p className="text-gray-300 text-sm">
                  Beats originating from above the ventricles, including atrial premature beats.
                </p>
              </div>
              
              <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-red-400 mb-2">Ventricular (V)</h3>
                <p className="text-gray-300 text-sm">
                  Beats originating from the ventricles, which may need medical attention.
                </p>
              </div>
              
              <div className="p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-blue-400 mb-2">Fusion (F)</h3>
                <p className="text-gray-300 text-sm">
                  Beats that are a combination of normal and abnormal patterns.
                </p>
              </div>
              
              <div className="p-3 bg-gray-500/10 border border-gray-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-gray-400 mb-2">Other (Q)</h3>
                <p className="text-gray-300 text-sm">
                  Unclassifiable beats, paced beats, or artifacts.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-black/40 backdrop-blur-sm border border-white/20 rounded-xl p-6 mb-8">
            <h2 className="text-2xl font-bold text-white mb-4">Tips for Best Results</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-green-400 mb-2">Signal Quality</h3>
                <ul className="text-gray-300 space-y-1 list-disc list-inside text-sm">
                  <li>Ensure good skin contact with ECG electrodes</li>
                  <li>Stay still during recordings to minimize noise</li>
                  <li>Clean skin before applying electrodes</li>
                  <li>Check that your device is properly charged</li>
                </ul>
              </div>
              
              <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-blue-400 mb-2">Optimal Monitoring</h3>
                <ul className="text-gray-300 space-y-1 list-disc list-inside text-sm">
                  <li>Record for at least 2-5 minutes for reliable HRV analysis</li>
                  <li>Monitor in a quiet, comfortable environment</li>
                  <li>Avoid talking or moving during critical measurements</li>
                  <li>Take regular breaks from monitoring</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-black/40 backdrop-blur-sm border border-white/20 rounded-xl p-6 mb-8">
            <h2 className="text-2xl font-bold text-white mb-4">Session Recording</h2>
            <p className="text-gray-300 mb-4">
              Recording sessions allows you to save and analyze your ECG data:
            </p>
            
            <ol className="list-decimal list-inside text-gray-300 space-y-2 mb-4">
              <li>Click <span className="text-orange-400">📊 Session</span> to open the recording panel</li>
              <li>Click "Start Recording" to begin capturing data</li>
              <li>Monitor for your desired duration (recommended: 5-10 minutes)</li>
              <li>Click "Stop Recording" to end the session</li>
              <li>View the automatically generated analysis report</li>
              <li>Export data if needed for sharing with healthcare providers</li>
            </ol>
            
            <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
              <h4 className="text-sm font-medium text-green-400">Session Reports Include:</h4>
              <ul className="text-gray-300 text-sm mt-1 list-disc list-inside">
                <li>Complete HRV analysis with trends over time</li>
                <li>Heart rate statistics and variability</li>
                <li>Detected arrhythmic events (if any)</li>
                <li>Signal quality assessment</li>
                <li>Exportable data for further analysis</li>
              </ul>
            </div>
          </div>
          
          <div className="bg-black/40 backdrop-blur-sm border border-white/20 rounded-xl p-6">
            <h2 className="text-2xl font-bold text-white mb-4">Important Disclaimers</h2>
            
            <div className="space-y-4">
              <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-red-400 mb-2">⚠️ Medical Disclaimer</h3>
                <p className="text-gray-300 text-sm">
                  This application is for educational and research purposes only. It is not a medical device and should not be used for medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals for medical advice.
                </p>
              </div>
              
              <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-blue-400 mb-2">🔒 Privacy & Data</h3>
                <p className="text-gray-300 text-sm">
                  All ECG processing happens locally in your browser. No ECG data is transmitted to external servers. Your health data remains private and secure on your device.
                </p>
              </div>
              
              <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                <h3 className="text-lg font-medium text-yellow-400 mb-2">🔧 Technical Support</h3>
                <p className="text-gray-300 text-sm">
                  This application requires a modern web browser with Bluetooth support. Chrome, Edge, and other Chromium-based browsers work best. Ensure your ECG device is compatible with Web Bluetooth.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}