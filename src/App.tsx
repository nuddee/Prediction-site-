import React, { useState, useEffect, useRef } from 'react';
import { Camera, Settings, Play, Square, AlertCircle, Video, ChevronLeft, Server, Sparkles, Home, Info, Image as ImageIcon, Upload } from 'lucide-react';
import { GoogleGenAI, Type } from '@google/genai';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

type BoundingBox = {
  label: string;
  confidence: number;
  box: [number, number, number, number]; // [ymin, xmin, ymax, xmax]
};

type Screen = 'menu' | 'camera' | 'settings' | 'upload';

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<Screen>('menu');
  
  // Settings State
  const [apiUrl, setApiUrl] = useState('');
  const [useGemini, setUseGemini] = useState(true);
  const [intervalMs, setIntervalMs] = useState(2000);
  const [isRealtime, setIsRealtime] = useState(false);
  
  // Camera & Detection State
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const [lastPredictionTime, setLastPredictionTime] = useState(0);

  // Upload State
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isPredictingImage, setIsPredictingImage] = useState(false);
  const uploadImageRef = useRef<HTMLImageElement>(null);
  const uploadCanvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Start/Stop Camera
  const toggleCamera = async (forceState?: boolean) => {
    const targetState = forceState !== undefined ? forceState : !cameraActive;
    
    if (!targetState) {
      const currentStream = videoRef.current?.srcObject as MediaStream || stream;
      currentStream?.getTracks().forEach(track => track.stop());
      if (videoRef.current) videoRef.current.srcObject = null;
      setStream(null);
      setCameraActive(false);
      setIsDetecting(false);
    } else {
      try {
        const newStream = await navigator.mediaDevices.getUserMedia({ 
          video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } } 
        });
        setStream(newStream);
        setCameraActive(true);
        setError(null);
      } catch (err: any) {
        setError(`Failed to access camera: ${err.message}`);
        setCameraActive(false);
      }
    }
  };

  // Attach stream to video element when it becomes available
  useEffect(() => {
    if (currentScreen === 'camera' && cameraActive && stream && videoRef.current) {
      if (videoRef.current.srcObject !== stream) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch(console.error);
      }
    }
  }, [currentScreen, cameraActive, stream]);

  // Clean up camera when leaving camera screen
  useEffect(() => {
    if (currentScreen !== 'camera' && cameraActive) {
      toggleCamera(false);
    }
  }, [currentScreen]);

  // Draw bounding boxes on canvas
  const drawBoxesOnCanvas = (boxes: BoundingBox[], canvas: HTMLCanvasElement | null, mediaElement: HTMLElement | null) => {
    if (!canvas || !mediaElement) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Match canvas size to media display size
    canvas.width = mediaElement.clientWidth;
    canvas.height = mediaElement.clientHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    boxes.forEach(({ label, confidence, box }) => {
      const [ymin, xmin, ymax, xmax] = box;
      
      // Convert normalized coordinates (0-1000) to pixel coordinates
      const x = (xmin / 1000) * canvas.width;
      const y = (ymin / 1000) * canvas.height;
      const width = ((xmax - xmin) / 1000) * canvas.width;
      const height = ((ymax - ymin) / 1000) * canvas.height;

      // Draw box
      ctx.strokeStyle = '#10B981'; // Emerald 500
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      // Draw label background
      const text = `${label} ${(confidence * 100).toFixed(0)}%`;
      ctx.font = '600 14px Inter, sans-serif';
      const textWidth = ctx.measureText(text).width;
      
      ctx.fillStyle = '#10B981';
      ctx.fillRect(x, y - 24, textWidth + 12, 24);

      // Draw label text
      ctx.fillStyle = '#000000';
      ctx.fillText(text, x + 6, y - 7);
    });
  };

  // Perform detection on camera
  const performDetection = async () => {
    if (!videoRef.current || !canvasRef.current || !cameraActive || videoRef.current.videoWidth === 0) return;

    const startTime = performance.now();

    try {
      // Capture frame to canvas
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = videoRef.current.videoWidth;
      tempCanvas.height = videoRef.current.videoHeight;
      const ctx = tempCanvas.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(videoRef.current, 0, 0);

      let boxes: BoundingBox[] = [];

      if (useGemini) {
        // Use Gemini 2.5 Flash
        const base64Data = tempCanvas.toDataURL('image/jpeg', 0.8).split(',')[1];
        
        const response = await ai.models.generateContent({
          model: 'gemini-2.5-flash',
          contents: [
            {
              inlineData: {
                data: base64Data,
                mimeType: 'image/jpeg'
              }
            },
            {
              text: 'Detect all prominent objects in this image. Return a JSON array of objects, each with a "label", a "confidence" (number between 0 and 1), and a "box" array containing [ymin, xmin, ymax, xmax] normalized from 0 to 1000.'
            }
          ],
          config: {
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  label: { type: Type.STRING },
                  confidence: { type: Type.NUMBER },
                  box: {
                    type: Type.ARRAY,
                    items: { type: Type.NUMBER },
                    description: "[ymin, xmin, ymax, xmax] normalized from 0 to 1000"
                  }
                },
                required: ["label", "confidence", "box"]
              }
            }
          }
        });

        if (response.text) {
          boxes = JSON.parse(response.text);
        }
      } else {
        // Use Custom YOLO API
        if (!apiUrl) throw new Error('API URL is required for custom YOLO model');
        
        const blob = await new Promise<Blob | null>(resolve => tempCanvas.toBlob(resolve, 'image/jpeg', 0.8));
        if (!blob) throw new Error('Failed to create image blob');

        const formData = new FormData();
        formData.append('file', blob);

        const res = await fetch(apiUrl, {
          method: 'POST',
          body: formData
        });

        if (!res.ok) throw new Error(`API returned ${res.status}`);
        
        const data = await res.json();
        boxes = Array.isArray(data) ? data : (data.boxes || data.predictions || []);
      }

      drawBoxesOnCanvas(boxes, canvasRef.current, videoRef.current);
      setError(null);

      const endTime = performance.now();
      setLastPredictionTime(endTime - startTime);
      setFps(1000 / (endTime - startTime));

    } catch (err: any) {
      console.error('Detection error:', err);
      setError(`Detection failed: ${err.message}`);
      drawBoxesOnCanvas([], canvasRef.current, videoRef.current);
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const imageUrl = URL.createObjectURL(file);
    setUploadedImage(imageUrl);
    setCurrentScreen('upload');
    setError(null);
    
    // Clear previous canvas
    if (uploadCanvasRef.current) {
      const ctx = uploadCanvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, uploadCanvasRef.current.width, uploadCanvasRef.current.height);
    }
  };

  const performImagePrediction = async () => {
    if (!uploadImageRef.current || !uploadCanvasRef.current) return;

    setIsPredictingImage(true);
    setError(null);

    const startTime = performance.now();

    try {
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = uploadImageRef.current.naturalWidth;
      tempCanvas.height = uploadImageRef.current.naturalHeight;
      const ctx = tempCanvas.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(uploadImageRef.current, 0, 0);

      let boxes: BoundingBox[] = [];

      if (useGemini) {
        const base64Data = tempCanvas.toDataURL('image/jpeg', 0.8).split(',')[1];
        
        const response = await ai.models.generateContent({
          model: 'gemini-2.5-flash',
          contents: [
            {
              inlineData: {
                data: base64Data,
                mimeType: 'image/jpeg'
              }
            },
            {
              text: 'Detect all prominent objects in this image. Return a JSON array of objects, each with a "label", a "confidence" (number between 0 and 1), and a "box" array containing [ymin, xmin, ymax, xmax] normalized from 0 to 1000.'
            }
          ],
          config: {
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  label: { type: Type.STRING },
                  confidence: { type: Type.NUMBER },
                  box: {
                    type: Type.ARRAY,
                    items: { type: Type.NUMBER },
                    description: "[ymin, xmin, ymax, xmax] normalized from 0 to 1000"
                  }
                },
                required: ["label", "confidence", "box"]
              }
            }
          }
        });

        if (response.text) {
          boxes = JSON.parse(response.text);
        }
      } else {
        if (!apiUrl) throw new Error('API URL is required for custom YOLO model');
        
        const blob = await new Promise<Blob | null>(resolve => tempCanvas.toBlob(resolve, 'image/jpeg', 0.8));
        if (!blob) throw new Error('Failed to create image blob');

        const formData = new FormData();
        formData.append('file', blob);

        const res = await fetch(apiUrl, {
          method: 'POST',
          body: formData
        });

        if (!res.ok) throw new Error(`API returned ${res.status}`);
        
        const data = await res.json();
        boxes = Array.isArray(data) ? data : (data.boxes || data.predictions || []);
      }

      drawBoxesOnCanvas(boxes, uploadCanvasRef.current, uploadImageRef.current);

      const endTime = performance.now();
      setLastPredictionTime(endTime - startTime);

    } catch (err: any) {
      console.error('Detection error:', err);
      setError(`Detection failed: ${err.message}`);
    } finally {
      setIsPredictingImage(false);
    }
  };

  // Detection Loop
  useEffect(() => {
    let isActive = true;
    let timeoutId: number;

    const loop = async () => {
      if (!isActive || !isDetecting || !cameraActive || currentScreen !== 'camera') return;
      
      await performDetection();
      
      if (isActive) {
        timeoutId = window.setTimeout(loop, isRealtime ? 0 : intervalMs);
      }
    };

    if (isDetecting && cameraActive && currentScreen === 'camera') {
      loop();
    } else {
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }

    return () => {
      isActive = false;
      clearTimeout(timeoutId);
    };
  }, [isDetecting, cameraActive, currentScreen, intervalMs, isRealtime, useGemini, apiUrl]);

  // Handle window resize to keep canvas aligned
  useEffect(() => {
    const handleResize = () => {
      if (canvasRef.current && videoRef.current) {
        canvasRef.current.width = videoRef.current.clientWidth;
        canvasRef.current.height = videoRef.current.clientHeight;
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // --- Screens ---

  const renderMenuScreen = () => (
    <div className="flex-1 flex flex-col p-6 bg-zinc-950 overflow-y-auto">
      <div className="mt-8 mb-10">
        <div className="w-16 h-16 bg-emerald-500/10 rounded-2xl flex items-center justify-center mb-6 shadow-[0_0_30px_rgba(16,185,129,0.15)]">
          <Video className="w-8 h-8 text-emerald-500" />
        </div>
        <h1 className="text-4xl font-bold text-white mb-2 tracking-tight">YOLO Vision</h1>
        <p className="text-zinc-400 text-lg">
          Real-time object detection.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 w-full max-w-md">
        <button 
          onClick={() => {
            setCurrentScreen('camera');
            toggleCamera(true);
          }}
          className="col-span-2 bg-emerald-500 hover:bg-emerald-400 text-zinc-950 rounded-[32px] p-6 flex flex-col items-start justify-between gap-8 transition-all active:scale-[0.98] min-h-[180px]"
        >
          <div className="bg-zinc-950/10 p-3 rounded-2xl">
            <Camera className="w-8 h-8" />
          </div>
          <div className="text-left">
            <div className="font-bold text-2xl mb-1">Open Camera</div>
            <div className="text-emerald-950/70 text-sm font-medium">Start real-time detection</div>
          </div>
        </button>
        
        <button 
          onClick={() => setCurrentScreen('settings')}
          className="bg-zinc-900 hover:bg-zinc-800 text-white border border-zinc-800/50 rounded-[32px] p-6 flex flex-col items-start justify-between gap-6 transition-all active:scale-[0.98] min-h-[160px]"
        >
          <div className="bg-zinc-800/50 p-3 rounded-2xl">
            <Settings className="w-6 h-6 text-zinc-300" />
          </div>
          <div className="text-left">
            <div className="font-bold text-lg mb-1">Settings</div>
            <div className="text-zinc-500 text-xs font-medium">Configure models</div>
          </div>
        </button>

        <button 
          onClick={() => fileInputRef.current?.click()}
          className="bg-zinc-900 hover:bg-zinc-800 border border-zinc-800/50 text-white rounded-[32px] p-6 flex flex-col items-start justify-between gap-6 transition-all active:scale-[0.98] min-h-[160px]"
        >
          <div className="bg-indigo-500/10 p-3 rounded-2xl">
            <ImageIcon className="w-6 h-6 text-indigo-400" />
          </div>
          <div className="text-left">
            <div className="font-bold text-lg mb-1">Upload Image</div>
            <div className="text-zinc-500 text-xs font-medium">Predict from gallery</div>
          </div>
        </button>
      </div>
      
      <input 
        type="file" 
        accept="image/*" 
        className="hidden" 
        ref={fileInputRef}
        onChange={handleImageUpload}
      />
    </div>
  );

  const renderUploadScreen = () => (
    <div className="flex-1 flex flex-col bg-black relative overflow-hidden">
      {/* Top Bar */}
      <div className="absolute top-0 inset-x-0 p-4 z-20 flex items-center justify-between bg-gradient-to-b from-black/80 to-transparent">
        <button 
          onClick={() => setCurrentScreen('menu')}
          className="w-10 h-10 bg-black/50 backdrop-blur-md rounded-full flex items-center justify-center text-white border border-white/10 active:bg-white/20 transition-colors"
        >
          <ChevronLeft className="w-6 h-6" />
        </button>
        
        <button 
          onClick={() => setCurrentScreen('settings')}
          className="w-10 h-10 bg-black/50 backdrop-blur-md rounded-full flex items-center justify-center text-white border border-white/10 active:bg-white/20 transition-colors"
        >
          <Settings className="w-5 h-5" />
        </button>
      </div>

      {/* Image Viewport */}
      <div className="flex-1 relative flex items-center justify-center p-4 mt-16 mb-24">
        {uploadedImage ? (
          <div className="relative w-full h-full flex items-center justify-center">
            <img 
              ref={uploadImageRef}
              src={uploadedImage}
              alt="Uploaded"
              className="max-w-full max-h-full object-contain rounded-lg"
              onLoad={() => {
                // Resize canvas to match image when loaded
                if (uploadCanvasRef.current && uploadImageRef.current) {
                  uploadCanvasRef.current.width = uploadImageRef.current.clientWidth;
                  uploadCanvasRef.current.height = uploadImageRef.current.clientHeight;
                }
              }}
            />
            <canvas 
              ref={uploadCanvasRef}
              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none"
            />
          </div>
        ) : (
          <div className="text-zinc-500">No image selected</div>
        )}
      </div>

      {/* Bottom Controls */}
      <div className="absolute bottom-0 inset-x-0 p-6 pb-8 z-20 bg-gradient-to-t from-black/90 via-black/50 to-transparent flex flex-col items-center">
        {error && (
          <div className="mb-4 bg-red-500/90 text-white text-sm px-4 py-2 rounded-lg max-w-xs text-center backdrop-blur-sm">
            {error}
          </div>
        )}
        
        <div className="flex gap-4 w-full max-w-xs">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex-1 bg-zinc-800 hover:bg-zinc-700 text-white py-3 rounded-xl font-medium transition-colors"
          >
            Change Image
          </button>
          <button
            onClick={performImagePrediction}
            disabled={isPredictingImage || !uploadedImage}
            className={`flex-1 py-3 rounded-xl font-medium transition-colors flex items-center justify-center gap-2 ${
              isPredictingImage || !uploadedImage
                ? 'bg-emerald-500/50 text-white/50 cursor-not-allowed'
                : 'bg-emerald-500 hover:bg-emerald-400 text-white shadow-[0_0_15px_rgba(16,185,129,0.3)]'
            }`}
          >
            {isPredictingImage ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <>
                <Sparkles className="w-5 h-5" />
                Predict
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );

  const renderCameraScreen = () => (
    <div className="flex-1 flex flex-col bg-black relative overflow-hidden">
      {/* Top Bar */}
      <div className="absolute top-0 inset-x-0 p-4 z-20 flex items-center justify-between bg-gradient-to-b from-black/80 to-transparent">
        <button 
          onClick={() => setCurrentScreen('menu')}
          className="w-10 h-10 bg-black/50 backdrop-blur-md rounded-full flex items-center justify-center text-white border border-white/10 active:bg-white/20 transition-colors"
        >
          <ChevronLeft className="w-6 h-6" />
        </button>
        
        {isDetecting && (
          <div className="bg-black/50 backdrop-blur-md border border-white/10 rounded-full px-4 py-1.5 flex items-center gap-2 text-xs font-mono text-zinc-300">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            {fps.toFixed(1)} FPS
          </div>
        )}

        <button 
          onClick={() => setCurrentScreen('settings')}
          className="w-10 h-10 bg-black/50 backdrop-blur-md rounded-full flex items-center justify-center text-white border border-white/10 active:bg-white/20 transition-colors"
        >
          <Settings className="w-5 h-5" />
        </button>
      </div>

      {/* Video Viewport */}
      <div className="flex-1 relative flex items-center justify-center">
        {!cameraActive ? (
          <div className="text-center p-8">
            <AlertCircle className="w-12 h-12 mx-auto text-red-500 mb-4" />
            <p className="text-zinc-400 mb-4">{error || "Camera is initializing..."}</p>
            {error && (
              <button 
                onClick={() => toggleCamera(true)}
                className="bg-zinc-800 text-white px-6 py-3 rounded-xl font-medium"
              >
                Retry Camera
              </button>
            )}
          </div>
        ) : (
          <>
            <video 
              ref={videoRef} 
              className="absolute inset-0 w-full h-full object-cover"
              playsInline
              muted
            />
            <canvas 
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
            />
          </>
        )}
      </div>

      {/* Bottom Controls */}
      <div className="absolute bottom-0 inset-x-0 p-6 pb-8 z-20 bg-gradient-to-t from-black/90 via-black/50 to-transparent flex flex-col items-center">
        {error && cameraActive && (
          <div className="mb-4 bg-red-500/90 text-white text-sm px-4 py-2 rounded-lg max-w-xs text-center backdrop-blur-sm">
            {error}
          </div>
        )}
        
        <button
          onClick={() => setIsDetecting(!isDetecting)}
          disabled={!cameraActive}
          className={`w-20 h-20 rounded-full flex items-center justify-center transition-all active:scale-95 ${
            !cameraActive 
              ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
              : isDetecting
                ? 'bg-red-500 text-white shadow-[0_0_20px_rgba(239,68,68,0.4)]'
                : 'bg-emerald-500 text-white shadow-[0_0_20px_rgba(16,185,129,0.4)]'
          }`}
        >
          {isDetecting ? (
            <Square className="w-8 h-8 fill-current" />
          ) : (
            <Play className="w-8 h-8 fill-current ml-1" />
          )}
        </button>
        <p className="text-white/70 text-sm mt-4 font-medium">
          {isDetecting ? 'Tap to Stop' : 'Tap to Detect'}
        </p>
      </div>
    </div>
  );

  const renderSettingsScreen = () => (
    <div className="flex-1 flex flex-col bg-zinc-950">
      {/* Header */}
      <div className="p-4 border-b border-zinc-900 flex items-center gap-3 bg-zinc-950 sticky top-0 z-10">
        <button 
          onClick={() => setCurrentScreen('menu')}
          className="w-10 h-10 rounded-full flex items-center justify-center text-zinc-400 hover:bg-zinc-900 active:bg-zinc-800 transition-colors -ml-2"
        >
          <ChevronLeft className="w-6 h-6" />
        </button>
        <h2 className="text-xl font-semibold text-white">Settings</h2>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-8">
        {/* Model Selection */}
        <section className="space-y-4">
          <h3 className="text-sm font-semibold text-zinc-500 uppercase tracking-wider px-1">Model Engine</h3>
          <div className="space-y-3">
            <button
              onClick={() => setUseGemini(true)}
              className={`w-full flex items-center gap-4 p-4 rounded-2xl border transition-all ${
                useGemini 
                  ? 'bg-indigo-500/10 border-indigo-500/50 text-indigo-300' 
                  : 'bg-zinc-900 border-zinc-800 text-zinc-400'
              }`}
            >
              <div className={`p-2 rounded-xl ${useGemini ? 'bg-indigo-500/20' : 'bg-zinc-800'}`}>
                <Sparkles className="w-6 h-6" />
              </div>
              <div className="text-left flex-1">
                <div className="font-semibold text-zinc-200">Gemini 2.5 Flash</div>
                <div className="text-sm opacity-70">Built-in AI fallback</div>
              </div>
              <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${useGemini ? 'border-indigo-500' : 'border-zinc-600'}`}>
                {useGemini && <div className="w-2.5 h-2.5 bg-indigo-500 rounded-full" />}
              </div>
            </button>
            
            <button
              onClick={() => setUseGemini(false)}
              className={`w-full flex items-center gap-4 p-4 rounded-2xl border transition-all ${
                !useGemini 
                  ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-300' 
                  : 'bg-zinc-900 border-zinc-800 text-zinc-400'
              }`}
            >
              <div className={`p-2 rounded-xl ${!useGemini ? 'bg-emerald-500/20' : 'bg-zinc-800'}`}>
                <Server className="w-6 h-6" />
              </div>
              <div className="text-left flex-1">
                <div className="font-semibold text-zinc-200">Hosted YOLO API</div>
                <div className="text-sm opacity-70">Custom endpoint</div>
              </div>
              <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${!useGemini ? 'border-emerald-500' : 'border-zinc-600'}`}>
                {!useGemini && <div className="w-2.5 h-2.5 bg-emerald-500 rounded-full" />}
              </div>
            </button>
          </div>
        </section>

        {/* Custom API Settings */}
        {!useGemini && (
          <section className="space-y-4 animate-in fade-in slide-in-from-top-4 duration-300">
            <h3 className="text-sm font-semibold text-zinc-500 uppercase tracking-wider px-1">API Configuration</h3>
            <div className="bg-zinc-900 rounded-2xl p-4 border border-zinc-800 space-y-4">
              <div>
                <label className="block text-sm font-medium text-zinc-300 mb-2">Endpoint URL</label>
                <input 
                  type="url" 
                  value={apiUrl}
                  onChange={(e) => setApiUrl(e.target.value)}
                  placeholder="https://your-yolo-api.com/predict"
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-all"
                />
              </div>
              <div className="flex gap-3 items-start bg-zinc-950/50 p-3 rounded-xl border border-zinc-800/50">
                <Info className="w-5 h-5 text-zinc-500 shrink-0 mt-0.5" />
                <p className="text-xs text-zinc-400 leading-relaxed">
                  Expects a POST request with a <code className="text-emerald-400">file</code> form-data field. Must return a JSON array of objects with <code className="text-emerald-400">label</code>, <code className="text-emerald-400">confidence</code>, and <code className="text-emerald-400">box</code> [ymin, xmin, ymax, xmax] (0-1000).
                </p>
              </div>
            </div>
          </section>
        )}

        {/* Performance Settings */}
        <section className="space-y-4">
          <h3 className="text-sm font-semibold text-zinc-500 uppercase tracking-wider px-1">Performance</h3>
          <div className="bg-zinc-900 rounded-2xl p-5 border border-zinc-800 space-y-4">
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-zinc-300">Real-time Mode</label>
              <button 
                onClick={() => setIsRealtime(!isRealtime)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${isRealtime ? 'bg-emerald-500' : 'bg-zinc-700'}`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${isRealtime ? 'translate-x-6' : 'translate-x-1'}`} />
              </button>
            </div>
            
            {!isRealtime ? (
              <div className="pt-2 border-t border-zinc-800/50">
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm font-medium text-zinc-300">Polling Interval</label>
                  <span className="text-emerald-400 font-mono bg-emerald-500/10 px-2 py-1 rounded-md text-sm">{intervalMs}ms</span>
                </div>
                <input 
                  type="range" 
                  min="200" 
                  max="5000" 
                  step="100"
                  value={intervalMs}
                  onChange={(e) => setIntervalMs(Number(e.target.value))}
                  className="w-full accent-emerald-500 h-2 bg-zinc-800 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-zinc-500 mt-2">
                  <span>Fast (High load)</span>
                  <span>Slow (Low load)</span>
                </div>
              </div>
            ) : (
              <div className="text-xs text-zinc-400 bg-zinc-950 p-3 rounded-xl border border-zinc-800">
                Frames are processed continuously as fast as the model can handle them. This may cause high API usage.
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );

  return (
    <div className="h-screen w-full bg-zinc-950 text-zinc-100 font-sans flex flex-col overflow-hidden selection:bg-emerald-500/30">
      {currentScreen === 'menu' && renderMenuScreen()}
      {currentScreen === 'camera' && renderCameraScreen()}
      {currentScreen === 'upload' && renderUploadScreen()}
      {currentScreen === 'settings' && renderSettingsScreen()}
    </div>
  );
}
