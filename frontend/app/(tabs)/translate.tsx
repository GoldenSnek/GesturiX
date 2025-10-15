import React, { useState, useEffect, useRef } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StatusBar,
    ActivityIndicator,
    PanResponder,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { CameraView, useCameraPermissions, CameraType } from 'expo-camera';
import { useFocusEffect } from '@react-navigation/native';
import { useRouter } from 'expo-router';
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';
import AppHeader from '../../components/AppHeader';

// Configuration
const WS_URL = 'ws://192.168.255.136:8000/ws/translate'; // Replace with your computer's IP
const FRAME_INTERVAL = 200; // Send frame every 200ms (5 FPS)
const PREDICTION_TIMEOUT = 3000; // Clear prediction after 3s of no detection

export default function Translate() {
    const [isCameraActive, setIsCameraActive] = useState(false);
    const [facing, setFacing] = useState<CameraType>('front');
    const [isTranslating, setIsTranslating] = useState(false);
    const [translatedText, setTranslatedText] = useState('Camera Off. Tap to begin.');
    const [isConnected, setIsConnected] = useState(false);
    const [currentPrediction, setCurrentPrediction] = useState('');
    const [confidence, setConfidence] = useState(0);

    const [permission, requestPermission] = useCameraPermissions();
    const insets = useSafeAreaInsets();
    const router = useRouter();

    const cameraRef = useRef<any>(null);
    const wsRef = useRef<WebSocket | null>(null);
    // FIX: Use ReturnType<typeof setTimeout>
    const frameIntervalRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const predictionTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    // Swipe gesture
    const panResponder = useRef(
        PanResponder.create({
            onMoveShouldSetPanResponder: (_, gestureState) =>
                Math.abs(gestureState.dx) > 10,
            onPanResponderRelease: (_, gestureState) => {
                if (gestureState.dx < -30) {
                    router.push('/compose');
                }
            },
        })
    ).current;

    // Request camera permissions
    useEffect(() => {
        if (!permission?.granted) {
            requestPermission();
        }
    }, []);

    // Cleanup on unmount or screen blur
    useFocusEffect(
        React.useCallback(() => {
            return () => {
                stopTranslation();
                setIsCameraActive(false);
                setTranslatedText('Camera Off. Tap to begin.');
            };
        }, [])
    );

    // WebSocket Connection
    const connectWebSocket = () => {
        try {
            const ws = new WebSocket(WS_URL);

            ws.onopen = () => {
                console.log('✅ WebSocket connected');
                setIsConnected(true);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.error) {
                        console.error('Server error:', data.error);
                        return;
                    }

                    if (data.predictions && data.predictions.length > 0) {
                        const bestPrediction = data.predictions[0];
                        setCurrentPrediction(bestPrediction.label.toUpperCase());
                        setConfidence(bestPrediction.confidence);
                        setTranslatedText(bestPrediction.label.toUpperCase());

                        // Clear timeout and set new one
                        if (predictionTimeoutRef.current) {
                            clearTimeout(predictionTimeoutRef.current);
                        }
                        predictionTimeoutRef.current = setTimeout(() => {
                            setCurrentPrediction('');
                            setTranslatedText('No hand detected');
                        }, PREDICTION_TIMEOUT);
                    } else {
                        setCurrentPrediction('');
                        setTranslatedText('No hand detected');
                    }
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };

            ws.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                setIsConnected(false);
            };

            ws.onclose = () => {
                console.log('🔌 WebSocket disconnected');
                setIsConnected(false);
            };

            wsRef.current = ws;
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
        }
    };

    // Capture and send frame
    const captureAndSendFrame = async () => {
        if (!cameraRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            return;
        }

        try {
            // Take picture
            const photo = await cameraRef.current.takePictureAsync({
                quality: 0.3, // Lower quality for faster processing
                base64: true,
                skipProcessing: true,
            });

            if (!photo || !photo.base64) {
                return;
            }

            // Resize image for faster processing
            const resizedPhoto = await manipulateAsync(
                photo.uri,
                [{ resize: { width: 640 } }], // Resize to 640px width
                { compress: 0.5, format: SaveFormat.JPEG, base64: true }
            );

            // Send to backend
            const payload = {
                frame: resizedPhoto.base64,
            };

            wsRef.current.send(JSON.stringify(payload));
        } catch (error) {
            console.error('Error capturing frame:', error);
        }
    };

    // Start translation
    const startTranslation = () => {
        connectWebSocket();
        setTranslatedText('Connecting...');

        // Start frame capture interval
        frameIntervalRef.current = setInterval(() => {
            captureAndSendFrame();
        }, FRAME_INTERVAL);
    };

    // Stop translation
    const stopTranslation = () => {
    // Clear interval
    if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
    }

    // Clear prediction timeout
    if (predictionTimeoutRef.current) {
        clearTimeout(predictionTimeoutRef.current);
        predictionTimeoutRef.current = null;
    }

    // Close WebSocket
    if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
    }

    setIsConnected(false);
    setCurrentPrediction('');
    setIsTranslating(false);
    setIsCameraActive(false); // <<< This disables camera on stop
};


    // Toggle translation
    const toggleTranslation = () => {
        if (!permission?.granted) {
            requestPermission();
            return;
        }

        if (!isCameraActive) {
            setIsCameraActive(true);
            setTranslatedText('Camera On. Tap again to start recognition.');
            return;
        }

        setIsTranslating((prev) => {
            const newState = !prev;
            if (newState) {
                startTranslation();
            } else {
                stopTranslation();
                setTranslatedText('Recognition paused. Tap to continue.');
            }
            return newState;
        });
    };

    // Flip camera
    const flipCamera = () => {
        setFacing((prev) => (prev === 'back' ? 'front' : 'back'));
    };

    // Stop camera
    const stopCamera = () => {
        stopTranslation();
        setIsCameraActive(false);
        setIsTranslating(false);
        setTranslatedText('Camera Off. Tap to begin.');
    };

    if (!permission) {
        return (
            <View className="flex-1 justify-center items-center bg-secondary">
                <ActivityIndicator size="large" color="#FF6B00" />
                <Text className="text-primary mt-4">Loading camera permissions...</Text>
            </View>
        );
    }

    if (!permission.granted) {
        return (
            <View className="flex-1 justify-center items-center p-8 bg-secondary" style={{ paddingTop: insets.top }}>
                <Text className="text-primary text-xl font-bold mb-4 text-center">Camera Access Required</Text>
                <Text className="text-neutral text-center mb-6">
                    GesturiX needs access to your camera to recognize sign language.
                </Text>
                <TouchableOpacity
                    className="bg-accent rounded-full px-6 py-3 shadow-md"
                    onPress={requestPermission}
                >
                    <Text className="text-white font-bold">Grant Camera Permission</Text>
                </TouchableOpacity>
            </View>
        );
    }

    return (
        <View {...panResponder.panHandlers} className="flex-1 bg-secondary" style={{ paddingTop: insets.top }}>
            <StatusBar barStyle="light-content" />

            <AppHeader />

            {/* Camera Viewfinder */}
            <View className="px-5 pt-5 items-center">
                <View className="w-full aspect-[4/3] bg-primary rounded-2xl overflow-hidden mb-5 relative">
                    {isCameraActive && (
                        <CameraView
                            style={{ flex: 1 }}
                            facing={facing}
                            ref={cameraRef}
                        />
                    )}

                    {/* Overlay when not translating */}
                    {(!isCameraActive || !isTranslating) && (
                        <View className="absolute inset-0 justify-center items-center px-5 bg-black/50">
                            <MaterialIcons
                                name={isCameraActive ? 'pause-circle-outline' : 'videocam-off'}
                                size={80}
                                color="white"
                            />
                            <Text className="text-lg font-semibold text-white mt-4 mb-2 text-center">
                                {isCameraActive ? 'Ready for Recognition' : 'Camera is Off'}
                            </Text>
                            <Text className="text-sm text-white/80 text-center leading-5">
                                {isCameraActive
                                    ? 'Tap the button to begin signing and translation.'
                                    : 'Tap the button below to turn the camera on.'}
                            </Text>
                        </View>
                    )}

                    {/* Show current prediction overlay */}
                    {isTranslating && currentPrediction && (
                        <View className="absolute top-4 left-4 right-4 bg-black/70 rounded-lg p-3">
                            <Text className="text-white text-2xl font-bold text-center">
                                {currentPrediction}
                            </Text>
                            <Text className="text-white/70 text-sm text-center mt-1">
                                Confidence: {(confidence * 100).toFixed(1)}%
                            </Text>
                        </View>
                    )}

                    {/* Camera corners */}
                    <View className="absolute top-4 left-4 w-6 h-6 border-t-[3px] border-l-[3px] border-accent" />
                    <View className="absolute top-4 right-4 w-6 h-6 border-t-[3px] border-r-[3px] border-accent" />
                    <View className="absolute bottom-4 left-4 w-6 h-6 border-b-[3px] border-l-[3px] border-accent" />
                    <View className="absolute bottom-4 right-4 w-6 h-6 border-b-[3px] border-r-[3px] border-accent" />
                </View>

                {/* Control Buttons */}
                {isCameraActive ? (
                    <View className="flex-row justify-between items-center w-full px-10 mt-2">
                        <TouchableOpacity
                            onPress={stopCamera}
                            className="w-[60px] h-[60px] rounded-full justify-center items-center bg-red-600/70"
                        >
                            <MaterialIcons name="power-settings-new" size={28} color="white" />
                        </TouchableOpacity>

                        <TouchableOpacity
                            className={`
                                w-[80px] h-[80px] rounded-full justify-center items-center
                                shadow-lg shadow-black/30
                                ${isTranslating ? 'bg-highlight' : 'bg-accent'}
                            `}
                            onPress={toggleTranslation}
                        >
                            <MaterialIcons
                                name={isTranslating ? 'stop' : 'play-arrow'}
                                size={38}
                                color="white"
                            />
                        </TouchableOpacity>

                        <TouchableOpacity
                            onPress={flipCamera}
                            className="w-[60px] h-[60px] rounded-full justify-center items-center bg-black/40"
                        >
                            <MaterialIcons name="flip-camera-ios" size={26} color="white" />
                        </TouchableOpacity>
                    </View>
                ) : (
                    <View className="flex-row justify-center items-center mt-2">
                        <TouchableOpacity
                            className="w-[80px] h-[80px] rounded-full justify-center items-center bg-accent shadow-lg shadow-black/30"
                            onPress={toggleTranslation}
                        >
                            <MaterialIcons name="videocam" size={34} color="white" />
                        </TouchableOpacity>
                    </View>
                )}
            </View>

            {/* Translation Output */}
            <View className="px-5 pb-5">
                <Text className="text-base font-semibold text-primary mb-3">Output:</Text>
                <View className="bg-white rounded-xl p-5 min-h-[80px] border border-neutral/20 shadow-sm">
                    <Text className="text-lg text-primary leading-6 text-center">
                        {translatedText}
                    </Text>
                </View>

                {/* Status Indicator */}
                <View className="flex-row items-center justify-center mt-3">
                    <View
                        className={`w-2 h-2 rounded-full mr-2 
                            ${isTranslating && isConnected ? 'bg-accent' : isCameraActive ? 'bg-green-500' : 'bg-neutral'}
                        `}
                    />
                    <Text className="text-sm text-neutral font-medium">
                        {isTranslating && isConnected
                            ? 'TRANSLATING LIVE'
                            : isCameraActive
                            ? 'Camera Active'
                            : 'Idle'}
                    </Text>
                </View>
            </View>
        </View>
    );
}
