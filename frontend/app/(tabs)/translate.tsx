import React, { useState, useEffect, useRef } from "react";
import {
    View,
    Text,
    TouchableOpacity,
    StatusBar,
    ActivityIndicator,
    PanResponder,
    PanResponderGestureState,
    GestureResponderEvent,
    ImageBackground,
    StyleSheet,
    Modal, // 💡 Import Modal component
} from "react-native";
import { MaterialIcons } from "@expo/vector-icons";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { useFocusEffect } from "@react-navigation/native";
import { useRouter } from "expo-router";
import { useTheme } from "../../src/ThemeContext";
import AppHeader from "../../components/AppHeader";
import axios from "axios";
import Animated, { 
    FadeInUp, 
    useSharedValue, 
    useAnimatedStyle, 
    withTiming, 
    withRepeat, 
    Easing,
} from 'react-native-reanimated';

import {
    Camera,
    useCameraDevices,
    CameraDevice,
} from "react-native-vision-camera";

// --- AnimatedCorner Component ---
const AnimatedCorner = ({ isTranslating, borderStyle }: { isTranslating: boolean; borderStyle: any }) => {
    const opacity = useSharedValue(0);
    useEffect(() => {
        if (isTranslating) {
            opacity.value = withRepeat(
                withTiming(1, { duration: 800, easing: Easing.linear }),
                -1,
                true
            );
        } else {
            opacity.value = withTiming(0, { duration: 300 });
        }
    }, [isTranslating]);
    const animatedStyle = useAnimatedStyle(() => ({ opacity: opacity.value }));
    return (
        <Animated.View 
            style={[
                { width: 24, height: 24, borderColor: 'rgb(255,107,0)' }, 
                borderStyle, 
                animatedStyle
            ]} 
            className={`absolute border-accent`}
        />
    );
};
// ----------------------------------------------------

// 💡 New: Help Modal Component
const TranslationHelpModal = ({ isVisible, onClose, isDark }: { isVisible: boolean; onClose: () => void; isDark: boolean }) => {
    const modalBg = isDark ? "bg-darkbg/95" : "bg-white/95";
    const surfaceColor = isDark ? "bg-darksurface" : "bg-white";
    const textColor = isDark ? "text-secondary" : "text-primary";

    return (
        <Modal
            animationType="fade"
            transparent={true}
            visible={isVisible}
            onRequestClose={onClose}
        >
            <TouchableOpacity 
                className={`flex-1 justify-center items-center ${modalBg} p-8`}
                onPress={onClose} // Closes modal on tap anywhere
                activeOpacity={1}
            >
                <View 
                    className={`w-full rounded-2xl p-6 shadow-xl border border-accent ${surfaceColor}`}
                    style={{ maxHeight: '80%' }}
                >
                    <Text className={`text-2xl font-audiowide text-center mb-4 color-accent ${textColor}`}>
                        Translation Tips
                    </Text>
                    <View className="space-y-3">
                        <View className="flex-row items-start">
                            <Text className={`text-xl font-fredoka-medium mr-2 color-highlight ${textColor}`}>1.</Text>
                            <Text className={`flex-1 text-base font-montserrat-regular ${textColor}`}>
                                Frame your entire hand clearly within the camera view.
                            </Text>
                        </View>
                        <View className="flex-row items-start">
                            <Text className={`text-xl font-fredoka-medium mr-2 color-highlight ${textColor}`}>2.</Text>
                            <Text className={`flex-1 text-base font-montserrat-regular ${textColor}`}>
                                Hold the sign steady for at least 1 second for accurate recognition.
                            </Text>
                        </View>
                        <View className="flex-row items-start">
                            <Text className={`text-xl font-fredoka-medium mr-2 color-highlight ${textColor}`}>3.</Text>
                            <Text className={`flex-1 text-base font-montserrat-regular ${textColor}`}>
                                Use the Zoom feature for signs that are farther away.
                            </Text>
                        </View>
                        <View className="flex-row items-start">
                            <Text className={`text-xl font-fredoka-medium mr-2 color-highlight ${textColor}`}>4.</Text>
                            <Text className={`flex-1 text-base font-montserrat-regular ${textColor}`}>
                                Ensure good lighting or use the Flash in dark environments.
                            </Text>
                        </View>
                    </View>
                    <TouchableOpacity onPress={onClose} className="mt-6 p-2 rounded-full bg-accent/20 self-center">
                        <Text className="text-accent text-center font-fredoka-bold">Got it!</Text>
                    </TouchableOpacity>
                </View>
            </TouchableOpacity>
        </Modal>
    );
};


export default function Translate() {
    // 🧠 States
    const [isCameraActive, setIsCameraActive] = useState(false);
    const [isTranslating, setIsTranslating] = useState(false);
    const [translatedText, setTranslatedText] = useState("Camera Off. Tap to begin.");
    const [hasPermission, setHasPermission] = useState(false);
    const [isSending, setIsSending] = useState(false);
    const [prediction, setPrediction] = useState<string>("None");
    const [facing, setFacing] = useState<"front" | "back">("back");
    const [flash, setFlash] = useState<"on" | "off">("off");
    
    // Zoom state (default 1x)
    const [zoom, setZoom] = useState(1);
    
    // UI/UX Updates
    const [lastHandDetectionTime, setLastHandDetectionTime] = useState(Date.now());
    const [showGuidanceOverlay, setShowGuidanceOverlay] = useState(false);
    
    // 💡 New State: Modal visibility
    const [isHelpModalVisible, setIsHelpModalVisible] = useState(false);

    const cameraRef = useRef<Camera>(null);
    const devices = useCameraDevices();
    const device: CameraDevice | undefined =
        devices.find((d) => d.position === facing) ??
        devices.find((d) => d.position === "back");

    const insets = useSafeAreaInsets();
    const router = useRouter();
    const { isDark } = useTheme();

    // 🆕 Function to cycle zoom levels (1x -> 5x -> Max -> 1x)
    const cycleZoom = () => {
        if (!device) return;

        const currentZoom = zoom;
        let nextZoom = device.minZoom; // Default next step is 1x

        if (currentZoom < 3) {
            // If current is near 1x, go to 5x (or max if max is less than 5)
            nextZoom = Math.min(5, device.maxZoom); 
        } else if (currentZoom >= 3 && currentZoom < device.maxZoom) {
            // If current is near 5x, go to max zoom
            nextZoom = device.maxZoom;
        }
        
        setZoom(nextZoom); // Set the new zoom level
    };

    // 🪄 Simplified PanResponder for Swipe Navigation only
    const panResponder = useRef(
        PanResponder.create({
            // Allow responder if it's a drag (swipe)
            onStartShouldSetPanResponder: () => true,
            onMoveShouldSetPanResponder: (_, g) => Math.abs(g.dx) > 10,

            onPanResponderGrant: () => {},
            onPanResponderMove: () => {},

            onPanResponderRelease: (_, g) => {
                // Perform swipe navigation if significant horizontal drag to the left
                if (g.dx < -30) {
                    router.push("/compose");
                }
            },
        })
    ).current;

    // 🔐 Request camera permission 
    useEffect(() => {
        (async () => {
            const status = await Camera.requestCameraPermission();
            setHasPermission(status === "granted");
        })();
    }, []);

    // ➡️ useFocusEffect: Reset states on navigation
    useFocusEffect(
        React.useCallback(() => {
            setIsCameraActive(false);
            setIsTranslating(false);
            setTranslatedText("Camera Off. Tap to begin.");
            setZoom(1); 
            return () => {
                setIsCameraActive(false);
                setIsTranslating(false);
            };
        }, [])
    );

    // 📸 Send frames every 200ms
    useEffect(() => { 
        if (!isTranslating || !cameraRef.current) return;

        const interval = setInterval(async () => {
            if (isSending) return;
            setIsSending(true);
            try {
                // Placeholder logic for sending frame
                const photo = await cameraRef.current!.takePhoto({});
                const uri = photo.path.startsWith("file://")
                    ? photo.path
                    : `file://${photo.path}`;
                const formData = new FormData();
                formData.append("file", {
                    uri,
                    type: "image/jpeg",
                    name: "frame.jpg",
                } as any);

                const res = await axios.post("http://192.168.130.136:8000/predict", formData, { 
                    headers: { "Content-Type": "multipart/form-data" },
                });

                if (res.data.prediction) {
                    setPrediction(res.data.prediction);
                    if (res.data.prediction !== "None") {
                        setTranslatedText(`Detected sign: ${res.data.prediction.toUpperCase()}`);
                        setLastHandDetectionTime(Date.now()); 
                    } else {
                        setTranslatedText("No hand detected...");
                    }
                }
            } catch (err) {
                console.log("Error sending frame:", err);
            }

            setIsSending(false);
        }, 200);

        return () => clearInterval(interval);
    }, [isTranslating, isSending]);

    // 2. Guidance Overlay Logic
    useEffect(() => { 
        let guidanceInterval: number | undefined; 
        
        if (isTranslating) {
            guidanceInterval = setInterval(() => {
                const timeElapsed = Date.now() - lastHandDetectionTime;
                const threshold = 3000; // 3 seconds

                if (timeElapsed > threshold) {
                    setShowGuidanceOverlay(true);
                } else {
                    setShowGuidanceOverlay(false);
                }
            }, 500);
        } else {
            setShowGuidanceOverlay(false);
        }

        return () => {
            if (guidanceInterval) clearInterval(guidanceInterval);
        };
    }, [isTranslating, lastHandDetectionTime]);

    // --- Control Functions ---
    const toggleCamera = async () => { 
        if (!hasPermission) {
            const status = await Camera.requestCameraPermission();
            setHasPermission(status === "granted");
            return;
        }
        if (!isCameraActive) {
            setIsCameraActive(true);
            setTranslatedText("Camera On. Tap again to start recognition.");
            return;
        }

        setIsTranslating((prev) => !prev);
        if (!isTranslating) {
            setTranslatedText("Recognizing signs...");
            setLastHandDetectionTime(Date.now()); 
        } else {
            setTranslatedText("Recognition paused. Tap to continue.");
        }
    };

    const stopCamera = () => { 
        setIsCameraActive(false);
        setIsTranslating(false);
        setTranslatedText("Camera Off. Tap to begin.");
        setZoom(1); // Reset zoom on stop
    };

    const flipCamera = () => setFacing(facing === "back" ? "front" : "back");
    const toggleFlash = () => setFlash(flash === "off" ? "on" : "off");
    // --- End Control Functions ---

    // 🧠 Theme colors
    const bgColor = isDark ? "bg-darkbg" : "bg-secondary";
    const textColor = isDark ? "text-secondary" : "text-primary";
    const surfaceColor = isDark ? "bg-darksurface" : "bg-white";

    // 🪟 Permission / loading state
    if (!hasPermission || !device) {
        return (
            <View className={`flex-1 justify-center items-center ${bgColor}`}>
                <ActivityIndicator size="large" color="rgb(255,107,0)" />
                <Text className={`${textColor} mt-4 font-fredoka-medium`}>
                    {hasPermission ? "Loading cameras..." : "Requesting camera permission..."}
                </Text>
            </View>
        );
    }

    // Display zoom value
    const maxZoomFactor = device.maxZoom;
    const zoomText = `Zoom: ${zoom.toFixed(1)}x`;


    return (
        <View className={`flex-1 ${bgColor}`}>
            <ImageBackground
                source={require('../../assets/images/MainBG.png')}
                className="flex-1"
                resizeMode="cover"
            >
                <View
                    {...panResponder.panHandlers} // Pan responder applied to the whole screen view
                    className="flex-1"
                    style={{ paddingTop: insets.top }}
                >
                    <StatusBar barStyle={isDark ? "light-content" : "dark-content"} />

                    <Animated.View
                        entering={FadeInUp.duration(600).delay(200)}
                        className="px-5 pt-5 items-center"
                    >
                        {/* 1. 💡 New Feature: Small title above the camera */}
                        <Text className={`text-2xl font-audiowide mb-6 ${textColor}`}>
                            GesturiX Translator
                        </Text>
                        
                        {/* Camera container view */}
                        <View
                            className={`w-full aspect-[1/1] ${
                                isDark ? "bg-darkhover" : "bg-primary"
                            } border border-accent rounded-2xl overflow-hidden mb-5 relative`}
                        >
                            {isCameraActive && (
                                <Camera
                                    ref={cameraRef}
                                    style={{ flex: 1 }}
                                    device={device}
                                    isActive={true}
                                    photo={true}
                                    torch={facing === "back" ? flash : "off"}
                                    zoom={zoom} 
                                />
                            )}
                            
                            {/* Zoom Text Overlay (NOW A TOUCHABLE) */}
                            {isCameraActive && (
                                <TouchableOpacity 
                                    onPress={cycleZoom} // <-- Cycle Zoom on Tap
                                    className="absolute top-6 left-6 p-2 bg-black/30 rounded-xl z-50"
                                >
                                    <Text className="text-white text-sm font-fredoka-medium">
                                        {zoomText}
                                    </Text>
                                </TouchableOpacity>
                            )}

                            {/* Flip / Flash Buttons */}
                            {isCameraActive && (
                                <View className="absolute top-6 right-6 flex-row space-x-2 bg-black/30 rounded-xl p-1 z-50">
                                    {facing === "back" && (
                                        <TouchableOpacity onPress={toggleFlash} className="p-2">
                                            <MaterialIcons
                                                name={flash === "on" ? "flash-on" : "flash-off"}
                                                size={24}
                                                color="white"
                                            />
                                        </TouchableOpacity>
                                    )}
                                    <TouchableOpacity onPress={flipCamera} className="p-2">
                                        <MaterialIcons name="flip-camera-ios" size={24} color="white" />
                                    </TouchableOpacity>
                                </View>
                            )}

                            {/* Overlays */}
                            {isTranslating && showGuidanceOverlay && (
                                <View className="absolute inset-0 justify-center items-center px-5 bg-black/50 z-40">
                                    <MaterialIcons name="pan-tool" size={60} color="white" />
                                    <Text className="text-xl font-audiowide text-white mt-4 mb-2 text-center">Place hands in camera!</Text>
                                </View>
                            )}
                            {(!isCameraActive || !isTranslating) && (
                                <View className="absolute inset-0 justify-center items-center px-5 bg-black/50 z-40">
                                    <MaterialIcons name={isCameraActive ? "pause-circle-outline" : "videocam-off"} size={80} color="white" />
                                    <Text className="text-xl font-audiowide text-white mt-4 mb-2 text-center">
                                        {isCameraActive ? "Ready for Recognition" : "Camera is Off"}
                                    </Text>
                                </View>
                            )}

                            {/* Corners */}
                            <AnimatedCorner isTranslating={isTranslating} borderStyle={{ top: 16, left: 16, borderTopWidth: 3, borderLeftWidth: 3 }} />
                            <AnimatedCorner isTranslating={isTranslating} borderStyle={{ top: 16, right: 16, borderTopWidth: 3, borderRightWidth: 3 }} />
                            <AnimatedCorner isTranslating={isTranslating} borderStyle={{ bottom: 16, left: 16, borderBottomWidth: 3, borderLeftWidth: 3 }} />
                            <AnimatedCorner isTranslating={isTranslating} borderStyle={{ bottom: 16, right: 16, borderBottomWidth: 3, borderRightWidth: 3 }} />
                        </View>

                        {/* Buttons (Fixed) */}
                        {isCameraActive ? (
                            <View className="flex-row justify-between items-center w-full px-10 mt-2">
                                <TouchableOpacity
                                    onPress={stopCamera}
                                    className="w-[60px] h-[60px] rounded-full justify-center items-center bg-red-600/70"
                                >
                                    <MaterialIcons name="power-settings-new" size={28} color="white" />
                                </TouchableOpacity>

                                <TouchableOpacity
                                    className={`w-[80px] h-[80px] rounded-full justify-center items-center shadow-lg shadow-black/30 ${
                                        isTranslating ? "bg-highlight" : "bg-accent"
                                    }`}
                                    onPress={toggleCamera}
                                >
                                    <MaterialIcons
                                        name={isTranslating ? "stop" : "play-arrow"}
                                        size={38}
                                        color="white"
                                    />
                                </TouchableOpacity>

                                <View className="w-[60px]" />
                            </View>
                        ) : (
                            <View className="flex-row justify-center items-center mt-2">
                                <TouchableOpacity
                                    className="w-[80px] h-[80px] rounded-full justify-center items-center bg-accent shadow-lg shadow-black/30"
                                    onPress={toggleCamera}
                                >
                                    <MaterialIcons name="videocam" size={34} color="white" />
                                </TouchableOpacity>
                            </View>
                        )}

                    </Animated.View>

                    {/* Output */}
                    <Animated.View entering={FadeInUp.duration(600).delay(400)} className="px-5 pb-5 mt-5">
                        <Text className={`text-base font-audiowide mb-3 ${textColor}`}>Output:</Text>
                        <View className={`rounded-xl p-5 min-h-[80px] border border-accent shadow-sm ${surfaceColor}`}>
                            <Text className={`text-lg text-center leading-6 font-montserrat-bold ${textColor}`}>
                                {translatedText}
                            </Text>
                        </View>
                        
                        {/* 1. MOVED AND RESTYLED HELP BUTTON */}
{/* 1. MOVED AND RESTYLED HELP BUTTON - Layout adjusted for centering */}
                        <View className="relative flex-row items-center justify-center mt-3">
                            
                            {/* Status Indicator (Centered Content) */}
                            <View className="flex-row items-center">
                                <View className={`w-2 h-2 rounded-full mr-2 ${isTranslating ? "bg-accent" : isCameraActive ? "bg-green-500" : "bg-neutral"}`} />
                                <Text className={`text-sm font-fredoka-medium ${isDark ? "text-secondary" : "text-neutral"}`}>
                                    {isTranslating ? "TRANSLATING LIVE" : isCameraActive ? "Camera Active" : "Idle"}
                                </Text>
                            </View>
                            
                            {/* Tips button (Absolutely Positioned to the Right) */}
                            <TouchableOpacity
                                onPress={() => setIsHelpModalVisible(true)}
                                // Place the button at the end of the line, even though the content is centered
                                className="absolute right-0 top-1 p-2 bg-accent/20 rounded-xl"
                            >
                                <MaterialIcons name="help-outline" size={26} color="rgb(255,107,0)" />
                            </TouchableOpacity>
                        </View>
                        
                    </Animated.View>
                </View>

            </ImageBackground>
            
            {/* 3. 💡 New Feature: Simple Modal with Translation Tips */}
            <TranslationHelpModal 
                isVisible={isHelpModalVisible} 
                onClose={() => setIsHelpModalVisible(false)} 
                isDark={isDark} 
            />

        </View>
    );
}

// Optional: Add a simple StyleSheet if needed, though Tailwind-rn is used.
// const styles = StyleSheet.create({});