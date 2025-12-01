import React, { useState, useEffect, useRef } from "react";
import {
    View,
    Text,
    TouchableOpacity,
    StatusBar,
    ActivityIndicator,
    PanResponder,
    ScrollView,
    PanResponderGestureState,
    GestureResponderEvent,
    ImageBackground,
    StyleSheet,
    Modal,
    Alert, // 💡 Added Alert for the placeholder
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
    useCameraFormat,
} from "react-native-vision-camera";
import { ENDPOINTS } from "../../constants/ApiConfig";

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

// --- TranslationHelpModal Component ---
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
                onPress={onClose}
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
    const [isCameraActive, setIsCameraActive] = useState(true); // Default True
    const [isTranslating, setIsTranslating] = useState(false);
    const [statusMessage, setStatusMessage] = useState("Tap Play to begin recognition."); // Simplified message
    
    // Output States
    const [currentTranslation, setCurrentTranslation] = useState("");
    const [enhancedTranslation, setEnhancedTranslation] = useState(""); 
    const [isEnhancing, setIsEnhancing] = useState(false); 
    const [lastTranslatedLetter, setLastTranslatedLetter] = useState<string | null>(null);

    const [hasPermission, setHasPermission] = useState(false);
    const [isSending, setIsSending] = useState(false);
    const [prediction, setPrediction] = useState<string>("None");
    const [facing, setFacing] = useState<"front" | "back">("back");
    const [flash, setFlash] = useState<"on" | "off">("off");
    
    // Zoom state
    const [zoom, setZoom] = useState(1);
    
    // UI/UX Updates
    const [lastHandDetectionTime, setLastHandDetectionTime] = useState(Date.now());
    const [showGuidanceOverlay, setShowGuidanceOverlay] = useState(false);
    const [isHelpModalVisible, setIsHelpModalVisible] = useState(false);

    const cameraRef = useRef<Camera>(null);
    const devices = useCameraDevices();
    const device: CameraDevice | undefined =
        devices.find((d) => d.position === facing) ??
        devices.find((d) => d.position === "back");

    const format = useCameraFormat(device, [
        { photoResolution: { width: 640, height: 480 } },
        { fps: 30 }
    ]);

    const insets = useSafeAreaInsets();
    const router = useRouter();
    const { isDark } = useTheme();

    const cycleZoom = () => {
        if (!device) return;
        const currentZoom = zoom;
        let nextZoom = device.minZoom;

        if (currentZoom < 3) {
            nextZoom = Math.min(5, device.maxZoom); 
        } else if (currentZoom >= 3 && currentZoom < device.maxZoom) {
            nextZoom = device.maxZoom;
        }
        setZoom(nextZoom);
    };

    const panResponder = useRef(
        PanResponder.create({
            onStartShouldSetPanResponder: () => true,
            onMoveShouldSetPanResponder: (_, g) => Math.abs(g.dx) > 10,
            onPanResponderGrant: () => {},
            onPanResponderMove: () => {},
            onPanResponderRelease: (_, g) => {
                if (g.dx < -30) {
                    router.push("/learn");
                }
            },
        })
    ).current;

    useEffect(() => {
        (async () => {
            const status = await Camera.requestCameraPermission();
            setHasPermission(status === "granted");
        })();
    }, []);

    useFocusEffect(
        React.useCallback(() => {
            setIsCameraActive(true); // 💡 Force camera ON when entering tab
            setIsTranslating(false);
            setStatusMessage("Tap Play to begin recognition.");
            setCurrentTranslation(""); 
            setEnhancedTranslation("");
            setLastTranslatedLetter(null); 
            setZoom(1); 
            return () => {
                setIsCameraActive(false); // Turn off when leaving tab to save battery
                setIsTranslating(false);
            };
        }, [])
    );

    useEffect(() => { 
        if (!isTranslating || !cameraRef.current) return;

        const interval = setInterval(async () => {
            if (isSending) return; 

            setIsSending(true);
            try {
                const photo = await cameraRef.current!.takePhoto({
                    flash: 'off',
                    enableShutterSound: false, 
                });

                const uri = photo.path.startsWith("file://")
                    ? photo.path
                    : `file://${photo.path}`;
                
                const formData = new FormData();
                formData.append("file", {
                    uri,
                    type: "image/jpeg",
                    name: "frame.jpg",
                } as any)

                const res = await axios.post(ENDPOINTS.PREDICT, formData, { 
                    headers: { "Content-Type": "multipart/form-data" },
                });

                if (res.data.prediction) {
                    const newPrediction = res.data.prediction.toUpperCase();
                    setPrediction(newPrediction);
                    
                    const isLetter = newPrediction.length === 1 && newPrediction.match(/[A-Z]/);
                    
                    if (isLetter) {
                        setStatusMessage(`Detected sign: ${newPrediction}`);
                        setLastHandDetectionTime(Date.now());
                        
                        if (newPrediction !== lastTranslatedLetter) {
                            setCurrentTranslation(prev => prev + newPrediction);
                            setLastTranslatedLetter(newPrediction);
                        }
                    } else if (newPrediction.toUpperCase() === "NONE") {
                        setStatusMessage("No hand detected...");
                        setLastTranslatedLetter(null);
                    } else if (newPrediction.toUpperCase() === "SPACE") {
                        setStatusMessage("Detected: SPACE");
                        setLastHandDetectionTime(Date.now());
                        
                        if (currentTranslation.slice(-1) !== ' ') {
                             setCurrentTranslation(prev => prev + ' ');
                        }
                        setLastTranslatedLetter(null); 
                    } else {
                           setStatusMessage(`Model prediction: ${newPrediction}`);
                    }
                }
            } catch (err) {
                console.log("Error sending frame:", err);
            }

            setIsSending(false);
        }, 100);

        return () => clearInterval(interval);
    }, [isTranslating, isSending, lastTranslatedLetter, currentTranslation]);

    useEffect(() => { 
        let guidanceInterval: number | undefined;
        
        if (isTranslating) {
            guidanceInterval = setInterval(() => {
                const timeElapsed = Date.now() - lastHandDetectionTime;
                const threshold = 3000;

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

    const enhanceTranslationWithAI = async () => {
        if (!currentTranslation || isEnhancing) return;
        
        const rawTextToEnhance = currentTranslation.trim().replace(/\s+/g, ' '); 
        
        if (rawTextToEnhance.length < 2) {
            setEnhancedTranslation("Cannot enhance: Translation is too short.");
            return;
        }

        setIsEnhancing(true);
        setEnhancedTranslation("AI is enhancing translation...");
        try {
            const response = await axios.post(ENDPOINTS.ENHANCE, {
                raw_text: rawTextToEnhance,
            });

            if (response.data.enhanced_text) {
                setEnhancedTranslation(response.data.enhanced_text);
            } else {
                setEnhancedTranslation("AI enhancement failed to return text.");
            }
        } catch (error) {
            console.error("Error calling AI enhancement:", error);
            setEnhancedTranslation("AI connection error. Check server logs.");
        } finally {
            setIsEnhancing(false);
        }
    };

    // 💡 Renamed from toggleCamera to toggleTranslation since camera is always on
    const toggleTranslation = () => { 
        if (!hasPermission) {
            Alert.alert("Permission needed", "Camera permission is required.");
            return;
        }

        setIsTranslating((prev) => !prev);
        if (!isTranslating) { // Logic for STARTING translation
            setStatusMessage("Recognizing signs...");
            setLastHandDetectionTime(Date.now()); 
        } else { // Logic for STOPPING translation
            setStatusMessage("Recognition paused. Tap to continue.");
        }
    };

    const flipCamera = () => setFacing(facing === "back" ? "front" : "back");
    const toggleFlash = () => setFlash(flash === "off" ? "on" : "off");

    const bgColor = isDark ? "bg-darkbg" : "bg-secondary";
    const textColor = isDark ? "text-secondary" : "text-primary";
    const surfaceColor = isDark ? "bg-darksurface" : "bg-white";

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

    const maxZoomFactor = device.maxZoom;
    const zoomText = `Zoom: ${zoom.toFixed(1)}x`;

    return (
        <View className={`flex-1 ${bgColor}`}>
            <ImageBackground
                source={require('../../assets/images/MainBG.png')}
                className="flex-1"
                resizeMode="cover"
            >
                <ScrollView
                    {...panResponder.panHandlers}
                    className="flex-1"
                    style={{ paddingTop: insets.top }}
                    contentContainerStyle={{ paddingBottom: 160 }}
                >
                    <StatusBar barStyle={isDark ? "light-content" : "dark-content"} />

                    <Animated.View
                        entering={FadeInUp.duration(600).delay(200)}
                        className="px-5 pt-5 items-center"
                    >
                        <Text className={`text-2xl font-audiowide mb-6 ${textColor}`}>
                            GesturiX Translator
                        </Text>
                        
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
                                    format={format}
                                    isActive={true}
                                    photo={true}
                                    torch={facing === "back" ? flash : "off"}
                                    zoom={zoom} 
                                />
                            )}
                            
                            {isCameraActive && (
                                <TouchableOpacity 
                                    onPress={cycleZoom}
                                    className="absolute top-6 left-6 p-2 bg-black/30 rounded-xl z-50"
                                >
                                    <Text className="text-white text-sm font-fredoka-medium">
                                        {zoomText}
                                    </Text>
                                </TouchableOpacity>
                            )}

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

                            <AnimatedCorner isTranslating={isTranslating} borderStyle={{ top: 16, left: 16, borderTopWidth: 3, borderLeftWidth: 3 }} />
                            <AnimatedCorner isTranslating={isTranslating} borderStyle={{ top: 16, right: 16, borderTopWidth: 3, borderRightWidth: 3 }} />
                            <AnimatedCorner isTranslating={isTranslating} borderStyle={{ bottom: 16, left: 16, borderBottomWidth: 3, borderLeftWidth: 3 }} />
                            <AnimatedCorner isTranslating={isTranslating} borderStyle={{ bottom: 16, right: 16, borderBottomWidth: 3, borderRightWidth: 3 }} />
                        </View>

                        {/* Controls Container */}
                        <View className="flex-row justify-between items-center w-full px-10 mt-2">
                            
                            {/* 💡 LEFT: Text-to-Speech Placeholder (Replaces Off Button) */}
                            <TouchableOpacity
                                onPress={() => Alert.alert("Coming Soon UwU", "Text-to-Speech feature will be available here!")}
                                // Disabled look if no text, active look if text exists
                                className={`w-[60px] h-[60px] rounded-full justify-center items-center ${
                                    (currentTranslation.length > 0 || enhancedTranslation.length > 0) 
                                    ? "bg-sky-500/80" 
                                    : "bg-neutral-500/50"
                                }`}
                            >
                                <MaterialIcons name="record-voice-over" size={28} color="white" />
                            </TouchableOpacity>

                            {/* CENTER: Play/Pause Button */}
                            <TouchableOpacity
                                className={`w-[80px] h-[80px] rounded-full justify-center items-center shadow-lg shadow-black/30 ${
                                    isTranslating ? "bg-highlight" : "bg-accent"
                                }`}
                                onPress={toggleTranslation}
                            >
                                <MaterialIcons
                                    name={isTranslating ? "stop" : "play-arrow"}
                                    size={38}
                                    color="white"
                                />
                            </TouchableOpacity>

                            {/* RIGHT: AI Enhance Button */}
                            <TouchableOpacity
                                onPress={enhanceTranslationWithAI}
                                disabled={!currentTranslation || isEnhancing}
                                className={`w-[60px] h-[60px] rounded-full justify-center items-center shadow-lg ${
                                    !currentTranslation || isEnhancing
                                        ? "bg-neutral-500/50" 
                                        : "bg-highlight/70"
                                }`}
                            >
                                {isEnhancing ? (
                                    <ActivityIndicator color="white" size="small" />
                                ) : (
                                    <MaterialIcons name="auto-fix-high" size={28} color="white" />
                                )}
                            </TouchableOpacity>
                        </View>

                    </Animated.View>

                    {/* Output */}
                    <Animated.View entering={FadeInUp.duration(600).delay(400)} className="px-5 pb-5 mt-5">
                        <Text className={`text-base font-audiowide mb-3 ${textColor}`}>Raw Translation:</Text>
                        
                        <View className={`w-full rounded-xl p-5 min-h-[40px] border border-accent shadow-sm ${surfaceColor}`}>
                            <Text className={`text-xl text-center leading-6 font-montserrat-bold ${textColor}`}>
                                {currentTranslation.length > 0 ? currentTranslation : statusMessage}
                            </Text>
                        </View>
                        
                        {enhancedTranslation.length > 0 && (
                             <View className={`w-full rounded-xl p-5 pt-8 mt-4 min-h-[40px] border border-highlight shadow-md ${surfaceColor}`}>
                                <Text className={`text-base font-audiowide mb-2 ${textColor}`}>Enhanced Sentence:</Text>
                                <Text className={`text-xl text-center leading-7 font-montserrat-bold text-highlight`}>
                                    {enhancedTranslation}
                                </Text>
                            </View>
                        )}
                        
                        <View className="relative flex-row items-center justify-center mt-3">
                            <View className="flex-row items-center">
                                <View className={`w-2 h-2 rounded-full mr-2 ${isTranslating ? "bg-accent" : isCameraActive ? "bg-green-500" : "bg-neutral"}`} />
                                <Text className={`text-sm font-fredoka-medium ${isDark ? "text-secondary" : "text-neutral"}`}>
                                    {isTranslating ? "TRANSLATING LIVE" : "Ready"}
                                </Text>
                            </View>
                            
                            <TouchableOpacity
                                onPress={() => setIsHelpModalVisible(true)}
                                className="absolute right-0 top-1 p-2 bg-accent/20 rounded-xl"
                            >
                                <MaterialIcons name="help-outline" size={26} color="rgb(255,107,0)" />
                            </TouchableOpacity>
                        </View>
                        
                    </Animated.View>
                </ScrollView>

                <TranslationHelpModal 
                    isVisible={isHelpModalVisible} 
                    onClose={() => setIsHelpModalVisible(false)} 
                    isDark={isDark} 
                />

            </ImageBackground>
        </View>
    );
}