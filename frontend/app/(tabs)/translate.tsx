import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StatusBar,
  ActivityIndicator,
  PanResponder,
} from "react-native";
import { MaterialIcons } from "@expo/vector-icons";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { useFocusEffect } from "@react-navigation/native";
import { useRouter } from "expo-router";
import { useTheme } from "../../src/ThemeContext";
import AppHeader from "../../components/AppHeader";
import axios from "axios";

import {
  Camera,
  useCameraDevices,
  CameraDevice,
} from "react-native-vision-camera";

export default function Translate() {
  // 🧠 States
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [translatedText, setTranslatedText] = useState("Camera Off. Tap to begin.");
  const [hasPermission, setHasPermission] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [prediction, setPrediction] = useState<string>("None");

  const cameraRef = useRef<Camera>(null);
  const devices = useCameraDevices();
  const device: CameraDevice | undefined =
    devices.find((d) => d.position === "front") ??
    devices.find((d) => d.position === "back");

  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { isDark } = useTheme();

  // 🪄 Swipe navigation
  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (_, g) => Math.abs(g.dx) > 10,
      onPanResponderRelease: (_, g) => {
        if (g.dx < -30) router.push("/compose");
      },
    })
  ).current;

  // 🔐 Request camera permission
  useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission();
      console.log("Camera permission:", status);
      setHasPermission(status === "granted");
    })();
  }, []);

  useFocusEffect(
    React.useCallback(() => {
      setIsCameraActive(false);
      setIsTranslating(false);
      setTranslatedText("Camera Off. Tap to begin.");
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

        const res = await axios.post("http://192.168.123.137:8000/predict", formData, { //ilisi ni if necessary, akoa rani i auto detect later
          headers: { "Content-Type": "multipart/form-data" },
        });

        if (res.data.prediction) {
          setPrediction(res.data.prediction);

          // 🟢 CHANGED — update translated text on-screen
          if (res.data.prediction !== "None") {
            setTranslatedText(`Detected sign: ${res.data.prediction.toUpperCase()}`);
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
    } else {
      setTranslatedText("Recognition paused. Tap to continue.");
    }
  };

  const stopCamera = () => {
    setIsCameraActive(false);
    setIsTranslating(false);
    setTranslatedText("Camera Off. Tap to begin.");
  };

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

  return (
    <View
      {...panResponder.panHandlers}
      className={`flex-1 ${bgColor}`}
      style={{ paddingTop: insets.top }}
    >
      <StatusBar barStyle={isDark ? "light-content" : "dark-content"} />
      <AppHeader />

      {/* Camera View */}
      <View className="px-5 pt-5 items-center">
        <View
          className={`w-full aspect-[4/3] ${
            isDark ? "bg-darkhover" : "bg-primary"
          } rounded-2xl overflow-hidden mb-5 relative`}
        >
          {isCameraActive && (
            <Camera
              ref={cameraRef}
              style={{ flex: 1 }}
              device={device}
              isActive={true}
              photo={true}
            />
          )}

          {/* Overlay */}
          {(!isCameraActive || !isTranslating) && (
            <View className="absolute inset-0 justify-center items-center px-5 bg-black/50">
              <MaterialIcons
                name={isCameraActive ? "pause-circle-outline" : "videocam-off"}
                size={80}
                color="white"
              />
              <Text className="text-xl font-audiowide text-white mt-4 mb-2 text-center">
                {isCameraActive ? "Ready for Recognition" : "Camera is Off"}
              </Text>
              <Text className="text-sm text-white/80 text-center leading-5 font-fredoka">
                {isCameraActive
                  ? "Tap the button to begin signing and translation."
                  : "Tap the button below to turn the camera on."}
              </Text>
            </View>
          )}

          {/* Corners */}
          <View className="absolute top-4 left-4 w-6 h-6 border-t-[3px] border-l-[3px] border-accent" />
          <View className="absolute top-4 right-4 w-6 h-6 border-t-[3px] border-r-[3px] border-accent" />
          <View className="absolute bottom-4 left-4 w-6 h-6 border-b-[3px] border-l-[3px] border-accent" />
          <View className="absolute bottom-4 right-4 w-6 h-6 border-b-[3px] border-r-[3px] border-accent" />
        </View>

        {/* Buttons */}
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
      </View>

      {/* Output */}
      <View className="px-5 pb-5">
        <Text className={`text-base font-audiowide mb-3 ${textColor}`}>Output:</Text>
        <View
          className={`rounded-xl p-5 min-h-[80px] border border-accent shadow-sm ${surfaceColor}`}
        >
          {/* 🟢 CHANGED — show live prediction */}
          <Text
            className={`text-lg text-center leading-6 font-montserrat-bold ${textColor}`}
          >
            {translatedText}
          </Text>
        </View>

        <View className="flex-row items-center justify-center mt-3">
          <View
            className={`w-2 h-2 rounded-full mr-2 ${
              isTranslating
                ? "bg-accent"
                : isCameraActive
                ? "bg-green-500"
                : "bg-neutral"
            }`}
          />
          <Text
            className={`text-sm font-fredoka-medium ${
              isDark ? "text-secondary" : "text-neutral"
            }`}
          >
            {isTranslating
              ? "TRANSLATING LIVE"
              : isCameraActive
              ? "Camera Active"
              : "Idle"}
          </Text>
        </View>
      </View>
    </View>
  );
}