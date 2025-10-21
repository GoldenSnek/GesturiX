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
import AppHeader from '../../components/AppHeader'; 

export default function Translate() {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [facing, setFacing] = useState<CameraType>('front');
  const [isTranslating, setIsTranslating] = useState(false); 
  const [translatedText, setTranslatedText] = useState('Camera Off. Tap to begin.');
  
  const [permission, requestPermission] = useCameraPermissions();
  const insets = useSafeAreaInsets();
  const router = useRouter();

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (_, gestureState) =>
        Math.abs(gestureState.dx) > 10,
      onPanResponderRelease: (_, gestureState) => {
        if (gestureState.dx < -30) router.push('/compose');
      },
    })
  ).current;

  useEffect(() => {
    if (!permission?.granted) requestPermission();
  }, []);

  useFocusEffect(
    React.useCallback(() => {
      setIsCameraActive(false);
      setIsTranslating(false);
      setTranslatedText('Camera Off. Tap to begin.');
      return () => {
        setIsCameraActive(false);
        setIsTranslating(false);
      };
    }, [])
  );

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

    setIsTranslating(prev => !prev);
    if (!isTranslating) {
      setTranslatedText('Recognizing signs...');
      setTimeout(() => {
        setTranslatedText('Hello, how are you today?');
      }, 3000);
    } else {
      setTranslatedText('Recognition paused. Tap to continue.');
    }
  };

  const flipCamera = () => setFacing(prev => (prev === 'back' ? 'front' : 'back'));
  const stopCamera = () => {
    setIsCameraActive(false);
    setIsTranslating(false);
    setTranslatedText('Camera Off. Tap to begin.');
  };

  if (!permission) {
    return (
      <View className="flex-1 justify-center items-center bg-secondary">
        <ActivityIndicator size="large" color="#FF6B00" />
        <Text className="text-primary mt-4 font-fredoka-medium">
          Loading camera permissions...
        </Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View className="flex-1 justify-center items-center p-8 bg-secondary" style={{ paddingTop: insets.top }}>
        <Text className="text-primary text-xl font-audiowide mb-4 text-center">
          Camera Access Required
        </Text>
        <Text className="text-neutral text-center mb-6 font-fredoka">
          GesturiX needs access to your camera to recognize sign language.
        </Text>
        <TouchableOpacity 
          className="bg-accent rounded-full px-6 py-3 shadow-md"
          onPress={requestPermission}
        >
          <Text className="text-white font-fredoka-semibold text-lg">
            Grant Camera Permission
          </Text>
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
          {isCameraActive && <CameraView style={{ flex: 1 }} facing={facing} />}

          {/* Overlay */}
          {(!isCameraActive || !isTranslating) && (
            <View className="absolute inset-0 justify-center items-center px-5 bg-black/50">
              <MaterialIcons 
                name={isCameraActive ? "pause-circle-outline" : "videocam-off"} 
                size={80} 
                color="white" 
              />
              <Text className="text-xl font-audiowide text-white mt-4 mb-2 text-center">
                {isCameraActive ? 'Ready for Recognition' : 'Camera is Off'}
              </Text>
              <Text className="text-sm text-white/80 text-center leading-5 font-fredoka">
                {isCameraActive
                  ? 'Tap the button to begin signing and translation.'
                  : 'Tap the button below to turn the camera on.'}
              </Text>
            </View>
          )}

          {/* Overlay corners */}
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
              className={`w-[80px] h-[80px] rounded-full justify-center items-center shadow-lg shadow-black/30
                ${isTranslating ? 'bg-highlight' : 'bg-accent'}`}
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
        <Text className="text-base font-audiowide text-primary mb-3">
          Output:
        </Text>
        <View className="bg-white rounded-xl p-5 min-h-[80px] border border-accent shadow-sm">
          <Text className="text-lg text-primary text-center leading-6 font-montserrat-semibold">
            {translatedText}
          </Text>
        </View>
        
        {/* Status Indicator */}
        <View className="flex-row items-center justify-center mt-3">
          <View 
            className={`w-2 h-2 rounded-full mr-2 
              ${isTranslating ? 'bg-accent' : (isCameraActive ? 'bg-green-500' : 'bg-neutral')}
            `} 
          />
          <Text className="text-sm text-neutral font-fredoka-medium">
            {isTranslating ? 'TRANSLATING LIVE' : (isCameraActive ? 'Camera Active' : 'Idle')}
          </Text>
        </View>
      </View>
    </View>
  );
}