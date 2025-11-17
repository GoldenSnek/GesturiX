import React, { useEffect, useState, useRef } from 'react';
import { 
  View, 
  Text, 
  TouchableOpacity, 
  ScrollView, 
  Animated, 
  Easing, 
  ActivityIndicator, 
  ImageBackground // 💡 Added ImageBackground import
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import AppHeaderLearn from '../../../components/AppHeaderLearn';
import { useTheme } from '../../../src/ThemeContext';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { getCurrentUserId } from '../../../utils/supabaseApi';
import { alphabetSigns } from '../../../constants/alphabetSigns';
import { Video, ResizeMode } from 'expo-av';
import { markLetterCompleted, getCompletedLetters, resetLetterProgress, updateStreakOnLessonComplete } from '../../../utils/progressStorage';
import { Camera, useCameraDevices, CameraDevice } from 'react-native-vision-camera';
import axios from 'axios';


const TOTAL_LETTERS = 26;
const STORAGE_LAST_LETTER = 'letterscreen_last_letter';
const CAMERA_PANEL_HEIGHT = 370;


const Letters = () => {
  const insets = useSafeAreaInsets();
  const { isDark } = useTheme();

  // Define base color class for the outer container
  const bgColorClass = isDark ? 'bg-darkbg' : 'bg-secondary';


  const [doneLetters, setDoneLetters] = useState<string[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [completed, setCompleted] = useState(false);


  // Camera states
  const [isCameraPanelVisible, setCameraPanelVisible] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [hasPermission, setHasPermission] = useState(false);
  const [prediction, setPrediction] = useState<string>('None');
  const [isSending, setIsSending] = useState(false);
  // Always open with front camera by default
  const [facing, setFacing] = useState<'front' | 'back'>('front');
  const [flash, setFlash] = useState<'on' | 'off'>('off');


  // Animation state
  const slideAnim = useRef(new Animated.Value(0)).current;
  const [allowCameraInteraction, setAllowCameraInteraction] = useState(false);


  const cameraRef = useRef<Camera>(null);
  const devices = useCameraDevices();
  const device: CameraDevice | undefined =
    devices.find((d) => d.position === facing) ??
    devices.find((d) => d.position === 'front') ??
    devices.find((d) => d.position === 'back');


  // Progress load
  useEffect(() => {
    (async () => {
      const lastLetter = await AsyncStorage.getItem(STORAGE_LAST_LETTER);
      const letters = alphabetSigns.map(l => l.letter);
      const done = await getCompletedLetters(letters);
      setDoneLetters(done);


      if (lastLetter) {
        const lastIdx = alphabetSigns.findIndex(l => l.letter === lastLetter);
        if (lastIdx !== -1) {
          setCurrentIdx(lastIdx);
          return;
        }
      }


      const nextIdx = alphabetSigns.findIndex(l => !done.includes(l.letter));
      setCurrentIdx(nextIdx === -1 ? TOTAL_LETTERS - 1 : nextIdx);
    })();
  }, []);


  useEffect(() => {
    if (alphabetSigns[currentIdx]) {
      AsyncStorage.setItem(STORAGE_LAST_LETTER, alphabetSigns[currentIdx].letter);
    }
  }, [currentIdx]);


  useEffect(() => {
    setCompleted(doneLetters.includes(alphabetSigns[currentIdx].letter));
  }, [doneLetters, currentIdx]);


  // Camera permission
  useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission();
      setHasPermission(status === 'granted');
    })();
  }, []);


  // Animate slide and set activation/interaction
  useEffect(() => {
    Animated.timing(slideAnim, {
      toValue: isCameraPanelVisible ? 1 : 0,
      duration: 330,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    }).start(() => {
      setIsCameraActive(isCameraPanelVisible);
      setAllowCameraInteraction(isCameraPanelVisible);
      if (!isCameraPanelVisible) setPrediction('None');
      // Always reset to front when opened
      if (isCameraPanelVisible) setFacing('front');
    });
  }, [isCameraPanelVisible]);


  // Camera detection and prediction
  useEffect(() => {
    if (!isCameraActive || !cameraRef.current) return;
    const interval = setInterval(async () => {
      if (isSending) return;
      setIsSending(true);
      try {
        const photo = await cameraRef.current!.takePhoto({});
        const uri = photo.path.startsWith('file://') ? photo.path : `file://${photo.path}`;
        const formData = new FormData();
        formData.append('file', {
          uri,
          type: 'image/jpeg',
          name: 'frame.jpg',
        } as any);


        // NOTE: Hardcoded local IP address - for testing only!
        const res = await axios.post('http://192.168.174.136:8000/predict', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });


        if (res.data.prediction && res.data.prediction !== 'None') {
          setPrediction(res.data.prediction.toUpperCase());
        } else {
          setPrediction('No Hand Detected');
        }
      } catch (err) {
        setPrediction('Camera error');
      }
      setIsSending(false);
    }, 200);


    return () => clearInterval(interval);
  }, [isCameraActive, isSending]);


  const handleComplete = async () => {
    await markLetterCompleted(alphabetSigns[currentIdx].letter);
    await updateStreakOnLessonComplete();
    const letters = alphabetSigns.map(l => l.letter);
    const done = await getCompletedLetters(letters);
    setDoneLetters(done);
    setCompleted(true);
    if (currentIdx < TOTAL_LETTERS - 1) {
      setTimeout(() => {
        setCurrentIdx(currentIdx + 1);
      }, 200);
    }
  };


  const handleReset = async () => {
    await resetLetterProgress();
    setDoneLetters([]);
    setCurrentIdx(0);
  };


  const canSelectLetter = (idx: number) =>
    idx === 0 ||
    doneLetters.includes(alphabetSigns[idx - 1].letter) ||
    doneLetters.includes(alphabetSigns[idx].letter);


  // Camera toggling
  const handleToggleCameraPanel = async () => {
    if (!isCameraPanelVisible) {
      if (!hasPermission) {
        const status = await Camera.requestCameraPermission();
        setHasPermission(status === 'granted');
        if (status !== 'granted') return;
      }
      setCameraPanelVisible(true);
    } else {
      setAllowCameraInteraction(false);
      setTimeout(() => setCameraPanelVisible(false), 10);
    }
  };


  const flipCamera = () => setFacing(facing === 'back' ? 'front' : 'back');
  const toggleFlash = () => setFlash(flash === 'off' ? 'on' : 'off');


  const completedCount = doneLetters.length;
  const letterData = alphabetSigns[currentIdx];


  // Calculate slide up from below video panel
  const translateY = slideAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [CAMERA_PANEL_HEIGHT + 16, 0],
  });


  return (
    // 1. Outer View sets the base background color
    <View className={`flex-1 ${bgColorClass}`}>
      <ImageBackground
        source={require('../../../assets/images/MainBG.png')}
        className="flex-1"
        resizeMode="cover"
      >
        <View
          className="flex-1"
          style={{ paddingTop: insets.top }}
        >
          <AppHeaderLearn
            title="Learn Letters"
            completedCount={completedCount}
            totalCount={TOTAL_LETTERS}
            onResetProgress={handleReset}
          />


          <ScrollView
            className="flex-1 p-4"
            contentContainerStyle={{ paddingBottom: isCameraPanelVisible ? CAMERA_PANEL_HEIGHT + 170 : 150 }}
          >
            <Text
              className={`text-lg mb-4 ${isDark ? 'text-secondary' : 'text-primary'}`}
              style={{ fontFamily: 'Audiowide-Regular' }}
            >
              Select a Letter
            </Text>


            {/* Letter Grid */}
            <View className="flex-row flex-wrap justify-between mb-1">
              {alphabetSigns.map((item, idx) => {
                const isCompleted = doneLetters.includes(item.letter);
                const isSelected = currentIdx === idx;
                const canSelect = canSelectLetter(idx);


                return (
                  <TouchableOpacity
                    key={item.letter}
                    className={`w-[18%] aspect-square rounded-lg items-center justify-center m-[1%] border-2 ${
                      isCompleted
                        ? 'border-accent bg-secondary'
                        : isDark
                          ? 'border-darkhover bg-darksurface'
                          : 'border-neutral bg-lighthover'
                    }`}
                    activeOpacity={canSelect ? 0.95 : 1}
                    onPress={() => { if (canSelect) setCurrentIdx(idx); }}
                    disabled={!canSelect}
                    style={isSelected ? { borderWidth: 3, borderColor: '#FF6B00' } : {}}
                  >
                    <Text
                      style={{
                        fontFamily: 'Fredoka-SemiBold',
                        fontSize: 24,
                        color: isCompleted
                          ? '#FF6B00'
                          : canSelect
                            ? (isDark ? '#E5E7EB' : '#6B7280')
                            : (isDark ? '#4B5563' : '#D1D5DB'),
                      }}
                    >
                      {item.letter}
                    </Text>
                    {isCompleted && (
                      <View className="absolute top-1 right-1">
                        <MaterialIcons name="check-circle" size={16} color="#FF6B00" />
                      </View>
                    )}
                  </TouchableOpacity>
                );
              })}
            </View>


            {/* --- Practice Section --- */}
            <Text
              className={`text-lg mb-4 ${isDark ? 'text-secondary' : 'text-primary'}`}
              style={{ fontFamily: 'Audiowide-Regular' }}
            >
              Practice: "{letterData.letter}"
            </Text>


            {/* Letter Video with RECORD/VIDEO Icon */}
            <View style={{ position: 'relative', marginBottom: 20 }}>
              <View
                style={{
                  width: '100%',
                  aspectRatio: 16 / 9,
                  borderRadius: 20,
                  backgroundColor: isDark ? '#222' : '#fffcfa',
                  borderWidth: 2,
                  borderColor: isDark ? '#FFB366' : '#FF6B00',
                  shadowColor: isDark ? '#FFB366' : '#FF6B00',
                  shadowOffset: { width: 0, height: 3 },
                  shadowOpacity: 0.15,
                  shadowRadius: 7,
                  elevation: 4,
                  alignItems: 'center',
                  justifyContent: 'center',
                  overflow: 'hidden',
                }}
              >
                <Video
                  source={letterData.videos[0]}
                  style={{
                    width: '100%',
                    height: '100%',
                    borderRadius: 20,
                    backgroundColor: isDark ? '#1a1a1a' : '#fff'
                  }}
                  resizeMode={ResizeMode.COVER}
                  useNativeControls
                />
              </View>


              {/* Video/Recorder-style Toggle Icon */}
              <TouchableOpacity
                onPress={handleToggleCameraPanel}
                activeOpacity={0.8}
                style={{
                  position: 'absolute',
                  top: 12,
                  right: 12,
                  backgroundColor: '#FF6B00',
                  borderRadius: 20,
                  padding: 10,
                  shadowColor: '#000',
                  shadowOpacity: 0.28,
                  shadowRadius: 6,
                  elevation: 5,
                  zIndex: 12,
                }}
              >
                <MaterialIcons 
                  name={isCameraPanelVisible ? "videocam-off" : "videocam"}
                  size={20}
                  color="white"
                />
              </TouchableOpacity>


            </View>


            {/* Camera Panel - slide below video panel */}
            {isCameraPanelVisible && (
              <Animated.View
                style={{
                  width: '100%',
                  height: CAMERA_PANEL_HEIGHT,
                  transform: [{ translateY }],
                  marginBottom: 20,
                  position: 'relative',
                  borderRadius: 22,
                  backgroundColor: isDark ? '#1E1A1A' : '#f5eee3',
                  borderWidth: 2,
                  borderColor: '#FF6B00',
                  shadowColor: '#FF6B00',
                  shadowOffset: { width: 0, height: 3 },
                  shadowOpacity: 0.16,
                  shadowRadius: 11,
                  elevation: 8,
                  flexDirection: 'column',
                  overflow: 'hidden',
                }}
                pointerEvents={allowCameraInteraction ? 'auto' : 'none'}
              >
                {hasPermission && device && allowCameraInteraction ? (
                  <>
                    <Camera
                      ref={cameraRef}
                      style={{ flex: 1, borderRadius: 20 }}
                      device={device}
                      isActive={isCameraActive}
                      photo={true}
                      torch={facing === 'back' ? flash : 'off'}
                      className="rounded-2xl"
                    />
                    
                    {/* Camera controls and corners */}
                    <View style={{
                      position: 'absolute',
                      top: 13,
                      right: 16,
                      flexDirection: 'row',
                      backgroundColor: 'rgba(70,44,17,0.19)',
                      borderRadius: 99,
                      padding: 6,
                      zIndex: 13,
                    }}>
                      <TouchableOpacity onPress={flipCamera} style={{ paddingHorizontal: 6 }}>
                        <MaterialIcons name="flip-camera-ios" size={26} color="white" />
                      </TouchableOpacity>
                      {/* Optionally show flash when back camera */}
                      {facing === 'back' && (
                        <TouchableOpacity onPress={toggleFlash} style={{ paddingHorizontal: 6 }}>
                          <MaterialIcons
                            name={flash === 'on' ? 'flash-on' : 'flash-off'}
                            size={26}
                            color="white"
                          />
                        </TouchableOpacity>
                      )}
                    </View>
                  </>
                ) : (
                  <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', paddingBottom: 24 }}>
                    <ActivityIndicator size="large" color="#FF6B00" />
                    <Text style={{
                      color: isDark ? '#fff' : '#333',
                      marginTop: 14,
                      fontFamily: 'Fredoka-Regular',
                    }}>
                      {hasPermission ? "Loading camera..." : "No camera permission"}
                    </Text>
                  </View>
                )}


                {/* Prediction Area */}
                <View style={{
                  backgroundColor: isDark ? '#181818' : '#f2f1efff',
                  borderBottomLeftRadius: 20,
                  borderBottomRightRadius: 20,
                  borderTopWidth: 1,
                  borderColor: '#FF6B00',
                  paddingVertical: 10,
                  alignItems: 'center',
                  minHeight: 65,
                }}>
                  <Text style={{
                    fontFamily: 'Audiowide-Regular',
                    fontSize: 15,
                    color: '#FF6B00',
                    marginBottom: 4,
                  }}>
                    Detected Sign:
                  </Text>
                  <Text style={{
                    fontFamily: 'Fredoka-SemiBold',
                    fontSize: 20,
                    color: isDark ? '#fff' : '#222',
                    letterSpacing: 1,
                  }}>
                    {prediction}
                  </Text>
                  <View style={{
                    flexDirection: 'row',
                    alignItems: 'center',
                    marginTop: 6,
                  }}>
                    <View style={{
                      width: 9,
                      height: 9,
                      borderRadius: 5,
                      backgroundColor: isCameraActive ? '#FF6B00' : '#b8bab9',
                      marginRight: 7,
                    }} />
                    <Text style={{
                      fontFamily: 'Montserrat-SemiBold',
                      fontSize: 13,
                      color: isDark ? '#ccc' : '#5e6272',
                    }}>
                      {isCameraActive ? 'LIVE' : 'Paused'}
                    </Text>
                  </View>
                </View>
              </Animated.View>
            )}


            {/* Tips Section - UPDATED TO MATCH phrase.tsx */}
            <Text
              style={{
                marginVertical: 8,
                paddingHorizontal: 4,
                flexDirection: 'row',
                flexWrap: 'wrap',
              }}
            >
              <Text
                style={{
                  fontWeight: 'bold',
                  fontFamily: 'Montserrat-Bold',
                  fontSize: 14,
                  color: isDark ? '#FFA500' : '#FF6B00',
                }}
              >
                Tips:
              </Text>
              <Text
                style={{
                  fontFamily: 'Montserrat-SemiBold',
                  color: isDark ? '#ccc' : '#333',
                  fontSize: 13,
                }}
              >
                {' '}{letterData.tips}
              </Text>
            </Text>


            {/* Practice Buttons */}
            <View className="flex-row justify-between mb-4">
              {['Slow Motion', 'Repeat', 'Practice'].map((label) => (
                <TouchableOpacity
                  key={label}
                  className={`flex-1 rounded-full py-3 mx-1 items-center border border-accent ${
                    isDark ? 'bg-darksurface' : 'bg-lighthover'
                  }`}
                >
                  <Text
                    className={`${isDark ? 'text-secondary' : 'text-primary'}`}
                    style={{ fontFamily: 'Fredoka-Regular' }}
                  >
                    {label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>


            {/* Mark as Completed Button */}
            <TouchableOpacity
              className={`w-full bg-accent rounded-full py-4 items-center shadow-md ${completed ? 'opacity-60' : ''}`}
              disabled={completed}
              onPress={handleComplete}
            >
              <Text
                className="text-secondary text-lg"
                style={{ fontFamily: 'Fredoka-SemiBold' }}
              >
                {completed ? 'Completed!' : 'Mark as Completed'}
              </Text>
            </TouchableOpacity>
          </ScrollView>
        </View>
      </ImageBackground>
    </View>
  );
};


export default Letters;