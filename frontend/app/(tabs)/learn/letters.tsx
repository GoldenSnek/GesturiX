import React, { useEffect, useState, useRef, useCallback } from 'react';
import { 
  View, 
  Text, 
  TouchableOpacity, 
  ScrollView, 
  Animated, 
  Easing, 
  ActivityIndicator, 
  ImageBackground 
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import AppHeaderLearn from '../../../components/AppHeaderLearn';
import { useTheme } from '../../../src/ThemeContext';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { alphabetSigns } from '../../../constants/alphabetSigns';
import { 
  markLetterCompleted, 
  getCompletedLetters, 
  resetLetterProgress, 
  updateStreakOnLessonComplete, 
  updatePracticeTime
} from '../../../utils/progressStorage';
import { 
  getCurrentUserId, 
  getUserSavedItems, 
  saveItem, 
  unsaveItem,
  SavedItem 
} from '../../../utils/supabaseApi';
import { Camera, useCameraDevices, CameraDevice } from 'react-native-vision-camera';
import axios from 'axios';
import { Video, ResizeMode } from 'expo-av';
import { ENDPOINTS } from '../../../constants/ApiConfig';
import { useFocusEffect, useLocalSearchParams } from 'expo-router'; 
import { useSettings } from '../../../src/SettingsContext';
import * as Haptics from 'expo-haptics';

const TOTAL_LETTERS = 26;
const CAMERA_PANEL_HEIGHT = 370;

const Letters = () => {
  const insets = useSafeAreaInsets();
  const { isDark } = useTheme();
  const { vibrationEnabled } = useSettings();
  const { initialLetter } = useLocalSearchParams<{ initialLetter?: string }>(); 

  const bgColorClass = isDark ? 'bg-darkbg' : 'bg-secondary';
  const textColor = isDark ? 'text-secondary' : 'text-primary';

  const [doneLetters, setDoneLetters] = useState<string[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [completed, setCompleted] = useState(false);

  const [userSavedItems, setUserSavedItems] = useState<SavedItem[]>([]);
  const [isSaved, setIsSaved] = useState(false);
  const [userId, setUserId] = useState<string | null>(null);

  const [isSlowMotion, setIsSlowMotion] = useState(false);
  const [isRepeating, setIsRepeating] = useState(true);

  const [isCameraPanelVisible, setCameraPanelVisible] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [hasPermission, setHasPermission] = useState(false);
  const [prediction, setPrediction] = useState<string>('None');
  
  const isSending = useRef(false);

  const [facing, setFacing] = useState<'front' | 'back'>('front');
  const [flash, setFlash] = useState<'on' | 'off'>('off');

  const [hasVibratedForCurrent, setHasVibratedForCurrent] = useState(false);

  const slideAnim = useRef(new Animated.Value(0)).current;
  const [allowCameraInteraction, setAllowCameraInteraction] = useState(false);

  const cameraRef = useRef<Camera>(null);
  const devices = useCameraDevices();
  const device: CameraDevice | undefined =
    devices.find(d => d.position === facing) ??
    devices.find(d => d.position === 'front') ??
    devices.find(d => d.position === 'back');
  
  const currentLetter = alphabetSigns[currentIdx]?.letter;

  useFocusEffect(
    useCallback(() => {
      const startTime = Date.now();
      return () => {
        const durationMs = Date.now() - startTime;
        if (durationMs > 2000) {
          updatePracticeTime(durationMs / 1000 / 60 / 60);
        }
      };
    }, [])
  );

  useFocusEffect(
    useCallback(() => {
      const loadUserAndSaved = async () => {
        const uid = await getCurrentUserId();
        setUserId(uid);
        if (uid) {
          const items = await getUserSavedItems(uid);
          setUserSavedItems(items);
        }
      };
      loadUserAndSaved();
    }, [])
  );

  useEffect(() => {
    if (currentLetter) {
      const found = userSavedItems.find(
        i => i.item_type === 'letter' && i.item_identifier === currentLetter
      );
      setIsSaved(!!found);
    }
    setHasVibratedForCurrent(false); 
  }, [currentIdx, userSavedItems, currentLetter]);

  const handleToggleSave = async () => {
    if (!userId || !currentLetter) return;

    if (isSaved) {
      setIsSaved(false); 
      await unsaveItem(userId, 'letter', currentLetter);
      const items = await getUserSavedItems(userId);
      setUserSavedItems(items);
    } else {
      setIsSaved(true); 
      await saveItem(userId, 'letter', currentLetter);
      const items = await getUserSavedItems(userId);
      setUserSavedItems(items);
    }
  };

  useEffect(() => {
    (async () => {
      const uid = await getCurrentUserId();
      const letters = alphabetSigns.map(l => l.letter);
      const done = await getCompletedLetters(letters);
      setDoneLetters(done);

      const firstUncompletedIdx = alphabetSigns.findIndex(l => !done.includes(l.letter));
      const defaultIdx = firstUncompletedIdx === -1 ? 0 : firstUncompletedIdx;

      if (initialLetter) {
        const paramIdx = alphabetSigns.findIndex(l => l.letter === initialLetter);
        if (paramIdx !== -1) {
          setCurrentIdx(paramIdx);
          return;
        }
      }

      if (uid) {
        const storageKey = `user_${uid}_letters_last_idx`;
        const lastLetter = await AsyncStorage.getItem(storageKey);
        
        if (lastLetter) {
          const lastIdx = alphabetSigns.findIndex(l => l.letter === lastLetter);
          if (lastIdx !== -1) {
            setCurrentIdx(lastIdx);
            return;
          }
        }
      }

      setCurrentIdx(defaultIdx);
    })();
  }, [initialLetter]); 

  useEffect(() => {
    if (alphabetSigns[currentIdx] && userId) {
      const storageKey = `user_${userId}_letters_last_idx`;
      AsyncStorage.setItem(storageKey, alphabetSigns[currentIdx].letter);
    }
  }, [currentIdx, userId]);

  useEffect(() => {
    setCompleted(doneLetters.includes(alphabetSigns[currentIdx].letter));
  }, [doneLetters, currentIdx]);

  useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission();
      setHasPermission(status === 'granted');
    })();
  }, []);

  useEffect(() => {
    Animated.timing(slideAnim, {
      toValue: isCameraPanelVisible ? 1 : 0,
      duration: 330,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    }).start(() => {
      setIsCameraActive(isCameraPanelVisible);
      setAllowCameraInteraction(isCameraPanelVisible);
      if (!isCameraPanelVisible) {
        setPrediction('None');
        setHasVibratedForCurrent(false);
      }
      if (isCameraPanelVisible) setFacing('front');
    });
  }, [isCameraPanelVisible]);

  useEffect(() => {
    if (!isCameraActive || !cameraRef.current) return;
    
    const interval = setInterval(async () => {
      if (isSending.current) return; 
      isSending.current = true;
      try {
        const photo = await cameraRef.current!.takePhoto({});
        const uri = photo.path.startsWith('file://') ? photo.path : `file://${photo.path}`;
        const formData = new FormData();
        formData.append('file', { uri, type: 'image/jpeg', name: 'frame.jpg' } as any);
        
        const res = await axios.post(ENDPOINTS.PREDICT, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        
        if (res.data.prediction && res.data.prediction !== 'None') {
          const pred = res.data.prediction.toUpperCase();
          setPrediction(pred);

          if (pred === currentLetter && !hasVibratedForCurrent) {
            if (vibrationEnabled) {
              Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
            }
            setHasVibratedForCurrent(true);
          } 
          else if (pred !== currentLetter) {
            setHasVibratedForCurrent(false);
          }

        } else {
          setPrediction('No Hand Detected');
          setHasVibratedForCurrent(false);
        }
      } catch (e) {
        //silent catch
      }
      isSending.current = false;
    }, 200);
    return () => clearInterval(interval);
  }, [isCameraActive, currentLetter, hasVibratedForCurrent, vibrationEnabled]);

  const handleComplete = async () => {
    await markLetterCompleted(alphabetSigns[currentIdx].letter);
    await updateStreakOnLessonComplete();
    const letters = alphabetSigns.map(l => l.letter);
    const done = await getCompletedLetters(letters);
    setDoneLetters(done);
    setCompleted(true);
    if (currentIdx < TOTAL_LETTERS - 1) {
      setTimeout(() => setCurrentIdx(currentIdx + 1), 200);
    }
  };

  const handleReset = async () => {
    await resetLetterProgress();
    setDoneLetters([]);
    setCurrentIdx(0);
  };

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

  const translateY = slideAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [CAMERA_PANEL_HEIGHT + 16, 0],
  });

  return (
    <View className={`flex-1 ${bgColorClass}`}>
      <ImageBackground
        source={require('../../../assets/images/MainBG.png')}
        className="flex-1"
        resizeMode="cover"
      >
        <View className="flex-1" style={{ paddingTop: insets.top }}>
          <AppHeaderLearn
            title="Learn Letters"
            completedCount={completedCount}
            totalCount={TOTAL_LETTERS}
            onResetProgress={handleReset}
          />

          <ScrollView
            className="flex-1 p-4"
            contentContainerStyle={{ paddingBottom: 150 }}
          >
            <Text
              className={`text-lg mb-4 ${isDark ? 'text-secondary' : 'text-primary'}`}
              style={{ fontFamily: 'Audiowide-Regular' }}
            >
              Select a Letter
            </Text>

            <View className="flex-row flex-wrap justify-between">
              {alphabetSigns.map((item, idx) => {
                const isCompleted = doneLetters.includes(item.letter);
                const isSelected = currentIdx === idx;

                return (
                  <TouchableOpacity
                    key={item.letter}
                    className={`w-[18%] aspect-square rounded-lg items-center justify-center m-[1%] border-2 ${
                      isCompleted
                        ? 'border-accent bg-secondary'
                        : isDark ? 'border-darkhover bg-darksurface' : 'border-neutral bg-lighthover'
                    }`}
                    activeOpacity={0.95}
                    onPress={() => setCurrentIdx(idx)}
                    style={isSelected ? { borderWidth: 3, borderColor: '#FF6B00' } : {}}
                  >
                    <Text
                      style={{
                        fontFamily: 'Fredoka-SemiBold',
                        fontSize: 24,
                        color: isCompleted ? '#FF6B00' : (isDark ? '#E5E7EB' : '#6B7280'),
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

            <Text
              className={`text-lg mb-4 ${isDark ? 'text-secondary' : 'text-primary'}`}
              style={{ fontFamily: 'Audiowide-Regular' }}
            >
              Practice: "{letterData.letter}"
            </Text>

            <View style={{ position: 'relative', marginBottom: 20 }}>
              <View
                className="border-accent"
                style={{
                  width: '100%',
                  aspectRatio: 16 / 9,
                  borderRadius: 20,
                  backgroundColor: isDark ? '#222' : '#fffcfa',
                  borderWidth: 2,
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
                  key={currentIdx}
                  source={letterData.image}
                  style={{ width: '100%', height: '100%', borderRadius: 20 }}
                  resizeMode={ResizeMode.COVER}
                  shouldPlay
                  isLooping={isRepeating}
                  rate={isSlowMotion ? 0.5 : 1.0}
                  useNativeControls
                  isMuted
                />
              </View>
            </View>

            {/* Camera Panel */}
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
                  borderColor: prediction === currentLetter ? '#10B981' : '#FF6B00',
                  shadowColor: prediction === currentLetter ? '#10B981' : '#FF6B00',
                  shadowOffset: { width: 0, height: 3 },
                  shadowOpacity: 0.16,
                  shadowRadius: 11,
                  elevation: 8,
                  flexDirection: 'column',
                  overflow: 'hidden',
                }}
                pointerEvents={allowCameraInteraction ? 'auto' : 'none'}
              >
                {/* Camera View */}
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
                    
                    <View className="absolute top-6 right-6 flex-row space-x-2 bg-black/30 rounded-xl p-1 z-50">
                        {facing === 'back' && (
                            <TouchableOpacity onPress={toggleFlash} className="p-2">
                                <MaterialIcons
                                    name={flash === 'on' ? 'flash-on' : 'flash-off'}
                                    size={24}
                                    color="white"
                                />
                            </TouchableOpacity>
                        )}
                        <TouchableOpacity onPress={flipCamera} className="p-2">
                            <MaterialIcons name="flip-camera-ios" size={24} color="white" />
                        </TouchableOpacity>
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

                <View style={{
                  backgroundColor: isDark ? '#181818' : '#f2f1efff',
                  borderBottomLeftRadius: 20,
                  borderBottomRightRadius: 20,
                  borderTopWidth: 1,
                  borderColor: prediction === currentLetter ? '#10B981' : '#FF6B00',
                  paddingVertical: 10,
                  alignItems: 'center',
                  minHeight: 65,
                }}>
                  <Text style={{
                    fontFamily: 'Audiowide-Regular',
                    fontSize: 15,
                    color: prediction === currentLetter ? '#10B981' : '#FF6B00',
                    marginBottom: 4,
                  }}>
                    {prediction === currentLetter ? "Correct!" : "Detected Sign:"}
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

            <View className="flex-row justify-between mb-4">
              <TouchableOpacity
                onPress={() => setIsSlowMotion(!isSlowMotion)}
                className={`flex-1 rounded-xl py-2 mx-1 items-center justify-center border border-accent ${
                  isSlowMotion 
                    ? 'bg-accent' 
                    : (isDark ? 'bg-darksurface' : 'bg-lighthover')
                }`}
              >
                <MaterialIcons 
                  name="speed" 
                  size={20} 
                  color={isSlowMotion ? '#F8F8F8' : (isDark ? '#F8F8F8' : '#2C2C2C')} 
                  style={{ marginBottom: 2 }}
                />
                <Text
                  className={`text-xs text-center ${isSlowMotion ? 'text-secondary' : textColor}`}
                  style={{ fontFamily: 'Fredoka-Regular' }}
                  numberOfLines={1}
                >
                  Slow Mo
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                onPress={() => setIsRepeating(!isRepeating)}
                className={`flex-1 rounded-xl py-2 mx-1 items-center justify-center border border-accent ${
                  isRepeating 
                    ? 'bg-accent' 
                    : (isDark ? 'bg-darksurface' : 'bg-lighthover')
                }`}
              >
                <MaterialIcons 
                  name="replay" 
                  size={20} 
                  color={isRepeating ? '#F8F8F8' : (isDark ? '#F8F8F8' : '#2C2C2C')} 
                  style={{ marginBottom: 2 }}
                />
                <Text
                  className={`text-xs text-center ${isRepeating ? 'text-secondary' : textColor}`}
                  style={{ fontFamily: 'Fredoka-Regular' }}
                  numberOfLines={1}
                >
                  Repeat
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                onPress={handleToggleCameraPanel}
                className={`flex-1 rounded-xl py-2 mx-1 items-center justify-center border border-accent ${
                  isCameraPanelVisible ? 'bg-accent' : (isDark ? 'bg-darksurface' : 'bg-lighthover')
                }`}
              >
                <MaterialIcons 
                  name={isCameraPanelVisible ? "videocam-off" : "videocam"}
                  size={20} 
                  color={isCameraPanelVisible ? '#F8F8F8' : (isDark ? '#F8F8F8' : '#2C2C2C')} 
                  style={{ marginBottom: 2 }}
                />
                <Text
                  className={`text-xs text-center ${isCameraPanelVisible ? 'text-secondary' : textColor}`}
                  style={{ fontFamily: 'Fredoka-Regular' }}
                  numberOfLines={1}
                >
                  Practice
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                onPress={handleToggleSave}
                className={`flex-1 rounded-xl py-2 mx-1 items-center justify-center border border-accent ${
                  isSaved
                    ? 'bg-accent' 
                    : (isDark ? 'bg-darksurface' : 'bg-lighthover')
                }`}
              >
                <MaterialIcons 
                  name={isSaved ? "bookmark" : "bookmark-outline"} 
                  size={20} 
                  color={isSaved ? '#F8F8F8' : (isDark ? '#F8F8F8' : '#2C2C2C')} 
                  style={{ marginBottom: 2 }}
                />
                <Text
                  className={`text-xs text-center ${isSaved ? 'text-secondary' : textColor}`}
                  style={{ fontFamily: 'Fredoka-Regular' }}
                  numberOfLines={1}
                >
                  {isSaved ? 'Saved' : 'Save'}
                </Text>
              </TouchableOpacity>

            </View>

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