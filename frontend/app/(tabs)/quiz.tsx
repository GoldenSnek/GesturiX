import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ImageBackground,
  StyleSheet,
  Animated,
  ActivityIndicator,
  Dimensions,
  PanResponder, // ✅ Added PanResponder
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import { Camera, useCameraDevices, useCameraFormat } from 'react-native-vision-camera';
import axios from 'axios';
import * as Haptics from 'expo-haptics';
import { Video, ResizeMode } from 'expo-av';
import { useTheme } from '../../src/ThemeContext';
import { alphabetSigns } from '../../constants/alphabetSigns';
import { updateStreakOnLessonComplete } from '../../utils/progressStorage';
import { useFocusEffect, useRouter } from 'expo-router';
import { ENDPOINTS } from '../../constants/ApiConfig'; // 🔌 Import Config

// --- Configuration ---
const TOTAL_QUESTIONS = 10;
const DETECTION_INTERVAL = 600;
const REQUIRED_MATCH_STREAK = 2;

type QuestionType = 'recognition' | 'performance';

interface QuizQuestion {
  type: QuestionType;
  target: typeof alphabetSigns[0];
  options?: string[];
}

type GameState = 'menu' | 'playing' | 'summary';
type AnswerState = 'idle' | 'correct' | 'incorrect';

export default function QuizScreen() {
  const insets = useSafeAreaInsets();
  const { isDark } = useTheme();
  const router = useRouter();

  // --- Colors ---
  const colors = {
    text: isDark ? '#F8F8F8' : '#2C2C2C',
    subText: isDark ? '#A8A8A8' : '#666666',
    accent: '#FF6B00',
    success: '#10B981',
    error: '#EF4444',
    surface: isDark ? '#333333' : '#FFFFFF',
    border: isDark ? '#444444' : '#E5E5E5',
    activeBorder: isDark ? '#FF6B00' : '#FF6B00',
  };

  // --- State ---
  const [hasPermission, setHasPermission] = useState(false);
  const [gameState, setGameState] = useState<GameState>('menu');
  const [questions, setQuestions] = useState<QuizQuestion[]>([]);
  const [currentQIndex, setCurrentQIndex] = useState(0);
  const [score, setScore] = useState(0);

  // Feedback State
  const [answerState, setAnswerState] = useState<AnswerState>('idle');
  const [selectedOption, setSelectedOption] = useState<string | null>(null);

  // Camera State
  const [isDetecting, setIsDetecting] = useState(false);
  const [currentPrediction, setCurrentPrediction] = useState('...');
  const [matchStreak, setMatchStreak] = useState(0);

  const cameraRef = useRef<Camera>(null);
  const progressAnim = useRef(new Animated.Value(0)).current;

  // --- Camera Setup ---
  const devices = useCameraDevices();
  const device = devices.find((d) => d.position === 'front') ?? devices.find((d) => d.position === 'back');
  const format = useCameraFormat(device, [
    { photoResolution: { width: 640, height: 480 } },
    { fps: 30 }
  ]);

  // ✅ PanResponder for Swipe Navigation
  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (_, gestureState) => Math.abs(gestureState.dx) > 10,
      onPanResponderRelease: (_, gestureState) => {
        if (gestureState.dx < -30) {
          // Swipe Left -> Go to Profile
          router.push('/profile');
        } else if (gestureState.dx > 30) {
          // Swipe Right -> Go to Learn
          router.push('/learn');
        }
      },
    })
  ).current;

  useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission();
      setHasPermission(status === 'granted');
    })();
  }, []);

  // Reset when leaving tab
  useFocusEffect(
    useCallback(() => {
      return () => {
        setGameState('menu');
        setIsDetecting(false);
        setAnswerState('idle');
      };
    }, [])
  );

  // --- Logic ---

  const startQuiz = () => {
    const shuffled = [...alphabetSigns].sort(() => 0.5 - Math.random());
    const selection = shuffled.slice(0, TOTAL_QUESTIONS);

    const newQuestions: QuizQuestion[] = selection.map(item => {
      const type: QuestionType = Math.random() > 0.5 ? 'recognition' : 'performance';
      let options: string[] | undefined;

      if (type === 'recognition') {
        const others = alphabetSigns.filter(l => l.letter !== item.letter);
        const distractors = others.sort(() => 0.5 - Math.random()).slice(0, 3);
        options = [item.letter, ...distractors.map(d => d.letter)].sort(() => 0.5 - Math.random());
      }

      return { type, target: item, options };
    });

    setQuestions(newQuestions);
    setCurrentQIndex(0);
    setScore(0);
    setGameState('playing');
    setAnswerState('idle');
    setSelectedOption(null);
    setIsDetecting(newQuestions[0].type === 'performance');
    
    progressAnim.setValue(0);
    Animated.timing(progressAnim, {
      toValue: (1 / TOTAL_QUESTIONS) * 100,
      duration: 500,
      useNativeDriver: false,
    }).start();
  };

  const handleRecognitionAnswer = (selected: string) => {
    if (answerState !== 'idle') return;
    const correct = selected === questions[currentQIndex].target.letter;
    setSelectedOption(selected);
    processResult(correct);
  };

  const handlePerformanceSuccess = () => {
    if (answerState !== 'idle') return;
    processResult(true);
  };

  const skipQuestion = () => {
    if (answerState !== 'idle') return;
    processResult(false);
  };

  const processResult = (correct: boolean) => {
    setIsDetecting(false);
    setAnswerState(correct ? 'correct' : 'incorrect');

    if (correct) {
      setScore(s => s + 1);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } else {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    }

    setTimeout(nextQuestion, 1200);
  };

  const nextQuestion = () => {
    const nextIdx = currentQIndex + 1;
    if (nextIdx >= TOTAL_QUESTIONS) {
      finishGame();
    } else {
      setAnswerState('idle');
      setSelectedOption(null);
      setMatchStreak(0);
      setCurrentPrediction('...');
      setCurrentQIndex(nextIdx);
      
      Animated.timing(progressAnim, {
        toValue: ((nextIdx + 1) / TOTAL_QUESTIONS) * 100,
        duration: 500,
        useNativeDriver: false,
      }).start();

      const nextQ = questions[nextIdx];
      setIsDetecting(nextQ.type === 'performance');
    }
  };

  const finishGame = async () => {
    setGameState('summary');
    setIsDetecting(false);
    if (score > 0) await updateStreakOnLessonComplete();
  };

  // --- Camera Loop ---
  useEffect(() => {
    if (!isDetecting || gameState !== 'playing' || answerState !== 'idle' || !cameraRef.current) return;

    const interval = setInterval(async () => {
      try {
        if (!cameraRef.current) return;
        const photo = await cameraRef.current.takePhoto({ flash: 'off', enableShutterSound: false });
        const uri = `file://${photo.path}`;
        const formData = new FormData();
        formData.append('file', { uri, type: 'image/jpeg', name: 'quiz.jpg' } as any);

        const res = await axios.post(ENDPOINTS.PREDICT, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 2000,
        });

        const pred = res.data.prediction ? res.data.prediction.toUpperCase() : '...';
        setCurrentPrediction(pred);

        if (pred === questions[currentQIndex]?.target.letter) {
          setMatchStreak(prev => {
            const nextVal = prev + 1;
            if (nextVal >= REQUIRED_MATCH_STREAK) {
              handlePerformanceSuccess();
              return 0;
            }
            return nextVal;
          });
        } else {
          setMatchStreak(0);
        }
      } catch (e) {}
    }, DETECTION_INTERVAL);

    return () => clearInterval(interval);
  }, [isDetecting, gameState, currentQIndex, answerState]);

  // --- Renderers ---

  const renderMenu = () => (
    <View className="flex-1 justify-center items-center px-6">
      <View className={`w-full p-8 rounded-[32px] shadow-sm border ${isDark ? 'bg-darksurface border-gray-700' : 'bg-white border-gray-100'}`}>
        <View className="items-center mb-6 bg-accent/10 p-6 rounded-full self-center">
          <MaterialIcons name="school" size={56} color={colors.accent} />
        </View>
        <Text className={`text-3xl text-center font-audiowide mb-3 ${colors.text}`}>
          Sign Master
        </Text>
        <Text className={`text-center mb-8 font-montserrat-regular text-base leading-6 ${colors.subText}`}>
          Challenge yourself! Mix of video recognition and live performance.
        </Text>

        <TouchableOpacity 
          onPress={startQuiz}
          className="w-full bg-accent py-4 rounded-full shadow-md active:opacity-80 flex-row justify-center items-center"
        >
          <Text className="text-white font-fredoka-semibold text-lg mr-2">Start Quiz</Text>
          <MaterialIcons name="arrow-forward" size={20} color="white" />
        </TouchableOpacity>
      </View>
    </View>
  );

  const renderSummary = () => (
    <View className="flex-1 justify-center items-center px-6">
      <View className={`w-full p-8 rounded-[32px] shadow-sm border ${isDark ? 'bg-darksurface border-gray-700' : 'bg-white border-gray-100'}`}>
        <MaterialIcons name={score >= (TOTAL_QUESTIONS * 0.6) ? "emoji-events" : "thumb-up"} size={70} color={colors.accent} className="mb-4 self-center" />
        
        <Text className={`text-2xl font-audiowide mb-2 text-center ${colors.text}`}>
          {score >= (TOTAL_QUESTIONS * 0.6) ? "Great Job!" : "Practice Makes Perfect"}
        </Text>
        
        <View className="bg-accent/5 px-8 py-6 rounded-3xl border border-accent/20 mb-8 mt-4 self-center w-full items-center">
          <Text className={`text-sm font-montserrat-semibold mb-1 ${colors.subText}`}>FINAL SCORE</Text>
          <Text className="text-5xl font-fredoka-bold text-accent">
            {score}<Text className="text-3xl text-gray-400">/{TOTAL_QUESTIONS}</Text>
          </Text>
        </View>

        <TouchableOpacity 
          onPress={() => setGameState('menu')}
          className="w-full bg-accent py-4 rounded-full shadow-sm active:opacity-80"
        >
          <Text className="text-white text-center font-fredoka-semibold text-lg">Back to Menu</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  const renderPlaying = () => {
    const q = questions[currentQIndex];
    if (!q) return <ActivityIndicator color={colors.accent} />;
    const isPerformance = q.type === 'performance';

    return (
      <View className="flex-1">
        {/* Header Bar */}
        <View className="flex-row items-center px-6 mt-2 mb-4">
          <TouchableOpacity 
            onPress={() => setGameState('menu')} 
            className={`p-2 rounded-full ${isDark ? 'bg-gray-800' : 'bg-gray-100'}`}
          >
            <MaterialIcons name="close" size={20} color={colors.text} />
          </TouchableOpacity>
          
          <View className="flex-1 mx-4 h-2 bg-gray-200/50 rounded-full overflow-hidden">
            <Animated.View 
              style={{ 
                width: progressAnim.interpolate({ inputRange: [0, 100], outputRange: ['0%', '100%'] }),
                backgroundColor: colors.accent 
              }}
              className="h-full rounded-full" 
            />
          </View>
          
          <View className={`px-3 py-1 rounded-full ${isDark ? 'bg-gray-800' : 'bg-gray-100'}`}>
            <Text className={`font-fredoka-medium text-xs ${colors.text}`}>
              {currentQIndex + 1} / {TOTAL_QUESTIONS}
            </Text>
          </View>
        </View>

        <View className="flex-1 px-6">
          {/* === RECOGNITION === */}
          {!isPerformance && (
            <View className="flex-1">
              <Text className={`text-xl font-audiowide mb-6 text-center ${colors.text}`}>
                Identify this sign
              </Text>
              
              <View className={`w-full aspect-[16/10] rounded-2xl overflow-hidden border shadow-sm mb-8 bg-black ${isDark ? 'border-gray-700' : 'border-gray-100'}`}>
                <Video
                  source={q.target.image}
                  style={{ width: '100%', height: '100%' }}
                  resizeMode={ResizeMode.COVER}
                  shouldPlay
                  isLooping
                  isMuted
                />
              </View>

              <View className="flex-row flex-wrap justify-between gap-y-3">
                {q.options?.map((opt, idx) => {
                  let btnBg = isDark ? '#2A2A2A' : '#FFF';
                  let btnBorder = isDark ? '#444' : '#E5E5E5';
                  let btnText = colors.text;

                  // Subtle Logic: Only change the specific buttons involved
                  if (answerState !== 'idle') {
                    if (opt === q.target.letter) {
                      // Correct answer always turns green
                      btnBg = isDark ? 'rgba(16, 185, 129, 0.15)' : '#ECFDF5';
                      btnBorder = colors.success;
                      btnText = colors.success;
                    } else if (opt === selectedOption && opt !== q.target.letter) {
                      // Wrong selection turns red
                      btnBg = isDark ? 'rgba(239, 68, 68, 0.15)' : '#FEF2F2';
                      btnBorder = colors.error;
                      btnText = colors.error;
                    } else {
                      // Others fade out slightly
                      btnBorder = 'transparent';
                      btnBg = isDark ? '#222' : '#F9F9F9';
                    }
                  }

                  return (
                    <TouchableOpacity
                      key={idx}
                      onPress={() => handleRecognitionAnswer(opt)}
                      disabled={answerState !== 'idle'}
                      className="w-[48%] py-5 rounded-2xl border-2 items-center justify-center shadow-sm"
                      style={{ backgroundColor: btnBg, borderColor: btnBorder }}
                    >
                      <Text className="text-2xl font-fredoka-bold" style={{ color: btnText }}>
                        {opt}
                      </Text>
                    </TouchableOpacity>
                  );
                })}
              </View>
            </View>
          )}

          {/* === PERFORMANCE === */}
          {isPerformance && (
            <View className="flex-1 items-center">
              <Text className={`text-lg font-audiowide mb-1 text-center ${colors.subText}`}>
                Sign the letter
              </Text>
              <Text className="text-accent text-7xl font-fredoka-bold mb-6">
                {q.target.letter}
              </Text>

              <View 
                className="w-full aspect-[3/4] rounded-3xl overflow-hidden bg-black relative shadow-md"
                style={{ 
                  maxHeight: '52%', 
                  borderWidth: 4,
                  // Only the border changes color for feedback
                  borderColor: answerState === 'correct' ? colors.success : (answerState === 'incorrect' ? colors.error : colors.accent)
                }}
              >
                {hasPermission && device ? (
                  <Camera
                    ref={cameraRef}
                    style={StyleSheet.absoluteFill}
                    device={device}
                    format={format}
                    isActive={isDetecting}
                    photo={true}
                  />
                ) : (
                  <View className="flex-1 justify-center items-center"><Text className="text-white">No Camera</Text></View>
                )}

                {/* Sleek Overlay for Success */}
                {answerState === 'correct' && (
                  <View className="absolute inset-0 bg-black/40 justify-center items-center">
                    <View className="bg-white p-4 rounded-full">
                      <MaterialIcons name="check" size={40} color={colors.success} />
                    </View>
                  </View>
                )}

                {/* Debug/Hint Text */}
                {answerState === 'idle' && (
                  <View className="absolute bottom-3 right-3 bg-black/60 px-3 py-1 rounded-lg">
                    <Text className="text-white/80 font-mono text-xs">
                      {currentPrediction}
                    </Text>
                  </View>
                )}
              </View>

              {/* Skip Button - Closer to camera */}
              <View className="w-full mt-6 items-center">
                <TouchableOpacity
                  onPress={skipQuestion}
                  disabled={answerState !== 'idle'}
                  className={`px-8 py-3 rounded-full border ${isDark ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-white'}`}
                >
                  <Text className={`font-montserrat-semibold text-sm ${colors.subText}`}>
                    I don't know this one
                  </Text>
                </TouchableOpacity>
              </View>
            </View>
          )}
        </View>
      </View>
    );
  };

  return (
    <View className={`flex-1 ${isDark ? 'bg-darkbg' : 'bg-secondary'}`}>
      <ImageBackground
        source={require('../../assets/images/MainBG.png')}
        className="flex-1"
        resizeMode="cover"
      >
        {/* ✅ Applied PanResponder Handlers here to catch swipes across the screen */}
        <View 
          className="flex-1" 
          style={{ paddingTop: insets.top }}
          {...panResponder.panHandlers}
        >
          {gameState === 'menu' && renderMenu()}
          {gameState === 'playing' && renderPlaying()}
          {gameState === 'summary' && renderSummary()}
        </View>
      </ImageBackground>
    </View>
  );
}