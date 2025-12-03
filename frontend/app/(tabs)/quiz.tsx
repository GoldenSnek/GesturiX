import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ImageBackground,
  StyleSheet,
  Animated,
  ActivityIndicator,
  PanResponder,
  Modal,
  ScrollView,
  Vibration,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons, Ionicons } from '@expo/vector-icons';
import { Camera, useCameraDevices, useCameraFormat } from 'react-native-vision-camera';
import axios from 'axios';
import * as Haptics from 'expo-haptics';
import { Video, ResizeMode } from 'expo-av';
import { useTheme } from '../../src/ThemeContext';
import { useSettings } from '../../src/SettingsContext'; 
import { alphabetSigns } from '../../constants/alphabetSigns';
import { phrases } from '../../constants/phrases';
import { updateStreakOnLessonComplete, updatePracticeTime } from '../../utils/progressStorage';
import { useFocusEffect, useRouter } from 'expo-router';
import { ENDPOINTS } from '../../constants/ApiConfig';
import { supabase } from '../../src/supabaseClient';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { getCurrentUserId } from '../../utils/supabaseApi';

// --- CONFIGURATION ---
const DETECTION_INTERVAL = 500; // ms between camera checks
const REQUIRED_MATCH_STREAK = 2; // How many consecutive correct frames needed

// --- TYPES ---
type ItemType = 'letter' | 'phrase';

interface QuizItem {
  id: string;
  label: string;
  video: any;
  type: ItemType;
}

type QuestionType = 'recognition' | 'performance' | 'sequence';
type DifficultyLevel = 'beginner' | 'intermediate' | 'advanced';

interface QuizQuestion {
  type: QuestionType;
  target: QuizItem;               
  sequenceTargets?: QuizItem[];   
  options?: string[];             
  correctAnswer: string;          
}

interface QuestionAttempt {
  questionIndex: number;
  isCorrect: boolean;
  timeTaken: number;
  skipped: boolean;
}

type GameState = 'menu' | 'difficulty' | 'playing' | 'summary';
type AnswerState = 'idle' | 'correct' | 'incorrect' | 'timeout';

interface QuizSettings {
  difficulty: DifficultyLevel;
  questionsCount: number;
  timeLimit: number;
  soundEnabled: boolean;
}

export default function QuizScreen() {
  const insets = useSafeAreaInsets();
  const { isDark } = useTheme();
  const { vibrationEnabled } = useSettings(); 
  const router = useRouter();

  // 🎨 Unified Theme Colors
  const colors = {
    text: isDark ? '#secondary' : '#primary',
    textColor: isDark ? '#E5E7EB' : '#1F2937',
    subText: isDark ? '#9CA3AF' : '#4B5563',
    accent: '#FF6B00',
    success: '#10B981',
    error: '#EF4444',
    warning: '#F59E0B',
    surface: isDark ? '#1F1F1F' : '#FFFFFF',
    border: isDark ? '#374151' : '#E5E7EB', 
    bg: isDark ? 'bg-darkbg' : 'bg-secondary',
    cardBg: isDark ? 'bg-darksurface' : 'bg-white',
    optionBg: isDark ? '#2A2A2A' : '#FFFFFF',
  };

  // --- STATE ---
  const [hasPermission, setHasPermission] = useState(false);
  const [gameState, setGameState] = useState<GameState>('menu');
  const [questions, setQuestions] = useState<QuizQuestion[]>([]);
  const [currentQIndex, setCurrentQIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [questionAttempts, setQuestionAttempts] = useState<QuestionAttempt[]>([]);

  // Gameplay State
  const [answerState, setAnswerState] = useState<AnswerState>('idle');
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  
  // Camera / AI State
  const [isDetecting, setIsDetecting] = useState(false);
  const [currentPrediction, setCurrentPrediction] = useState('...');
  const [matchStreak, setMatchStreak] = useState(0);

  // Timer State
  const [timeRemaining, setTimeRemaining] = useState(30);
  const [startTime, setStartTime] = useState<number>(0); 
  const [questionStart, setQuestionStart] = useState<number>(0); 

  // UX State
  const [showTutorial, setShowTutorial] = useState(false);
  const [showHint, setShowHint] = useState(false);

  const [settings, setSettings] = useState<QuizSettings>({
    difficulty: 'beginner',
    questionsCount: 10,
    timeLimit: 25,
    soundEnabled: true,
  });

  // Refs
  const cameraRef = useRef<Camera>(null);
  const progressAnim = useRef(new Animated.Value(0)).current;
  const timerAnim = useRef(new Animated.Value(100)).current;
  const timerInterval = useRef<any>(null);

  // Camera Setup
  const devices = useCameraDevices();
  const device = devices.find((d) => d.position === 'front') ?? devices.find((d) => d.position === 'back');
  const format = useCameraFormat(device, [
    { photoResolution: { width: 640, height: 480 } },
    { fps: 30 }
  ]);

  // Swipe Handlers
  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (_, gestureState) =>
        gameState === 'menu' && Math.abs(gestureState.dx) > 20,
      onPanResponderRelease: (_, gestureState) => {
        if (gestureState.dx < -50) router.push('/profile');
        else if (gestureState.dx > 50) router.push('/learn');
      },
    })
  ).current;

  // --- INIT ---
  useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission();
      setHasPermission(status === 'granted');
      
      const uid = await getCurrentUserId();
      if (uid) {
        const { data } = await supabase
          .from('user_settings')
          .select('sound_effects_enabled')
          .eq('user_id', uid)
          .single();
        if (data) {
          setSettings(prev => ({
            ...prev,
            soundEnabled: data.sound_effects_enabled ?? true,
          }));
        }
      }

      const seen = await AsyncStorage.getItem('quiz_tutorial_seen');
      if (!seen) setShowTutorial(true);
    })();
  }, []);

  useFocusEffect(
    useCallback(() => {
      return () => stopGame();
    }, [])
  );

  const stopGame = () => {
    setGameState('menu');
    setIsDetecting(false);
    if (timerInterval.current) clearInterval(timerInterval.current);
  };

  // --- SOUND & HAPTICS ---
  const feedback = (type: 'success' | 'error' | 'tick') => {
    if (vibrationEnabled) {
      if (type === 'success') Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      else if (type === 'error') Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      else Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }
  };

  // --- QUIZ GENERATION ---
  const allLetters: QuizItem[] = useMemo(() => 
    alphabetSigns.map(l => ({ id: l.letter, label: l.letter, video: l.image, type: 'letter' })), 
  []);

  const allPhrases: QuizItem[] = useMemo(() => 
    phrases.map(p => ({ id: p.id, label: p.text, video: p.videoUrl, type: 'phrase' })), 
  []);

  const generateQuestions = (difficulty: DifficultyLevel): QuizQuestion[] => {
    const count = difficulty === 'beginner' ? 5 : difficulty === 'intermediate' ? 10 : 15;
    
    const fullPool = [...allLetters, ...allPhrases];
    const shuffled = fullPool.sort(() => 0.5 - Math.random()).slice(0, count);
    
    return shuffled.map(item => {
      let type: QuestionType = 'recognition';
      
      if (item.type === 'phrase') {
        type = 'recognition'; 
      } else {
        const r = Math.random();
        if (difficulty === 'beginner') {
          type = r > 0.3 ? 'recognition' : 'performance';
        } else {
          if (r < 0.4) type = 'recognition';
          else if (r < 0.8) type = 'performance';
          else type = 'sequence';
        }
      }

      const q: QuizQuestion = { type, target: item, correctAnswer: item.label };

      if (type === 'recognition') {
        const sameTypeItems = item.type === 'phrase' ? allPhrases : allLetters;
        const distractors = sameTypeItems
          .filter(i => i.id !== item.id)
          .sort(() => 0.5 - Math.random())
          .slice(0, 3);
          
        q.options = [item.label, ...distractors.map(d => d.label)].sort(() => 0.5 - Math.random());
      } 
      else if (type === 'sequence') {
        const seq = [item, ...allLetters.filter(i => i.id !== item.id).sort(()=>0.5-Math.random()).slice(0, 2)];
        q.sequenceTargets = seq;
        q.correctAnswer = seq.map(s => s.label).join('');
        
        const fake1 = seq.slice().reverse().map(s => s.label).join('');
        const fake2 = allLetters.sort(()=>0.5-Math.random()).slice(0, 3).map(s => s.label).join('');
        q.options = [q.correctAnswer, fake1, fake2].sort(() => 0.5 - Math.random());
      }

      return q;
    });
  };

  // --- GAMEPLAY LOGIC ---

  const startGame = (difficulty: DifficultyLevel) => {
    const config = {
      beginner: { time: 25, count: 5 }, 
      intermediate: { time: 20, count: 10 }, 
      advanced: { time: 10, count: 15 }, 
    };
    
    setSettings(prev => ({ ...prev, difficulty, questionsCount: config[difficulty].count, timeLimit: config[difficulty].time }));
    
    const newQs = generateQuestions(difficulty);
    setQuestions(newQs);
    setCurrentQIndex(0);
    setScore(0);
    setQuestionAttempts([]);
    
    setStartTime(Date.now());
    setGameState('playing');
  };

  useEffect(() => {
    if (gameState === 'playing' && questions.length > 0) {
      loadQuestion();
    }
  }, [currentQIndex, gameState, questions]);

  const loadQuestion = () => {
    if (timerInterval.current) clearInterval(timerInterval.current);
    
    setAnswerState('idle');
    setSelectedOption(null);
    setMatchStreak(0);
    setCurrentPrediction('...');
    setQuestionStart(Date.now());
    setShowHint(false);
    
    const q = questions[currentQIndex];
    if (!q) return;

    setIsDetecting(q.type === 'performance');
    
    startTimer(settings.timeLimit);

    Animated.timing(progressAnim, {
      toValue: ((currentQIndex + 1) / questions.length) * 100,
      duration: 500,
      useNativeDriver: false,
    }).start();
  };

  const startTimer = (seconds: number) => {
    setTimeRemaining(seconds);
    timerAnim.setValue(100);

    timerInterval.current = setInterval(() => {
      setTimeRemaining(prev => {
        if (prev <= 1) {
          clearInterval(timerInterval.current);
          handleTimeout(); 
          return 0;
        }
        const newVal = prev - 1;
        const toValue = (newVal / seconds) * 100;
        
        Animated.timing(timerAnim, {
          toValue: isNaN(toValue) ? 0 : toValue, 
          duration: 1000,
          useNativeDriver: false,
        }).start();
        
        if (newVal <= 5) feedback('tick');
        return newVal;
      });
    }, 1000);
  };

  const handleAnswer = (answer: string) => {
    if (answerState !== 'idle') return;
    const correct = answer === questions[currentQIndex].correctAnswer;
    setSelectedOption(answer);
    processResult(correct);
  };

  const handleTimeout = () => {
    setAnswerState('timeout');
    feedback('error');
    processResult(false, true);
  };

  const processResult = (correct: boolean, timeout = false, skipped = false) => {
    clearInterval(timerInterval.current);
    setIsDetecting(false);

    const timeTaken = Math.floor((Date.now() - questionStart) / 1000);
    
    setQuestionAttempts(prev => [...prev, {
      questionIndex: currentQIndex,
      isCorrect: correct,
      timeTaken,
      skipped: skipped
    }]);

    if (correct) {
      setScore(s => s + 1);
      setAnswerState('correct');
      feedback('success');
    } else {
      setAnswerState(timeout ? 'timeout' : 'incorrect');
      if (!skipped) {
        feedback('error'); 
        if (vibrationEnabled && !timeout) Vibration.vibrate(100);
      }
    }

    setTimeout(() => {
      const nextIdx = currentQIndex + 1;
      if (nextIdx < questions.length) {
        setCurrentQIndex(nextIdx); 
      } else {
        endGame();
      }
    }, 1500);
  };

  const endGame = async () => {
    setGameState('summary');
    const durationMinutes = (Date.now() - startTime) / 60000;
    const accuracy = Math.round((score / questions.length) * 100);
    
    const uid = await getCurrentUserId();
    if (uid) {
        await updatePracticeTime(durationMinutes / 60); 
        
        if (accuracy >= 70) {
            await updateStreakOnLessonComplete();
        }
    }
  };

  useEffect(() => {
    if (!isDetecting || gameState !== 'playing' || answerState !== 'idle' || !cameraRef.current) return;

    const loop = setInterval(async () => {
      try {
        if (!cameraRef.current) return; 

        const photo = await cameraRef.current.takePhoto({ flash: 'off', enableShutterSound: false });
        if (!photo) return;

        const uri = `file://${photo.path}`;
        const formData = new FormData();
        formData.append('file', { uri, type: 'image/jpeg', name: 'quiz.jpg' } as any);

        const res = await axios.post(ENDPOINTS.PREDICT, formData, {
           headers: { 'Content-Type': 'multipart/form-data' }, timeout: 1500 
        });

        const pred = res.data.prediction?.toUpperCase() || '...';
        setCurrentPrediction(pred);

        if (pred === questions[currentQIndex].correctAnswer) {
          setMatchStreak(p => {
            const next = p + 1;
            if (next >= REQUIRED_MATCH_STREAK) processResult(true);
            return next;
          });
        } else {
          setMatchStreak(0);
        }
      } catch (e) { /* ignore */ }
    }, DETECTION_INTERVAL);

    return () => clearInterval(loop);
  }, [isDetecting, gameState, answerState, currentQIndex]);


  // --- RENDERERS ---

  const renderMenu = () => (
    <View className="flex-1 justify-center items-center px-6">
      <View className={`w-full p-8 rounded-[32px] shadow-lg border border-accent ${colors.cardBg}`}>
        <View className="items-center mb-6 bg-accent/10 p-6 rounded-full self-center">
          <MaterialIcons name="school" size={60} color={colors.accent} />
        </View>
        <Text className={`text-3xl font-audiowide text-center mb-4`} style={{ color: colors.textColor }}>
          Sign Quiz
        </Text>
        <Text className={`text-center font-montserrat-regular mb-8`} style={{ color: colors.subText }}>
          Test your skills in Letters, Phrases, and Sequence challenges.
        </Text>

        <TouchableOpacity onPress={() => setGameState('difficulty')} className="bg-accent py-4 rounded-full mb-4 shadow-md active:opacity-90">
          <Text className="text-white text-center font-fredoka-bold text-lg">Start Quiz</Text>
        </TouchableOpacity>
        
        <TouchableOpacity onPress={() => setShowTutorial(true)} className={`py-4 rounded-full border-2 ${isDark ? 'border-highlight' : 'border-highlight'}`}>
          <Text className={`text-center font-fredoka-medium`} style={{ color: colors.textColor }}>How to Play</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  const renderDifficulty = () => (
    <View className="flex-1 justify-center px-6">
      <Text className={`text-2xl font-audiowide text-center mb-8`} style={{ color: colors.textColor }}>Select Difficulty</Text>
      
      {(['beginner', 'intermediate', 'advanced'] as DifficultyLevel[]).map(level => (
        <TouchableOpacity 
          key={level} 
          onPress={() => startGame(level)}
          className={`w-full py-5 rounded-2xl border-2 mb-4 items-center ${colors.cardBg} ${isDark ? 'border-gray-700' : 'border-gray-200'}`}
        >
          <Text className={`text-xl font-fredoka-bold capitalize text-accent`}>{level}</Text>
          <Text className="text-gray-500 text-xs mt-1 font-montserrat-regular">
             {level === 'beginner' ? '5 Qs • 25s' : level === 'intermediate' ? '10 Qs • 20s' : '15 Qs • 10s'}
          </Text>
        </TouchableOpacity>
      ))}
      
      <TouchableOpacity onPress={() => setGameState('menu')} className="mt-4 self-center p-2">
        <Text className="text-highlight font-montserrat-medium">Cancel</Text>
      </TouchableOpacity>
    </View>
  );

  const renderPlaying = () => {
    const q = questions[currentQIndex];
    if (!q) return <ActivityIndicator size="large" color={colors.accent} />;
    
    const isPhrase = q.target.type === 'phrase';

    return (
      <View className="flex-1">
        {/* Header */}
        <View className="flex-row items-center justify-between px-6 py-4">
          <TouchableOpacity onPress={() => { stopGame(); router.back(); }} className="p-2">
            <MaterialIcons name="close" size={24} color={colors.textColor} />
          </TouchableOpacity>
          
          <View className="flex-1 mx-4 h-2 bg-gray-200 rounded-full overflow-hidden">
            <Animated.View style={{ flex: 1, backgroundColor: colors.accent, width: progressAnim.interpolate({ inputRange: [0, 100], outputRange: ['0%', '100%'] }) }} />
          </View>

          <View className={`flex-row items-center px-3 py-1 rounded-full ${isDark ? 'bg-gray-800' : 'bg-gray-100'}`}>
            <MaterialIcons name="timer" size={16} color={timeRemaining < 10 ? colors.error : colors.textColor} />
            <Text className={`ml-1 font-fredoka-bold`} style={{ color: timeRemaining < 10 ? colors.error : colors.textColor }}>{timeRemaining}</Text>
          </View>
        </View>

        {/* Content */}
        <ScrollView className="flex-1 px-6" contentContainerStyle={{ paddingBottom: 40 }}>
          
          <Text className={`text-xl font-audiowide text-center mb-6`} style={{ color: colors.textColor }}>
            {q.type === 'performance' ? `Sign: ${q.target.label}` : 
             q.type === 'sequence' ? "Identify the Sequence" : 
             isPhrase ? "Identify the Phrase" : "Identify the Sign"}
          </Text>

          {/* Sequence Mode Visuals */}
          {q.type === 'sequence' && q.sequenceTargets && (
            <View className="flex-row justify-center gap-2 mb-6">
               {q.sequenceTargets.map((item, i) => (
                 <View key={i} className="w-[30%] aspect-[4/5] bg-black rounded-xl overflow-hidden border border-accent shadow-sm">
                   <Video source={item.video} style={{ flex: 1 }} resizeMode={ResizeMode.COVER} isLooping shouldPlay isMuted />
                 </View>
               ))}
            </View>
          )}

          {/* Recognition Mode Visuals */}
          {q.type === 'recognition' && (
            <View className="w-full aspect-video bg-black rounded-2xl overflow-hidden mb-6 shadow-sm border-2 border-accent">
              <Video
                source={q.target.video}
                style={{ flex: 1 }}
                resizeMode={ResizeMode.COVER}
                isLooping shouldPlay isMuted
              />
            </View>
          )}

          {/* Performance Mode Visuals (Camera) */}
          {q.type === 'performance' && (
             <View className="w-full aspect-[3/4] bg-black rounded-3xl overflow-hidden mb-6 relative border-4 border-accent">
               {device && <Camera ref={cameraRef} style={StyleSheet.absoluteFill} device={device} format={format} isActive={isDetecting} photo={true} />}
               
               <View className="absolute bottom-4 left-4 right-4 bg-black/70 p-3 rounded-xl flex-row justify-between items-center backdrop-blur-md">
                 <Text className="text-white font-mono font-bold">Detecting: {currentPrediction}</Text>
                 <View className="flex-row gap-1">
                   {[...Array(REQUIRED_MATCH_STREAK)].map((_, i) => (
                     <View key={i} className={`w-3 h-3 rounded-full ${i < matchStreak ? 'bg-green-400' : 'bg-gray-600'}`} />
                   ))}
                 </View>
               </View>

               {answerState === 'correct' && (
                  <View className="absolute inset-0 bg-black/40 justify-center items-center">
                    <View className="bg-white p-4 rounded-full shadow-lg">
                      <MaterialIcons name="check" size={40} color={colors.success} />
                    </View>
                  </View>
               )}
             </View>
          )}

          {/* Options (Multiple Choice) */}
          {q.type !== 'performance' && (
            <View className="gap-y-3">
              {q.options?.map((opt, i) => {
                const isSelected = selectedOption === opt;
                const isCorrect = q.correctAnswer === opt;
                
                let bg = colors.optionBg;
                let border = colors.border;
                
                if (answerState !== 'idle') {
                  if (isCorrect) { bg = 'rgba(16, 185, 129, 0.15)'; border = colors.success; }
                  else if (isSelected) { bg = 'rgba(239, 68, 68, 0.15)'; border = colors.error; }
                } else if (isSelected) {
                  border = colors.accent;
                }

                return (
                  <TouchableOpacity 
                    key={i} 
                    className={`w-full py-4 rounded-xl border-2 items-center justify-center shadow-sm`}
                    style={{ backgroundColor: bg, borderColor: border }}
                    disabled={answerState !== 'idle'}
                    onPress={() => handleAnswer(opt)}
                  >
                    <Text className={`text-xl font-fredoka-bold text-center`} style={{ color: colors.textColor }}>{opt}</Text>
                  </TouchableOpacity>
                );
              })}
            </View>
          )}

          {/* Controls: Skip & Hint */}
          <View className="mt-6 flex-row justify-center space-x-8">
            <TouchableOpacity 
              onPress={() => setShowHint(true)} 
              className="flex-row items-center opacity-70"
              disabled={answerState !== 'idle'}
            >
              <MaterialIcons name="lightbulb-outline" size={20} color={colors.textColor} />
              <Text className={`ml-2 font-montserrat-medium text-sm`} style={{ color: colors.textColor }}>Hint</Text>
            </TouchableOpacity>

            <TouchableOpacity 
              onPress={() => processResult(false, false, true)} 
              className="flex-row items-center opacity-70"
              disabled={answerState !== 'idle'}
            >
               <MaterialIcons name="skip-next" size={20} color={colors.textColor} />
               <Text className={`ml-2 font-montserrat-medium text-sm underline`} style={{ color: colors.textColor }}>
                 I don't know this one
               </Text>
            </TouchableOpacity>
          </View>

        </ScrollView>
      </View>
    );
  };

  const renderSummary = () => (
    <View className="flex-1 justify-center px-6">
       <View className={`p-8 rounded-[32px] ${colors.cardBg} border border-accent items-center`}>
         <MaterialIcons name={score > questions.length/2 ? "emoji-events" : "trending-up"} size={60} color={colors.accent} style={{marginBottom: 16}} />
         
         <Text className={`text-3xl font-audiowide text-center mb-2`} style={{ color: colors.textColor }}>
           {score > questions.length / 2 ? "Great Job!" : "Keep Practicing"}
         </Text>
         
         <View className="my-6 items-center">
            <Text className={`text-6xl font-fredoka-bold text-accent`}>
              {Math.round((score / questions.length) * 100)}%
            </Text>
            <Text className="text-gray-500 text-sm mt-1 font-montserrat-medium">ACCURACY</Text>
         </View>
         
         <View className="flex-row gap-4 mb-8 w-full">
            <View className={`flex-1 p-4 rounded-2xl items-center ${isDark ? 'bg-gray-800' : 'bg-gray-50'}`}>
               <Text className="text-green-500 font-bold text-xl">{score}</Text>
               <Text className="text-gray-500 text-xs">CORRECT</Text>
            </View>
            <View className={`flex-1 p-4 rounded-2xl items-center ${isDark ? 'bg-gray-800' : 'bg-gray-50'}`}>
               <Text className="text-red-500 font-bold text-xl">{questions.length - score}</Text>
               <Text className="text-gray-500 text-xs">INCORRECT</Text>
            </View>
         </View>

         <TouchableOpacity onPress={() => setGameState('menu')} className="bg-accent w-full py-4 rounded-full shadow-md">
           <Text className="text-white text-center font-fredoka-bold text-lg">Back to Menu</Text>
         </TouchableOpacity>
       </View>
    </View>
  );

  // --- RENDER ---
  return (
    <View className={`flex-1 ${isDark ? 'bg-darkbg' : 'bg-secondary'}`}>
      <ImageBackground source={require('../../assets/images/MainBG.png')} className="flex-1" resizeMode="cover">
        <View className="flex-1" style={{ paddingTop: insets.top }} {...panResponder.panHandlers}>
          {gameState === 'menu' && renderMenu()}
          {gameState === 'difficulty' && renderDifficulty()}
          {gameState === 'playing' && renderPlaying()}
          {gameState === 'summary' && renderSummary()}
        </View>
      </ImageBackground>

      {/* Tutorial Modal */}
      <Modal visible={showTutorial} transparent animationType="fade" onRequestClose={() => setShowTutorial(false)}>
        <View className="flex-1 justify-center items-center bg-black/60 px-6">
          <View className={`p-8 rounded-3xl w-full border border-accent max-w-sm ${colors.cardBg}`}>
            <Text className={`text-2xl font-audiowide mb-6 text-center`} style={{ color: colors.textColor }}>Quiz Rules</Text>
            
            <View className="gap-4 mb-8">
              <View className="flex-row items-center">
                <MaterialIcons name="visibility" size={24} color={colors.accent} />
                <Text className={`ml-3 flex-1`} style={{ color: colors.textColor }}>
                  <Text className="font-bold">Recognition:</Text> Watch the video and identify the correct letter or phrase.
                </Text>
              </View>
              <View className="flex-row items-center">
                <MaterialIcons name="videocam" size={24} color={colors.accent} />
                <Text className={`ml-3 flex-1`} style={{ color: colors.textColor }}>
                  <Text className="font-bold">Performance:</Text> Use your camera to sign the letter shown.
                </Text>
              </View>
              <View className="flex-row items-center">
                <MaterialIcons name="view-week" size={24} color={colors.accent} />
                <Text className={`ml-3 flex-1`} style={{ color: colors.textColor }}>
                  <Text className="font-bold">Sequence:</Text> Identify the sequence of letters shown.
                </Text>
              </View>
            </View>

            <TouchableOpacity 
              onPress={() => { setShowTutorial(false); AsyncStorage.setItem('quiz_tutorial_seen', 'true'); }} 
              className="bg-accent py-3 rounded-full"
            >
              <Text className="text-white text-center font-bold">Got it!</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
      
      {/* Hint Modal */}
      <Modal visible={showHint} transparent animationType="fade" onRequestClose={() => setShowHint(false)}>
          <View className="flex-1 justify-center items-center bg-black/60 px-6">
            <View className={`p-6 rounded-3xl w-full max-w-xs ${colors.cardBg}`}>
              <Text className={`text-xl font-fredoka-bold mb-2 text-center`} style={{ color: colors.textColor }}>Hint</Text>
              <Text className="text-center text-gray-500 mb-6">
                 {questions[currentQIndex]?.type === 'performance' 
                  ? "Ensure your hand is well-lit and centered in the frame." 
                  : "Watch closely for hand shape and movement."}
              </Text>
              <TouchableOpacity onPress={() => setShowHint(false)} className="bg-accent/20 py-3 rounded-full">
                <Text className="text-accent text-center font-bold">Close</Text>
              </TouchableOpacity>
            </View>
          </View>
      </Modal>
    </View>
  );
}