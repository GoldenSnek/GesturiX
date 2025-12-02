// File: frontend/app/(tabs)/learn/phrases.tsx
import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { 
  View, 
  Text, 
  TouchableOpacity, 
  ScrollView, 
  ImageBackground,
  Alert,
  Modal
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import AppHeaderLearn from '../../../components/AppHeaderLearn';
import { Video, ResizeMode } from 'expo-av';
import { phrases } from '../../../constants/phrases';
import { 
  markPhraseCompleted, 
  getCompletedPhrases, 
  resetPhraseProgress, 
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
import { useTheme } from '../../../src/ThemeContext';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect, useLocalSearchParams } from 'expo-router';

const CATEGORIES = [
  { key: 'greetings', label: 'Greetings' },
  { key: 'courtesy', label: 'Courtesy' },
  { key: 'questions', label: 'Questions' },
];

const FeatureModal = ({ isVisible, onClose, isDark }: { isVisible: boolean; onClose: () => void; isDark: boolean }) => {
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
            Feature Coming Soon
          </Text>
          <View className="space-y-3">
            <View className="flex-row items-start justify-center">
              <Text className={`text-base font-montserrat-regular text-center leading-6 ${textColor}`}>
                Our AI model is currently learning to recognize phrases efficiently. Stay tuned for updates!
              </Text>
            </View>
          </View>
          <TouchableOpacity onPress={onClose} className="mt-6 p-2 px-6 rounded-full bg-accent/20 self-center">
            <Text className="text-accent text-center font-fredoka-bold">Got it!</Text>
          </TouchableOpacity>
        </View>
      </TouchableOpacity>
    </Modal>
  );
};

export default function PhraseLearnScreen() {
  const insets = useSafeAreaInsets();
  const { isDark } = useTheme();
  const scrollRef = useRef<ScrollView>(null);
  const { initialPhraseId } = useLocalSearchParams<{ initialPhraseId?: string }>();

  const [doneIds, setDoneIds] = useState<string[]>([]);
  const [activeCategory, setActiveCategory] = useState(CATEGORIES[0].key);

  const phrasesForCategory = useMemo(
    () => phrases.filter(p => p.category === activeCategory), [activeCategory]
  );

  const [selectedPhrase, setSelectedPhrase] = useState(phrasesForCategory[0]);
  const [completed, setCompleted] = useState(false);

  const [userSavedItems, setUserSavedItems] = useState<SavedItem[]>([]);
  const [isSaved, setIsSaved] = useState(false);
  const [userId, setUserId] = useState<string | null>(null);

  const [isSlowMotion, setIsSlowMotion] = useState(false);
  const [isRepeating, setIsRepeating] = useState(false);
  const [isFeatureModalVisible, setFeatureModalVisible] = useState(false);

  const bgColorClass = isDark ? 'bg-darkbg' : 'bg-secondary';
  const textColor = isDark ? 'text-secondary' : 'text-primary';

  // 🕒 LEARNING TIME TRACKER
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

  // Load Saved
  useFocusEffect(
    useCallback(() => {
      const loadSaved = async () => {
        const uid = await getCurrentUserId();
        setUserId(uid);
        if (uid) {
          const items = await getUserSavedItems(uid);
          setUserSavedItems(items);
        }
      };
      loadSaved();
    }, [])
  );

  // Check saved status
  useEffect(() => {
    if (selectedPhrase?.id) {
      const found = userSavedItems.find(
        i => i.item_type === 'phrase' && i.item_identifier === selectedPhrase.id
      );
      setIsSaved(!!found);
    }
  }, [selectedPhrase, userSavedItems]);

  const handleToggleSave = async () => {
    if (!userId || !selectedPhrase?.id) return;

    if (isSaved) {
      setIsSaved(false);
      await unsaveItem(userId, 'phrase', selectedPhrase.id);
      const items = await getUserSavedItems(userId);
      setUserSavedItems(items);
    } else {
      setIsSaved(true);
      await saveItem(userId, 'phrase', selectedPhrase.id);
      const items = await getUserSavedItems(userId);
      setUserSavedItems(items);
    }
  };

  // 🧠 1. INITIAL LOAD: Restore state
  useEffect(() => {
    (async () => {
      const uid = await getCurrentUserId();
      const done = await getCompletedPhrases(phrases.map(p => p.id));
      setDoneIds(done);

      // Priority 1: Navigation Param
      if (initialPhraseId) {
        const match = phrases.find(p => p.id === initialPhraseId);
        if (match) {
          setActiveCategory(match.category);
          setSelectedPhrase(match);
          return;
        }
      }

      // Priority 2: Last saved state (User Specific)
      if (uid) {
        const storageKey = `user_${uid}_phrases_last_id`;
        const lastId = await AsyncStorage.getItem(storageKey);
        
        if (lastId) {
          const match = phrases.find(p => p.id === lastId);
          if (match) {
            setActiveCategory(match.category);
            setSelectedPhrase(match);
            return;
          }
        }
      }

      // Priority 3: Default to first
      // FIX: Removed unlocking logic. Defaults to first item of first category.
      setActiveCategory(CATEGORIES[0].key);
      setSelectedPhrase(phrases.filter(p => p.category === CATEGORIES[0].key)[0]);
    })();
  }, [initialPhraseId]);

  // 🧠 2. CATEGORY CHANGE LOGIC
  useEffect(() => {
    if (phrasesForCategory.length > 0) {
      const isPhraseInCurrentCat = phrasesForCategory.find(p => p.id === selectedPhrase?.id);

      if (!isPhraseInCurrentCat) {
        setSelectedPhrase(phrasesForCategory[0]);
        setTimeout(() => {
          if (scrollRef.current) {
            scrollRef.current.scrollTo({ x: 0, animated: true });
          }
        }, 50);
      } else {
        const index = phrasesForCategory.findIndex(p => p.id === selectedPhrase.id);
        if (index > 0) {
          setTimeout(() => {
            if (scrollRef.current) {
              const itemWidth = 90 + 18 * 2; 
              const gap = 12;
              const scrollToX = (itemWidth + gap) * index;
              scrollRef.current.scrollTo({ x: scrollToX, animated: true });
            }
          }, 100);
        }
      }
    }
  }, [activeCategory, phrasesForCategory]);

  // Save state (User Specific)
  useEffect(() => {
    if (selectedPhrase?.id && userId) {
      const storageKey = `user_${userId}_phrases_last_id`;
      AsyncStorage.setItem(storageKey, selectedPhrase.id);
    }
    setCompleted(selectedPhrase && doneIds.includes(selectedPhrase.id));
  }, [selectedPhrase, doneIds, userId]);

  // Progression logic
  const handleComplete = async () => {
    const allPhrasesInCat = phrases.filter(p => p.category === activeCategory);
    const currentIdx = allPhrasesInCat.findIndex(p => p.id === selectedPhrase.id);
    
    await markPhraseCompleted(selectedPhrase.id);
    await updateStreakOnLessonComplete();
    
    const done = await getCompletedPhrases(phrases.map(p => p.id));
    setDoneIds(done);
    setCompleted(true);

    if (currentIdx < allPhrasesInCat.length - 1) {
      setTimeout(() => {
        setSelectedPhrase(allPhrasesInCat[currentIdx + 1]);
        if (scrollRef.current) {
           const itemWidth = 90 + 18 * 2;
           const gap = 12;
           const scrollToX = (itemWidth + gap) * (currentIdx + 1);
           scrollRef.current.scrollTo({ x: scrollToX, animated: true });
        }
      }, 200);

    } else {
      const currentCatIdx = CATEGORIES.findIndex(c => c.key === activeCategory);
      if (currentCatIdx < CATEGORIES.length - 1) {
        setTimeout(() => {
          setActiveCategory(CATEGORIES[currentCatIdx + 1].key);
        }, 200);
      }
    }
  };

  const handleResetProgress = async () => {
    await resetPhraseProgress();
    const done = await getCompletedPhrases(phrases.map(p => p.id));
    setDoneIds(done);
    setCompleted(false);
    
    // Reset view to start
    setActiveCategory(CATEGORIES[0].key);
    setSelectedPhrase(phrases.filter(p => p.category === CATEGORIES[0].key)[0]);
  };

  const handleCameraAlert = () => {
    setFeatureModalVisible(true);
  };

  return (
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
            title="Learn Phrases"
            completedCount={doneIds.length}
            totalCount={phrases.length}
            onResetProgress={handleResetProgress}
          />

          <ScrollView className="flex-1 p-4" contentContainerStyle={{ paddingBottom: 150 }}>
            {/* Category Tabs */}
            <View
              style={{
                flexDirection: 'row',
                justifyContent: 'space-between',
                backgroundColor: isDark ? '#23201C' : '#f9f6f0',
                borderRadius: 28,
                padding: 7,
                marginBottom: 20,
                marginHorizontal: 12,
                borderWidth: 1,
                borderColor: isDark ? '#B5B1A2' : '#E5DDD4',
                shadowColor: '#000',
                shadowOpacity: 0.09,
                shadowOffset: { width: 0, height: 2 },
                shadowRadius: 4,
                elevation: isDark ? 2 : 1,
              }}
            >
              {CATEGORIES.map((cat, i) => (
                <React.Fragment key={cat.key}>
                  <TouchableOpacity
                    onPress={() => setActiveCategory(cat.key)}
                    activeOpacity={0.9}
                    style={{
                      flex: 1,
                      alignItems: 'center',
                      justifyContent: 'center',
                      borderRadius: 18,
                      paddingVertical: 1,
                      // FIX: Use Accent Color for Active Background
                      backgroundColor: activeCategory === cat.key
                        ? '#FF6B00' 
                        : (isDark ? '#292822' : '#FAF3E7'),
                      marginHorizontal: 5,
                      transform: [{ scale: activeCategory === cat.key ? 1.05 : 1.0 }],
                    }}
                  >
                    <Text style={{
                      fontFamily: 'Fredoka-SemiBold',
                      fontSize: 15,
                      // FIX: Use White Text for Active State
                      color: activeCategory === cat.key
                        ? '#FFFFFF'
                        : (isDark ? '#B3B3B3' : '#8A8A8A'),
                      letterSpacing: 1.1,
                    }}>
                      {cat.label}
                    </Text>
                  </TouchableOpacity>
                  {i < CATEGORIES.length - 1 && (
                    <View style={{
                      width: 2,
                      alignSelf: 'stretch',
                      backgroundColor: isDark ? '#B5B1A2' : '#E5DDD4',
                      marginVertical: 6,
                      borderRadius: 99,
                    }} />
                  )}
                </React.Fragment>
              ))}
            </View>

            {/* Horizontal Phrase Selector */}
            <ScrollView
              ref={scrollRef}
              horizontal
              showsHorizontalScrollIndicator={false}
              style={{ marginBottom: 16 }}
              contentContainerStyle={{ gap: 12, paddingRight: 8 }}
            >
              {phrasesForCategory.map((phrase, idx) => {
                const isCompleted = doneIds.includes(phrase.id);
                const isSelected = selectedPhrase.id === phrase.id;

                let backgroundColor, borderColor, textColor;
                if (isCompleted) {
                  backgroundColor = isDark ? '#1e1e1e' : '#faf3ec';
                  borderColor = '#FF6B00';
                  textColor = '#FF6B00';
                } else if (isSelected) {
                  backgroundColor = isDark ? '#FFAB7B' : '#FF6B00';
                  borderColor = '#FF6B00';
                  textColor = isDark ? '#1a1a1a' : '#fff';
                } else {
                  backgroundColor = isDark ? '#242424' : '#FFF3E6';
                  borderColor = isDark ? '#333' : '#ecd7c3';
                  textColor = isDark ? '#ccc' : '#FF6B00';
                }
                return (
                  <TouchableOpacity
                    key={phrase.id}
                    onPress={() => setSelectedPhrase(phrase)}
                    activeOpacity={0.8}
                    // FIX: Removed disabled prop to allow selecting any phrase
                    style={{
                      backgroundColor,
                      paddingVertical: 12,
                      paddingHorizontal: 18,
                      borderRadius: 13,
                      borderWidth: (isCompleted || isSelected) ? 2 : 1,
                      borderColor,
                      minWidth: 90,
                      alignItems: 'center',
                      justifyContent: 'center',
                      // FIX: Removed opacity reduction
                      opacity: 1, 
                      position: 'relative',
                    }}
                  >
                    <Text
                      style={{
                        fontFamily: 'Fredoka-SemiBold',
                        fontSize: 16,
                        color: textColor,
                      }}
                    >
                      {phrase.text}
                    </Text>
                    {isCompleted && (
                      <View style={{ position: 'absolute', top: 5, right: 8 }}>
                        <MaterialIcons name="check-circle" size={18} color="#FF6B00" />
                      </View>
                    )}
                  </TouchableOpacity>
                );
              })}
            </ScrollView>
            
            {/* Selected Phrase View */}
            <Text style={{
              fontSize: 24,
              fontFamily: 'Fredoka-SemiBold',
              color: isDark ? '#fff' : '#1A1A1A',
              marginBottom: 4,
            }}>
              {selectedPhrase.text}
            </Text>
            <View
              style={{
                width: '100%',
                height: 230,
                borderRadius: 20,
                backgroundColor: isDark ? '#222' : '#fffcfa',
                borderWidth: 2,
                // FIX: Main video border is now always Accent color
                borderColor: '#FF6B00',
                shadowColor: isDark ? '#FFB366' : '#FF6B00',
                shadowOffset: { width: 0, height: 3 },
                shadowOpacity: 0.12,
                shadowRadius: 8,
                elevation: 4,
                marginBottom: 16,
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden',
              }}
            >
              <Video
                key={selectedPhrase.id}
                source={selectedPhrase.videoUrl}
                rate={isSlowMotion ? 0.5 : 1.0}
                volume={1.0}
                isMuted={true}
                resizeMode={ResizeMode.COVER}
                shouldPlay={true}
                isLooping={isRepeating}
                useNativeControls
                style={{
                  width: '100%',
                  height: 220,
                  borderRadius: 18,
                  backgroundColor: isDark ? '#181818' : '#f5f5f5',
                }}
              />
            </View>

            {/* Tips Section */}
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
                {' '}{selectedPhrase.tips}
              </Text>
            </Text>

            {/* Controls */}
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
                onPress={handleCameraAlert}
                className={`flex-1 rounded-xl py-2 mx-1 items-center justify-center border border-accent ${
                  isDark ? 'bg-darksurface' : 'bg-lighthover'
                }`}
              >
                <MaterialIcons 
                  name="videocam" 
                  size={20} 
                  color={isDark ? '#F8F8F8' : '#2C2C2C'} 
                  style={{ marginBottom: 2 }}
                />
                <Text
                  className={`text-xs text-center ${textColor}`}
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
              onPress={handleComplete}
              disabled={completed}
              className={`w-full bg-accent rounded-full py-4 items-center shadow-md ${completed ? 'opacity-60' : ''}`}
            >
              <Text
                className="text-secondary text-lg"
                style={{ fontFamily: 'Fredoka-SemiBold' }}
              >
                {completed ? 'Completed!' : 'Mark as Completed'}
              </Text>
            </TouchableOpacity>
          </ScrollView>

          <FeatureModal 
            isVisible={isFeatureModalVisible}
            onClose={() => setFeatureModalVisible(false)}
            isDark={isDark}
          />
        </View>
      </ImageBackground>
    </View>
  );
}