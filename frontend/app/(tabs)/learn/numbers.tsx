// File: frontend/app/(tabs)/learn/numbers.tsx
import React, { useEffect, useState, useCallback } from 'react';
import { 
  View, 
  Text, 
  TouchableOpacity, 
  ScrollView, 
  ImageBackground,
  Modal 
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import AppHeaderLearn from '../../../components/AppHeaderLearn';
import { useTheme } from '../../../src/ThemeContext';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { numbersData } from '../../../constants/numbers';
import { 
  getCompletedNumbers, 
  markNumberCompleted, 
  resetNumberProgress, 
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
import { Video, ResizeMode } from 'expo-av';
import { useFocusEffect, useLocalSearchParams } from 'expo-router';

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
                Our AI model is currently learning to recognize number signs efficiently. Stay tuned for updates!
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

const Numbers = () => {
  const insets = useSafeAreaInsets();
  const { isDark } = useTheme(); 
  const { initialNumber } = useLocalSearchParams<{ initialNumber?: string }>(); 

  const bgColorClass = isDark ? 'bg-darkbg' : 'bg-secondary';
  const textColor = isDark ? 'text-secondary' : 'text-primary';

  const [doneNumbers, setDoneNumbers] = useState<number[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [completed, setCompleted] = useState(false);

  const [userSavedItems, setUserSavedItems] = useState<SavedItem[]>([]);
  const [isSaved, setIsSaved] = useState(false);
  const [userId, setUserId] = useState<string | null>(null);

  const [isSlowMotion, setIsSlowMotion] = useState(false);
  const [isRepeating, setIsRepeating] = useState(false);
  const [isFeatureModalVisible, setFeatureModalVisible] = useState(false);

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

  // Load user saved items
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
    const currentNum = numbersData[currentIdx]?.number;
    if (currentNum !== undefined) {
      const found = userSavedItems.find(
        i => i.item_type === 'number' && i.item_identifier === currentNum.toString()
      );
      setIsSaved(!!found);
    }
  }, [currentIdx, userSavedItems]);

  const handleToggleSave = async () => {
    const currentNum = numbersData[currentIdx]?.number;
    if (!userId || currentNum === undefined) return;

    if (isSaved) {
      setIsSaved(false);
      await unsaveItem(userId, 'number', currentNum.toString());
      const items = await getUserSavedItems(userId);
      setUserSavedItems(items);
    } else {
      setIsSaved(true);
      await saveItem(userId, 'number', currentNum.toString());
      const items = await getUserSavedItems(userId);
      setUserSavedItems(items);
    }
  };

  useEffect(() => {
    (async () => {
      const uid = await getCurrentUserId();
      const allNumbers = numbersData.map(n => n.number);
      const done = await getCompletedNumbers(allNumbers);
      setDoneNumbers(done);

      // Determine progress ceiling
      const firstUncompletedIdx = numbersData.findIndex(n => !done.includes(n.number));
      const maxAllowedIdx = firstUncompletedIdx === -1 ? 0 : firstUncompletedIdx;

      // Priority 1: Navigation Param
      if (initialNumber) {
        const numVal = parseInt(initialNumber, 10);
        const paramIdx = numbersData.findIndex(n => n.number === numVal);
        if (paramIdx !== -1 && paramIdx <= maxAllowedIdx) {
          setCurrentIdx(paramIdx);
          return;
        }
      }

      // Priority 2: Last saved state (User Specific)
      if (uid) {
        const storageKey = `user_${uid}_numbers_last_idx`;
        const lastNumStr = await AsyncStorage.getItem(storageKey);
        if (lastNumStr) {
          const lastNum = parseInt(lastNumStr, 10);
          const lastIdx = numbersData.findIndex(n => n.number === lastNum);
          if (lastIdx !== -1 && lastIdx <= maxAllowedIdx) {
            setCurrentIdx(lastIdx);
            return;
          }
        }
      }

      // Priority 3: Sequential Default
      setCurrentIdx(maxAllowedIdx);
    })();
  }, [initialNumber]); 

  // Save last state (User Specific)
  useEffect(() => {
    if (numbersData[currentIdx] && userId) {
      const storageKey = `user_${userId}_numbers_last_idx`;
      AsyncStorage.setItem(storageKey, numbersData[currentIdx].number.toString());
    }
  }, [currentIdx, userId]);

  useEffect(() => {
    setCompleted(doneNumbers.includes(numbersData[currentIdx].number));
  }, [doneNumbers, currentIdx]);

  const handleComplete = async () => {
    await markNumberCompleted(numbersData[currentIdx].number);
    await updateStreakOnLessonComplete();
    
    const allNumbers = numbersData.map(n => n.number);
    const done = await getCompletedNumbers(allNumbers);
    setDoneNumbers(done);
    setCompleted(true);

    if (currentIdx < numbersData.length - 1) {
      setTimeout(() => {
        setCurrentIdx(currentIdx + 1);
      }, 200);
    }
  };

  const handleReset = async () => {
    await resetNumberProgress();
    setDoneNumbers([]);
    setCurrentIdx(0);
  };

  const handleCameraAlert = () => {
    setFeatureModalVisible(true);
  };

  const canSelectNumber = (idx: number) =>
    idx === 0 ||
    doneNumbers.includes(numbersData[idx - 1].number) ||
    doneNumbers.includes(numbersData[idx].number);

  const currentData = numbersData[currentIdx];
  const completedCount = doneNumbers.length;

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
            title="Learn Numbers"
            completedCount={completedCount}
            totalCount={numbersData.length}
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
              Select a Number
            </Text>

            <View className="flex-row flex-wrap justify-between mb-1">
              {numbersData.map((item, index) => {
                const isCompleted = doneNumbers.includes(item.number);
                const isSelected = currentIdx === index;
                const canSelect = canSelectNumber(index);

                return (
                  <TouchableOpacity
                    key={item.number}
                    className={`w-[18%] aspect-square rounded-lg items-center justify-center m-[1%] border-2 ${
                      isCompleted
                        ? 'border-accent bg-secondary'
                        : isDark
                          ? 'border-darkhover bg-darksurface'
                          : 'border-neutral bg-lighthover'
                    }`}
                    onPress={() => { if (canSelect) setCurrentIdx(index); }}
                    activeOpacity={canSelect ? 0.7 : 1}
                    disabled={!canSelect}
                    style={isSelected ? { borderWidth: 3, borderColor: '#FF6B00' } : {}}
                  >
                    <Text
                      style={{
                        fontFamily: 'Fredoka-SemiBold',
                        fontSize: 20,
                        color: isCompleted
                          ? '#FF6B00'
                          : canSelect
                            ? (isDark ? '#E5E7EB' : '#6B7280')
                            : (isDark ? '#4B5563' : '#D1D5DB'),
                      }}
                    >
                      {item.number}
                    </Text>
                    {isCompleted && (
                      <View className="absolute top-1 right-1">
                        <MaterialIcons name="check-circle" size={14} color="#FF6B00" />
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
              Practice: "{currentData.number}"
            </Text>

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
                {/* Added key prop to force re-render when currentIdx changes */}
                <Video
                  key={currentIdx}
                  source={currentData.video}
                  style={{ width: '100%', height: '100%', borderRadius: 18 }}
                  resizeMode={ResizeMode.COVER}
                  shouldPlay={true}
                  isLooping={isRepeating}
                  useNativeControls
                  rate={isSlowMotion ? 0.5 : 1.0}
                  isMuted={true}
                />
              </View>
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
                {' '}{currentData.tips}
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
};

export default Numbers;