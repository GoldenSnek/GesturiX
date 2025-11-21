// File: frontend/app/(tabs)/learn/numbers.tsx
import React, { useEffect, useState } from 'react';
import { 
  View, 
  Text, 
  TouchableOpacity, 
  ScrollView, 
  ImageBackground,
  Alert
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import AppHeaderLearn from '../../../components/AppHeaderLearn';
import { useTheme } from '../../../src/ThemeContext';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { numbersData } from '../../../constants/numbers';
import { getCompletedNumbers, markNumberCompleted, resetNumberProgress, updateStreakOnLessonComplete } from '../../../utils/progressStorage';
import { Video, ResizeMode } from 'expo-av';

const STORAGE_LAST_NUMBER = 'numberscreen_last_number';

const Numbers = () => {
  const insets = useSafeAreaInsets();
  const { isDark } = useTheme(); 

  // Define base color class for the outer container
  const bgColorClass = isDark ? 'bg-darkbg' : 'bg-secondary';

  const [doneNumbers, setDoneNumbers] = useState<number[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [completed, setCompleted] = useState(false);

  // Load progress
  useEffect(() => {
    (async () => {
      const lastNumStr = await AsyncStorage.getItem(STORAGE_LAST_NUMBER);
      const allNumbers = numbersData.map(n => n.number);
      const done = await getCompletedNumbers(allNumbers);
      setDoneNumbers(done);

      if (lastNumStr) {
        const lastNum = parseInt(lastNumStr, 10);
        const lastIdx = numbersData.findIndex(n => n.number === lastNum);
        if (lastIdx !== -1) {
          setCurrentIdx(lastIdx);
          return;
        }
      }

      // If no last saved state, find first incomplete or default to 0
      const nextIdx = numbersData.findIndex(n => !done.includes(n.number));
      setCurrentIdx(nextIdx === -1 ? 0 : nextIdx);
    })();
  }, []);

  // Save state when index changes
  useEffect(() => {
    if (numbersData[currentIdx]) {
      AsyncStorage.setItem(STORAGE_LAST_NUMBER, numbersData[currentIdx].number.toString());
    }
  }, [currentIdx]);

  // Check completion status
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

    // Auto-advance
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
    Alert.alert(
      "Feature Coming Soon",
      "Our AI model is currently learning to recognize number signs efficiently. Stay tuned for updates!",
      [{ text: "OK" }]
    );
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
            {/* Title */}
            <Text
              className={`text-lg mb-4 ${isDark ? 'text-secondary' : 'text-primary'}`}
              style={{ fontFamily: 'Audiowide-Regular' }}
            >
              Select a Number
            </Text>

            {/* Grid */}
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

            {/* Practice Section */}
            <Text
              className={`text-lg mb-4 ${isDark ? 'text-secondary' : 'text-primary'}`}
              style={{ fontFamily: 'Audiowide-Regular' }}
            >
              Practice: "{currentData.number}"
            </Text>

            {/* Video Container */}
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
                  source={currentData.video}
                  style={{ width: '100%', height: '100%', borderRadius: 18 }}
                  resizeMode={ResizeMode.COVER}
                  shouldPlay={true}
                  isLooping
                  useNativeControls
                  isMuted={true}
                />
              </View>

              {/* "Coming Soon" Camera Button */}
              <TouchableOpacity
                onPress={handleCameraAlert}
                activeOpacity={0.8}
                style={{
                  position: 'absolute',
                  top: 12,
                  right: 12,
                  backgroundColor: '#A8A8A8', // Greyed out to indicate inactive/coming soon
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
                  name="videocam-off"
                  size={20}
                  color="white"
                />
              </TouchableOpacity>
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

            {/* Buttons */}
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

            {/* Completed Button */}
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
        </View>
      </ImageBackground>
    </View>
  );
};

export default Numbers;