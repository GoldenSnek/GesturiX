import React from 'react';
import { 
  View, 
  Text, 
  TouchableOpacity, 
  ScrollView, 
  ImageBackground // 💡 Added ImageBackground import
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import AppHeaderLearn from '../../../components/AppHeaderLearn';
import { useTheme } from '../../../src/ThemeContext';

const numberData = [
  { number: 1, completed: true },
  { number: 2, completed: true },
  { number: 3, completed: true },
  { number: 4, completed: false },
  { number: 5, completed: false },
  { number: 6, completed: false },
  { number: 7, completed: false },
  { number: 8, completed: false },
  { number: 9, completed: false },
  { number: 10, completed: false },
  { number: 11, completed: false },
  { number: 12, completed: false },
  { number: 13, completed: false },
  { number: 14, completed: false },
  { number: 15, completed: false },
  { number: 16, completed: false },
  { number: 17, completed: false },
  { number: 18, completed: false },
  { number: 19, completed: false },
  { number: 20, completed: false },
];

const Numbers = () => {
  const insets = useSafeAreaInsets();
  const { isDark } = useTheme(); 
  const completedCount = numberData.filter((item) => item.completed).length;

  // Define base color class for the outer container
  const bgColorClass = isDark ? 'bg-darkbg' : 'bg-secondary';

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
            title="Learn Numbers"
            completedCount={completedCount}
            totalCount={20}
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
              {numberData.map((item, index) => (
                <TouchableOpacity
                  key={index}
                  className={`w-[18%] aspect-square rounded-lg items-center justify-center m-[1%] border-2 ${
                    item.completed
                      ? 'border-accent bg-secondary'
                      : isDark
                        ? 'border-darkhover bg-darksurface'
                        : 'border-neutral bg-lighthover'
                  }`}
                >
                  <Text
                    style={{
                      fontFamily: 'Fredoka-SemiBold',
                      fontSize: 22,
                      color: item.completed
                        ? '#FF6B00'
                        : isDark
                          ? '#E5E7EB'
                          : '#6B7280',
                    }}
                  >
                    {item.number}
                  </Text>
                  {item.completed && (
                    <View className="absolute top-1 right-1">
                      <MaterialIcons name="check-circle" size={16} color="#FF6B00" />
                    </View>
                  )}
                </TouchableOpacity>
              ))}
            </View>

            {/* Practice Section */}
            <Text
              className={`text-lg mb-4 ${isDark ? 'text-secondary' : 'text-primary'}`}
              style={{ fontFamily: 'Audiowide-Regular' }}
            >
              Practice: "1"
            </Text>

            {/* Video Placeholder */}
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
                marginBottom: 20, // Adjusted margin to match letters.tsx structure
              }}
            >
              <Text
                className="text-secondary text-lg"
                style={{ fontFamily: 'Montserrat-SemiBold' }}
              >
                Sign for "1"
              </Text>
            </View>

            {/* Description */}
            <Text
              className={`text-lg mb-2 ${isDark ? 'text-secondary' : 'text-primary'}`}
              style={{ fontFamily: 'Audiowide-Regular' }}
            >
              Number: 1
            </Text>
            <Text
              className={`text-sm mb-4 ${isDark ? 'text-neutral' : 'text-neutral'}`}
              style={{ fontFamily: 'Montserrat-SemiBold' }}
            >
              Practice the sign for number 1. Watch the video demonstration and mimic
              the hand position carefully.
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
            <TouchableOpacity className="w-full bg-accent rounded-full py-4 items-center shadow-md">
              <Text
                className="text-secondary text-lg"
                style={{ fontFamily: 'Fredoka-SemiBold' }}
              >
                Completed
              </Text>
            </TouchableOpacity>
          </ScrollView>
        </View>
      </ImageBackground>
    </View>
  );
};

export default Numbers;