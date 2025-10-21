import React from 'react';
import { View, Text, TouchableOpacity, ScrollView } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import AppHeaderLearn from '../../../components/AppHeaderLearn';

const alphabetData = [
  { letter: 'A', completed: true },
  { letter: 'B', completed: true },
  { letter: 'C', completed: true },
  { letter: 'D', completed: false },
  { letter: 'E', completed: false },
  { letter: 'F', completed: false },
  { letter: 'G', completed: false },
  { letter: 'H', completed: false },
  { letter: 'I', completed: false },
  { letter: 'J', completed: false },
  { letter: 'K', completed: false },
  { letter: 'L', completed: false },
  { letter: 'M', completed: false },
  { letter: 'N', completed: false },
  { letter: 'O', completed: false },
  { letter: 'P', completed: false },
  { letter: 'Q', completed: false },
  { letter: 'R', completed: false },
  { letter: 'S', completed: false },
  { letter: 'T', completed: false },
  { letter: 'U', completed: false },
  { letter: 'V', completed: false },
  { letter: 'W', completed: false },
  { letter: 'X', completed: false },
  { letter: 'Y', completed: false },
  { letter: 'Z', completed: false },
];

const Letters = () => {
  const insets = useSafeAreaInsets();
  const completedCount = alphabetData.filter(item => item.completed).length;

  return (
    <View className="flex-1 bg-gray-100" style={{ paddingTop: insets.top }}>
      <AppHeaderLearn
        title="Learn Letters"
        completedCount={completedCount}
        totalCount={26}
      />

      <ScrollView className="flex-1 p-4" contentContainerStyle={{ paddingBottom: 150 }}>
        
        {/* Title */}
        <Text
          className="text-lg text-gray-800 mb-4"
          style={{ fontFamily: 'Audiowide-Regular' }}
        >
          Select a Letter
        </Text>

        {/* Grid of Letters */}
        <View className="flex-row flex-wrap justify-between mb-1">
          {alphabetData.map((item, index) => (
            <TouchableOpacity
              key={index}
              className={`w-[18%] aspect-square rounded-lg items-center justify-center m-[1%] border-2 ${
                item.completed ? 'border-[#FF6B00] bg-white' : 'border-gray-300 bg-gray-200'
              }`}
            >
              <Text
                style={{
                  fontFamily: 'Fredoka-SemiBold',
                  fontSize: 24,
                  color: item.completed ? '#FF6B00' : '#6B7280',
                }}
              >
                {item.letter}
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
          className="text-lg text-gray-800 mb-4"
          style={{ fontFamily: 'Audiowide-Regular' }}
        >
          Practice: "A"
        </Text>

        {/* Video Placeholder */}
        <View className="w-full aspect-video bg-gray-800 rounded-xl items-center justify-center overflow-hidden mb-6">
          <Text
            className="text-white text-lg"
            style={{ fontFamily: 'Montserrat-SemiBold' }}
          >
            Sign for "A"
          </Text>
        </View>

        {/* Description */}
        <Text
          className="text-lg text-gray-800 mb-2"
          style={{ fontFamily: 'Audiowide-Regular' }}
        >
          Letter: A
        </Text>
        <Text
          className="text-sm text-gray-600 mb-4"
          style={{ fontFamily: 'Montserrat-SemiBold' }}
        >
          Practice the sign for the letter A. Watch the video demonstration and
          mimic the hand position carefully.
        </Text>

        {/* Practice Buttons */}
        <View className="flex-row justify-between mb-4">
          {['Slow Motion', 'Repeat', 'Practice'].map((label) => (
            <TouchableOpacity
              key={label}
              className="flex-1 bg-gray-200 rounded-full py-3 mx-1 items-center border border-accent"
            >
              <Text
                className="text-gray-700"
                style={{ fontFamily: 'Fredoka-Regular' }}
              >
                {label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Completed Button */}
        <TouchableOpacity className="w-full bg-[#FF6B00] rounded-full py-4 items-center shadow-md">
          <Text
            className="text-white text-lg"
            style={{ fontFamily: 'Fredoka-SemiBold' }}
          >
            Completed
          </Text>
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
};

export default Letters;