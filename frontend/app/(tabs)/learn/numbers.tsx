import React from 'react';
import { View, Text, TouchableOpacity, ScrollView } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import AppHeaderLearn from '../../../components/AppHeaderLearn';

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
  const completedCount = numberData.filter(item => item.completed).length;

  return (
    <View className="flex-1 bg-gray-100" style={{ paddingTop: insets.top }}>
      <AppHeaderLearn
        title="Learn Numbers"
        completedCount={completedCount}
        totalCount={20}
      />

      <ScrollView className="flex-1 p-4" contentContainerStyle={{ paddingBottom: 150 }}>
        {/* Title */}
        <Text
          className="text-lg text-gray-800 mb-4"
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
                item.completed ? 'border-[#FF6B00] bg-white' : 'border-gray-300 bg-gray-200'
              }`}
            >
              <Text
                style={{
                  fontFamily: 'Fredoka-SemiBold',
                  fontSize: 22,
                  color: item.completed ? '#FF6B00' : '#6B7280',
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
          className="text-lg text-gray-800 mb-4"
          style={{ fontFamily: 'Audiowide-Regular' }}
        >
          Practice: "1"
        </Text>

        {/* Video Placeholder */}
        <View className="w-full aspect-video bg-gray-800 rounded-xl items-center justify-center overflow-hidden mb-6">
          <Text
            className="text-white text-lg"
            style={{ fontFamily: 'Montserrat-SemiBold' }}
          >
            Sign for "1"
          </Text>
        </View>

        {/* Description */}
        <Text
          className="text-lg text-gray-800 mb-2"
          style={{ fontFamily: 'Audiowide-Regular' }}
        >
          Number: 1
        </Text>
        <Text
          className="text-sm text-gray-600 mb-4"
          style={{ fontFamily: 'Montserrat-SemiBold' }}
        >
          Practice the sign for number 1. Watch the video demonstration and
          mimic the hand position carefully.
        </Text>

        {/* Buttons */}
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

export default Numbers;