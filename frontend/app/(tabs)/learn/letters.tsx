import React from 'react';
import { View, Text, TouchableOpacity, ScrollView, StyleSheet } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { router } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';

// Dummy data for letter selection, 'completed' status, and progress
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
      {/* Header with back button */}
      <LinearGradient
        colors={['#FF6B00', '#FFAB7B']}
        className="items-center py-5 flex-row justify-between px-4"
      >
        <TouchableOpacity onPress={() => router.back()}>
          <MaterialIcons name="arrow-back" size={24} color="primary" />
        </TouchableOpacity>
        <View>
          <Text className="text-primary text-xl font-bold">Learn Letters</Text>
          <Text className="text-primary text-sm mt-1">{completedCount}/26 completed</Text>
        </View>
        <View style={{ width: 24 }} />
      </LinearGradient>
      
      {/* Progress Bar */}
      <View className="w-full h-2 bg-white rounded-full">
        <View style={{ width: `${(completedCount / 26) * 100}%` }} className="h-full bg-[#FF6B00] rounded-full" />
      </View>

      <ScrollView className="flex-1 p-4">
        {/* Letter Selection Grid */}
        <Text className="text-lg font-bold text-gray-800 mb-4">Select a Letter</Text>
        <View className="flex-row flex-wrap justify-between mb-1">
          {alphabetData.map((item, index) => (
            <TouchableOpacity 
              key={index} 
              className={`w-[18%] aspect-square rounded-lg items-center justify-center m-[1%] border-2 ${item.completed ? 'border-[#FF6B00] bg-white' : 'border-gray-300 bg-gray-200'}`}
            >
              <Text className={`text-2xl font-bold ${item.completed ? 'text-[#FF6B00]' : 'text-gray-500'}`}>{item.letter}</Text>
              {item.completed && (
                <View className="absolute top-1 right-1">
                  <MaterialIcons name="check-circle" size={16} color="#FF6B00" />
                </View>
              )}
            </TouchableOpacity>
          ))}
        </View>

        {/* Video Placeholder */}
        <Text className="text-lg font-bold text-gray-800 mb-4">Practice: "A"</Text>
        <View className="w-full aspect-video bg-gray-800 rounded-xl items-center justify-center overflow-hidden mb-6">
          <Text className="text-white">Sign for "Hello"</Text>
        </View>

        {/* Practice Section */}
        <Text className="text-lg font-bold text-gray-800 mb-2">Letter: A</Text>
        <Text className="text-sm text-gray-500 mb-4">Practice the sign for letter A. Watch the video demonstration and practice the hand position shown.</Text>
        <View className="flex-row justify-between mb-4">
          <TouchableOpacity className="flex-1 bg-gray-200 rounded-full py-3 mx-1 items-center">
            <Text className="text-gray-600">Slow Motion</Text>
          </TouchableOpacity>
          <TouchableOpacity className="flex-1 bg-gray-200 rounded-full py-3 mx-1 items-center">
            <Text className="text-gray-600">Repeat</Text>
          </TouchableOpacity>
          <TouchableOpacity className="flex-1 bg-gray-200 rounded-full py-3 mx-1 items-center">
            <Text className="text-gray-600">Practice</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity className="w-full bg-[#FF6B00] rounded-full py-4 items-center">
          <Text className="text-white text-lg font-bold">Completed</Text>
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
};

export default Letters;

const styles = StyleSheet.create({});