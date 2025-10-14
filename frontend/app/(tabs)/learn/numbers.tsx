import React from 'react';
import { View, Text, TouchableOpacity, ScrollView, StyleSheet } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { router } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';
import AppHeaderLearn from '../../../components/AppHeaderLearn';

// Dummy data for number selection and 'completed' status
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
      totalCount={26}
      />

      <ScrollView className="flex-1 p-4" contentContainerStyle={{paddingBottom: 150,}}>
        {/* Number Selection Grid */}
        <Text className="text-lg font-bold text-gray-800 mb-4">Select a Number</Text>
        <View className="flex-row flex-wrap justify-between mb-1">
          {numberData.map((item, index) => (
            <TouchableOpacity 
              key={index} 
              className={`w-[18%] aspect-square rounded-lg items-center justify-center m-[1%] border-2 ${item.completed ? 'border-[#FF6B00] bg-white' : 'border-gray-300 bg-gray-200'}`}
            >
              <Text className={`text-xl font-bold ${item.completed ? 'text-[#FF6B00]' : 'text-gray-500'}`}>{item.number}</Text>
              {item.completed && (
                <View className="absolute top-1 right-1">
                  <MaterialIcons name="check-circle" size={16} color="#FF6B00" />
                </View>
              )}
            </TouchableOpacity>
          ))}
        </View>

        {/* Video Placeholder */}
        <Text className="text-lg font-bold text-gray-800 mb-4">Practice: "1"</Text>
        <View className="w-full aspect-video bg-gray-800 rounded-xl items-center justify-center overflow-hidden mb-6">
          <Text className="text-white">Sign for "1"</Text>
        </View>

        {/* Practice Section */}
        <Text className="text-lg font-bold text-gray-800 mb-2">Number: 1</Text>
        <Text className="text-sm text-gray-500 mb-4">Practice the sign for number 1. Watch the video demonstration and practice the hand position shown.</Text>
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

export default Numbers;

const styles = StyleSheet.create({});