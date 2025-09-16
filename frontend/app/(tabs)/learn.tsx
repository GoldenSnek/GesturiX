import React from 'react';
import { View, Text, TouchableOpacity, ScrollView, FlatList } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialIcons } from '@expo/vector-icons';

const alphabetData = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
];

const Learn = () => {
  const insets = useSafeAreaInsets();

  return (
    <View className="flex-1 bg-secondary" style={{ paddingTop: insets.top }}>
      {/* Header */}
      <LinearGradient
        colors={['#FF6B00', '#FFAB7B']}
        className="items-center py-5"
      >
        <Text className="text-primary text-3xl font-bold">GesturiX</Text>
      </LinearGradient>

      <ScrollView className="flex-1 p-4">
        {/* Alphabet Section */}
        <Text className="text-lg font-semibold text-primary mb-2">Learn the Alphabet</Text>
        <FlatList
          data={alphabetData}
          keyExtractor={(item) => item}
          numColumns={4}
          scrollEnabled={false}
          renderItem={({ item }) => (
            <TouchableOpacity className="flex-1 m-1 bg-white rounded-lg aspect-square items-center justify-center shadow-sm">
              <Text className="text-3xl font-bold text-accent">{item}</Text>
            </TouchableOpacity>
          )}
          className="mb-6"
        />

        {/* Common Phrases Section */}
        <Text className="text-lg font-semibold text-primary mb-4">Common Phrases</Text>
        <View className="flex-row flex-wrap mb-6">
          <TouchableOpacity className="bg-white rounded-xl p-4 mr-2 mb-2 shadow-sm flex-row items-center">
            <MaterialIcons name="record-voice-over" size={24} color="#A8A8A8" />
            <View className="ml-2">
              <Text className="font-medium text-primary">Greetings</Text>
              <Text className="text-sm text-neutral">Hello, Goodbye, Good Morning</Text>
            </View>
          </TouchableOpacity>
          <TouchableOpacity className="bg-white rounded-xl p-4 mr-2 mb-2 shadow-sm flex-row items-center">
            <MaterialIcons name="emoji-people" size={24} color="#A8A8A8" />
            <View className="ml-2">
              <Text className="font-medium text-primary">Everyday Questions</Text>
              <Text className="text-sm text-neutral">How are you? What's your name?</Text>
            </View>
          </TouchableOpacity>
        </View>

        {/* Numbers Section */}
        <Text className="text-lg font-semibold text-primary mb-4">Numbers</Text>
        <View className="flex-row flex-wrap mb-6">
          {['1-10', '11-20', '21-50', '51-100'].map((range, index) => (
            <TouchableOpacity
              key={index}
              className="bg-white rounded-xl p-4 mr-2 mb-2 shadow-sm flex-row items-center justify-between flex-1"
            >
              <Text className="font-medium text-primary">{range}</Text>
              <MaterialIcons name="arrow-forward" size={20} color="#A8A8A8" />
            </TouchableOpacity>
          ))}
        </View>
      </ScrollView>
    </View>
  );
};

export default Learn;