import React, { useState } from 'react';
import { View, Text, TouchableOpacity, ScrollView, StyleSheet } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { router } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';

// Dummy data for phrases categorized by type
const phraseData = {
  greetings: [
    { phrase: 'Hello', completed: true },
    { phrase: 'Good Morning', completed: true },
    { phrase: 'Good Afternoon', completed: true },
    { phrase: 'Good Evening', completed: false },
    { phrase: 'Goodbye', completed: false },
    { phrase: 'See you later', completed: false },
    { phrase: 'Nice to meet you', completed: false },
  ],
  courtesy: [
    { phrase: 'Thank you', completed: false },
    { phrase: 'You\'re welcome', completed: false },
    { phrase: 'Please', completed: false },
    { phrase: 'Excuse me', completed: false },
  ],
  questions: [
    { phrase: 'How are you?', completed: false },
    { phrase: 'What is your name?', completed: false },
    { phrase: 'Where are you from?', completed: false },
    { phrase: 'What is your favorite color?', completed: false },
  ],
};

const Phrases = () => {
  const insets = useSafeAreaInsets();
  const [activeCategory, setActiveCategory] = useState('greetings');
  const [activePhrase, setActivePhrase] = useState('Hello');

  const allPhrasesCount = Object.values(phraseData).flat().length;
  const completedCount = Object.values(phraseData).flat().filter(item => item.completed).length;

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
          <Text className="text-primary text-xl font-bold">Learn Phrases</Text>
          <Text className="text-primary text-sm mt-1">{completedCount}/{allPhrasesCount} completed</Text>
        </View>
        <View style={{ width: 24 }} />
      </LinearGradient>
      
      {/* Progress Bar */}
      <View className="w-full h-2 bg-white rounded-full">
        <View style={{ width: `${(completedCount / allPhrasesCount) * 100}%` }} className="h-full bg-[#FF6B00] rounded-full" />
      </View>

      <ScrollView className="flex-1 p-4">
        {/* Category Tabs */}
        <Text className="text-lg font-bold text-gray-800 mb-4">Choose Category</Text>
        <View className="flex-row justify-around mb-8 p-1 rounded-full bg-gray-200">
          {Object.keys(phraseData).map((category) => (
            <TouchableOpacity 
              key={category} 
              onPress={() => setActiveCategory(category)}
              className={`flex-1 items-center rounded-full py-2 ${activeCategory === category ? 'bg-[#FF6B00]' : ''}`}
            >
              <Text className={`font-bold ${activeCategory === category ? 'text-white' : 'text-gray-500'}`}>
                {category.charAt(0).toUpperCase() + category.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Phrase List */}
        <Text className="text-lg font-bold text-gray-800 mb-4">{activeCategory.charAt(0).toUpperCase() + activeCategory.slice(1)} Phrases</Text>
        <View className="mb-8">
          {phraseData[activeCategory].map((item, index) => (
            <TouchableOpacity 
              key={index} 
              onPress={() => setActivePhrase(item.phrase)}
              className={`w-full rounded-lg py-4 mb-2 ${activePhrase === item.phrase ? 'bg-[#FF6B00]' : 'bg-gray-200'}`}
            >
              <View className="flex-row items-center pl-4">
                <Text className={`text-xl font-bold ${activePhrase === item.phrase ? 'text-white' : 'text-gray-500'}`}>
                  {item.phrase}
                </Text>
              </View>
            </TouchableOpacity>
          ))}
        </View>

        {/* Video Placeholder */}
        <Text className="text-lg font-bold text-gray-800 mb-4">Practice: "{activePhrase}"</Text>
        <View className="w-full aspect-video bg-gray-800 rounded-xl items-center justify-center overflow-hidden mb-6">
          <Text className="text-white">Sign for "{activePhrase}"</Text>
        </View>

        {/* Practice Section */}
        <Text className="text-lg font-bold text-gray-800 mb-2">Phrase: {activePhrase}</Text>
        <Text className="text-sm text-gray-500 mb-4">Practice this common phrase. Pay attention to facial expressions and hand movements for proper communication.</Text>
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

        <TouchableOpacity className="w-full bg-[#FF6B00] rounded-full py-4 items-center mb-6">
          <Text className="text-white text-lg font-bold">Completed</Text>
        </TouchableOpacity>
        
        {/* Phrase Tips */}
        <Text className="text-lg font-bold text-gray-800 mb-2">Phrase Tips</Text>
        <View className="p-4 bg-gray-200 rounded-lg mb-4">
          <View className="flex-row items-center mb-2">
            <MaterialIcons name="emoji-emotions" size={24} color="#555" />
            <Text className="text-lg font-bold text-gray-800 ml-2">Facial Expressions</Text>
          </View>
          <Text className="text-sm text-gray-500">
            Facial expressions are crucial for conveying meaning in sign language.
          </Text>
        </View>
        <View className="p-4 bg-gray-200 rounded-lg">
          <View className="flex-row items-center mb-2">
            <MaterialIcons name="swap-horiz" size={24} color="#555" />
            <Text className="text-lg font-bold text-gray-800 ml-2">Natural Flow</Text>
          </View>
          <Text className="text-sm text-gray-500">
            Practice smooth transitions between signs for natural communication.
          </Text>
        </View>
      </ScrollView>
    </View>
  );
};

export default Phrases;

const styles = StyleSheet.create({});