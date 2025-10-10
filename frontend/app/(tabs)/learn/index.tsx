import React from 'react';
import { View, Text, TouchableOpacity, ScrollView, FlatList, ProgressBarAndroid } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialIcons } from '@expo/vector-icons';
import { router } from 'expo-router';
import AppHeader from '../../../components/AppHeader'; 

// Data for progress and categories
const progressData = [
  { label: 'Lessons Completed', value: '25', unit: 'Lessons Completed' },
  { label: 'Streak', value: '7 days', unit: '' },
  { label: 'Learning Time', value: '3.2 hrs', unit: '' },
];

const categoriesData = [
  {
    id: 'letters',
    title: 'Letters',
    subtitle: 'Learn the Alphabet',
    icon: 'text-fields', // Example icon, choose suitable one
    completed: 12,
    total: 26,
    progress: 0.46,
    color: '#FF6B00'
  },
  {
    id: 'numbers',
    title: 'Numbers',
    subtitle: 'Count in sign language',
    icon: 'format-list-numbered',
    completed: 12,
    total: 26,
    progress: 0.40,
    color: '#FF6B00'
  },
  {
    id: 'phrases',
    title: 'Phrases',
    subtitle: 'Common expressions',
    icon: 'record-voice-over',
    completed: 12,
    total: 26,
    progress: 0.17,
    color: '#FF6B00'
  },
];

const quickActionsData = [
  { id: 'quiz', title: 'Practice Quiz', subtitle: 'Test your knowledge', icon: 'quiz' },
  { id: 'video', title: 'Video Lessons', subtitle: 'Watch and Learn', icon: 'ondemand-video' },
  { id: 'review', title: 'Review', subtitle: 'Practice previous lessons', icon: 'autorenew' },
  { id: 'saved', title: 'Saved Signs', subtitle: 'Your favorites', icon: 'bookmark-outline' },
];


const Learn = () => {
  const insets = useSafeAreaInsets();

  return (
    <View className="flex-1 bg-white" style={{ paddingTop: insets.top }}>
      <AppHeader /> 

      <ScrollView className="flex-1 px-4 py-6">
        {/* Your Progress Section */}
        <Text className="text-xl font-bold text-gray-800 mb-4">Your Progress</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} className="mb-6">
          {progressData.map((item, index) => (
            <View key={index} className="bg-gray-100 rounded-xl p-4 mr-3 items-center justify-center w-32 h-24 shadow-sm border border-gray-200">
              <Text className="text-3x1 font-bold text-gray-800">{item.value}</Text>
              <Text className="text-xs text-gray-500 text-center mt-1">{item.label}</Text>
            </View>
          ))}
        </ScrollView>

        {/* Choose Category Section */}
        <Text className="text-xl font-bold text-gray-800 mb-4">Choose Category</Text>
        <View className="mb-6">
          {categoriesData.map((category) => (
            <TouchableOpacity 
              key={category.id} 
              className="bg-orange-500 rounded-xl p-5 mb-4 shadow-md flex-row items-center"
              onPress={() => router.push(`/(tabs)/learn/${category.id}` as any)}
            >
              <View className="w-16 h-16 rounded-full bg-white/30 items-center justify-center mr-4">
                <MaterialIcons name={category.icon as any} size={36} color="primary" />
              </View>
              <View className="flex-1">
                <Text className="text-2xl font-bold text-primary">{category.title}</Text>
                <Text className="text-sm text-primary mb-2">{category.subtitle}</Text>
                <View className="w-full h-2 bg-white/30 rounded-full">
                  <View style={{ width: `${category.progress * 100}%` }} className="h-full bg-primary rounded-full" />
                </View>
                <Text className="text-xs text-white/70 mt-1">{category.completed} of {category.total} completed</Text>
              </View>
            </TouchableOpacity>
          ))}
        </View>

        {/* Quick Actions Section */}
        <Text className="text-xl font-bold text-gray-800 mb-4">Quick Actions</Text>
        <View className="flex-row flex-wrap justify-between">
          {quickActionsData.map((action) => (
            <TouchableOpacity key={action.id} className="w-[48%] bg-gray-100 rounded-xl p-4 mb-4 items-center justify-center h-32 shadow-sm border border-gray-200">
              <MaterialIcons name={action.icon as any} size={30} color="#FF6B00" className="mb-2" />
              <Text className="text-base font-semibold text-gray-800 text-center">{action.title}</Text>
              <Text className="text-xs text-gray-500 text-center mt-1">{action.subtitle}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </ScrollView>
    </View>
  );
};

export default Learn;