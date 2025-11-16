import React, { useRef, useState, useEffect, useMemo } from 'react';

import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  PanResponder,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useTheme } from '../../../src/ThemeContext';
import AppHeader from '../../../components/AppHeader';

import { phrases } from '../../../constants/phrases';
import { getCompletedPhrases } from '../../../utils/progressStorage';
import { useFocusEffect } from '@react-navigation/native';

// 🟧 Import the Supabase API helper
import { fetchUserStatistics, getCurrentUserId } from '../../../utils/supabaseApi'; // Adjust path if different

// You may want to fetch totals from actual data for other categories too!
const totalPhrases = phrases.length;

const quickActionsData = [
  { id: 'quiz', title: 'Practice Quiz', subtitle: 'Test your knowledge', icon: 'quiz' },
  { id: 'video', title: 'Video Lessons', subtitle: 'Watch and Learn', icon: 'ondemand-video' },
  { id: 'review', title: 'Review', subtitle: 'Practice previous lessons', icon: 'autorenew' },
  { id: 'saved', title: 'Saved Signs', subtitle: 'Your favorites', icon: 'bookmark-outline' },
];

const Learn = () => {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { isDark } = useTheme();

  // NEW: State for phrase progress
  const [phrasesCompleted, setPhrasesCompleted] = useState(0);

  useFocusEffect(
    React.useCallback(() => {
      let isActive = true;
      (async () => {
        const done = await getCompletedPhrases(phrases.map(p => p.id));
        if (isActive) setPhrasesCompleted(done.length);
      })();
      return () => { isActive = false; };
    }, [])
  );

  // 🟧 NEW: State for Supabase statistics
  const [userStats, setUserStats] = useState({
    lessons_completed: 0,
    days_streak: 0,
    practice_hours: 0,
  });

  useFocusEffect(
  React.useCallback(() => {
    let isActive = true;
    async function loadStats() {
      const userId = await getCurrentUserId();
      if (typeof userId !== 'string') return;
      const stats = await fetchUserStatistics(userId);
      if (isActive) setUserStats(stats);
    }
    loadStats();
    return () => { isActive = false; }; // Clean up
  }, [])
);


  // 🟧 Dynamic progress data for the Progress cards
  const progressData = [
    { label: 'Lessons Completed', value: `${userStats.lessons_completed}` },
    { label: 'Streak', value: `${userStats.days_streak} days` },
    { label: 'Learning Time', value: `${(userStats.practice_hours).toFixed(1)} hrs` },
  ];

  const categoriesData = useMemo(() => [
    {
      id: 'letters',
      title: 'Letters',
      subtitle: 'Learn the Alphabet',
      icon: 'text-fields',
      completed: 12,
      total: 26,
      progress: 12 / 26,
    },
    {
      id: 'numbers',
      title: 'Numbers',
      subtitle: 'Count in sign language',
      icon: 'format-list-numbered',
      completed: 12,
      total: 26,
      progress: 12 / 26,
    },
    {
      id: 'phrases',
      title: 'Phrases',
      subtitle: 'Common expressions',
      icon: 'record-voice-over',
      completed: phrasesCompleted,
      total: totalPhrases,
      progress: totalPhrases > 0 ? phrasesCompleted / totalPhrases : 0,
    },
  ], [phrasesCompleted]);

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (_, gestureState) =>
        Math.abs(gestureState.dx) > 10,
      onPanResponderRelease: (_, gestureState) => {
        if (gestureState.dx < -30) router.push('/profile');
        else if (gestureState.dx > 30) router.push('/compose');
      },
    })
  ).current;

  return (
    <View
      {...panResponder.panHandlers}
      className={`flex-1 ${isDark ? 'bg-darkbg' : 'bg-secondary'}`}
      style={{ paddingTop: insets.top }}
    >
      <AppHeader />
      <ScrollView
        className="flex-1 px-4 py-6"
        contentContainerStyle={{ paddingBottom: 150 }}
      >
        {/* 🧠 Progress Section */}
        <Text
          className={`text-2xl mb-4 ${
            isDark ? 'text-secondary' : 'text-primary'
          }`}
          style={{ fontFamily: 'Audiowide-Regular' }}
        >
          Your Progress
        </Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} className="mb-6">
          {progressData.map((item, index) => (
            <View
              key={index}
              className={`rounded-xl p-4 mr-3 items-center justify-center w-32 h-24 shadow-sm border border-accent ${
                isDark ? 'bg-darksurface' : 'bg-secondary'
              }`}
            >
              <Text
                className={`text-2xl ${
                  isDark ? 'text-highlight' : 'text-primary'
                }`}
                style={{ fontFamily: 'Fredoka-SemiBold' }}
              >
                {item.value}
              </Text>
              <Text
                className={`text-xs text-center mt-1 ${
                  isDark ? 'text-neutral' : 'text-neutral'
                }`}
                style={{ fontFamily: 'Montserrat-SemiBold' }}
              >
                {item.label}
              </Text>
            </View>
          ))}
        </ScrollView>

        {/* 🏷️ Category Section */}
        <Text
          className={`text-2xl mb-4 ${
            isDark ? 'text-secondary' : 'text-primary'
          }`}
          style={{ fontFamily: 'Audiowide-Regular' }}
        >
          Choose Category
        </Text>

        {categoriesData.map((category) => (
          <TouchableOpacity
            key={category.id}
            className={`rounded-2xl p-5 mb-4 shadow-md flex-row items-center border-2 border-accent ${
              isDark ? 'bg-darksurface' : 'bg-highlight'
            }`}
            onPress={() => router.push(`/(tabs)/learn/${category.id}` as any)}
          >
            <View
              className={`w-16 h-16 rounded-full items-center justify-center mr-4 ${
                isDark ? 'bg-darkhover' : 'bg-white/30'
              }`}
            >
              <MaterialIcons
                name={category.icon as any}
                size={36}
                color="#FF6B00"
              />
            </View>

            <View className="flex-1">
              <Text
                className={`text-2xl ${
                  isDark ? 'text-secondary' : 'text-darkbg'
                }`}
                style={{ fontFamily: 'Fredoka-SemiBold' }}
              >
                {category.title}
              </Text>
              <Text
                className={`text-sm mb-2 ${
                  isDark ? 'text-neutral' : 'text-darkbg'
                }`}
                style={{ fontFamily: 'Montserrat-SemiBold' }}
              >
                {category.subtitle}
              </Text>
              <View
                className={`w-full h-2 rounded-full ${
                  isDark ? 'bg-darkhover' : 'bg-white/30'
                }`}
              >
                <View
                  style={{ width: `${category.progress * 100}%` }}
                  className="h-full bg-accent rounded-full"
                />
              </View>
              <Text
                className={`text-xs mt-1 ${
                  isDark ? 'text-neutral' : 'text-darkbg'
                }`}
                style={{ fontFamily: 'Montserrat-SemiBold' }}
              >
                {category.completed} of {category.total} completed
              </Text>
            </View>
          </TouchableOpacity>
        ))}

        {/* ⚡ Quick Actions */}
        <Text
          className={`text-2xl mb-4 ${
            isDark ? 'text-secondary' : 'text-primary'
          }`}
          style={{ fontFamily: 'Audiowide-Regular' }}
        >
          Quick Actions
        </Text>

        <View className="flex-row flex-wrap justify-between">
          {quickActionsData.map((action) => (
            <TouchableOpacity
              key={action.id}
              className={`w-[48%] rounded-2xl p-4 mb-4 items-center justify-center h-32 shadow-sm border border-accent ${
                isDark ? 'bg-darksurface' : 'bg-secondary'
              }`}
            >
              <MaterialIcons name={action.icon as any} size={30} color="#FF6B00" />
              <Text
                className={`text-base text-center mt-2 ${
                  isDark ? 'text-highlight' : 'text-primary'
                }`}
                style={{ fontFamily: 'Fredoka-SemiBold' }}
              >
                {action.title}
              </Text>
              <Text
                className={`text-xs text-center mt-1 ${
                  isDark ? 'text-neutral' : 'text-neutral'
                }`}
                style={{ fontFamily: 'Montserrat-SemiBold' }}
              >
                {action.subtitle}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </ScrollView>
    </View>
  );
};

export default Learn;
